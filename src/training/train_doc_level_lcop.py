"""
Training & Evaluation Pipeline for Document-Level Relation Extraction with
Localized Context Pooling (LCoP).

Fine-tunes RoBERTa with LCoP across full abstract sequences (up to 512 tokens),
directly evaluating against official evalsorel.py metrics.
Features:
- Robust Google Colab execution with limited GPU time allocation
- Resumable state checkpoints (latest_checkpoint.pt) so disconnects never lose progress
- Storage-safe saving (prevents Google Drive 15GB disk overflow)
- Official evalsorel.py evaluation (overall and per-class breakdown)
- Early stopping & time-budget guard for limited session runtimes
- Automatic persistent JSON & TXT metric summaries
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import math
import time
import json
import shutil
import tempfile
import argparse
import subprocess
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.models.localized_context_pooling import (
    DocumentLevelLCoPExtractor,
    LCoPConfig,
)
from src.data.preprocessing import RELATION_CLASSES, RELATION2ID


class DocLevelLCoPDataset(Dataset):
    """
    PyTorch Dataset for document-level candidate relation pairs with LCoP.
    Constructs masked sequences up to max_seq_length (default 512) preserving
    full abstract context around lifestyle factor and disease entity mentions.
    """

    def __init__(
        self,
        gold_dir: str,
        tokenizer: AutoTokenizer,
        max_seq_length: int = 512,
        is_training: bool = False,
    ):
        self.gold_dir = gold_dir
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_training = is_training

        self.examples: List[Dict[str, Any]] = []
        self._build_dataset()

    def _build_dataset(self):
        txt_files = sorted([f for f in os.listdir(self.gold_dir) if f.endswith(".txt")])

        for txt_file in txt_files:
            doc_id = os.path.splitext(txt_file)[0]
            txt_path = os.path.join(self.gold_dir, txt_file)
            ann_path = os.path.join(self.gold_dir, f"{doc_id}.ann")

            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            entities = {}
            gold_pairs: Dict[frozenset, List[str]] = {}

            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("T"):
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                t_id = parts[0]
                                tag_spans = parts[1].split()
                                tag = tag_spans[0]
                                if tag in ("lifestyle_factor", "disease"):
                                    bgn = int(tag_spans[1])
                                    end = int(tag_spans[2].split(";")[0])
                                    entities[t_id] = {
                                        "id": t_id,
                                        "tag": tag,
                                        "spans": [(bgn, end)],
                                        "text": parts[2] if len(parts) > 2 else "",
                                    }
                        elif line.startswith("R"):
                            parts = line.split("\t")
                            if len(parts) >= 2:
                                rel_info = parts[1].split()
                                rel_type = rel_info[0]
                                args = dict([arg.split(":") for arg in rel_info[1:]])
                                if "Arg1" in args and "Arg2" in args:
                                    pair_key = frozenset([args["Arg1"], args["Arg2"]])
                                    gold_pairs.setdefault(pair_key, []).append(rel_type)

            sorted_entities = sorted(entities.values(), key=lambda e: e["spans"][0][0])
            raw_text_last_offset = 0
            document_tokens_indices: List[Any] = []

            for entity in sorted_entities:
                e_id = entity["id"]
                bgn, end = entity["spans"][0]
                text_before = text[raw_text_last_offset:bgn]
                if len(text_before) > 0:
                    bert_tokens = self.tokenizer.tokenize(text_before)
                    document_tokens_indices.extend(self.tokenizer.convert_tokens_to_ids(bert_tokens))
                document_tokens_indices.append([e_id])
                entity["token_idx"] = len(document_tokens_indices) - 1
                raw_text_last_offset = end

            text_after = text[raw_text_last_offset:]
            if len(text_after) > 0:
                bert_tokens = self.tokenizer.tokenize(text_after)
                document_tokens_indices.extend(self.tokenizer.convert_tokens_to_ids(bert_tokens))

            lsfs = [e for e in sorted_entities if e["tag"] == "lifestyle_factor"]
            diss = [e for e in sorted_entities if e["tag"] == "disease"]

            for lsf in lsfs:
                for dis in diss:
                    if lsf["spans"][0][0] < dis["spans"][0][0]:
                        e1, e2 = lsf, dis
                    else:
                        e1, e2 = dis, lsf

                    dist = e2["token_idx"] - e1["token_idx"]
                    if dist + 4 > self.max_seq_length:
                        # Exceeds max sequence length even without surrounding context
                        continue

                    # Allocate full available context up to max_seq_length (e.g. 512)
                    count_around = math.floor((self.max_seq_length - (dist + 4)) / 2.0)
                    idx_before = max(0, e1["token_idx"] - count_around)
                    index_last = len(document_tokens_indices) - 1
                    idx_after = min(index_last, e2["token_idx"] + count_around)

                    seq = [self.tokenizer.cls_token_id or 0]
                    seq.extend([
                        t if isinstance(t, int) else 50002
                        for t in document_tokens_indices[idx_before:e1["token_idx"]]
                    ])
                    seq.append(50001)  # Focus entity 1 [unused1]
                    seq.extend([
                        t if isinstance(t, int) else 50002
                        for t in document_tokens_indices[e1["token_idx"] + 1 : e2["token_idx"]]
                    ])
                    seq.append(50001)  # Focus entity 2 [unused1]
                    seq.extend([
                        t if isinstance(t, int) else 50002
                        for t in document_tokens_indices[e2["token_idx"] + 1 : idx_after]
                    ])
                    seq.append(self.tokenizer.sep_token_id or 2)

                    # Multi-hot target vector
                    target = np.zeros(len(RELATION_CLASSES), dtype=np.float32)
                    pair_key = frozenset([e1["id"], e2["id"]])
                    if pair_key in gold_pairs:
                        for rel in gold_pairs[pair_key]:
                            if rel in RELATION2ID:
                                target[RELATION2ID[rel]] = 1.0

                    self.examples.append({
                        "doc_id": doc_id,
                        "e1_id": e1["id"],
                        "e2_id": e2["id"],
                        "input_ids": seq,
                        "labels": target,
                    })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.examples[idx]


def collate_lcop_batch(batch: List[Dict[str, Any]], pad_token_id: int = 1) -> Dict[str, Any]:
    max_len = max(len(item["input_ids"]) for item in batch)
    batch_size = len(batch)

    input_ids = np.ones((batch_size, max_len), dtype=np.int64) * pad_token_id
    attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
    labels = np.zeros((batch_size, len(batch[0]["labels"])), dtype=np.float32)

    for i, item in enumerate(batch):
        seq = item["input_ids"]
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1
        labels[i] = item["labels"]

    return {
        "doc_ids": [item["doc_id"] for item in batch],
        "e1_ids": [item["e1_id"] for item in batch],
        "e2_ids": [item["e2_id"] for item in batch],
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float),
    }


def find_evalsorel_script() -> str:
    """Finds the absolute path to the official evalsorel.py script."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "..", "evaluation", "evalsorel.py"),
        "src/evaluation/evalsorel.py",
        "base_repo/TrainRelationExtractionSystem/evalsorel.py",
        "TrainRelationExtractionSystem/evalsorel.py",
    ]
    for p in candidate_paths:
        abs_p = os.path.abspath(p)
        if os.path.exists(abs_p):
            return abs_p
    raise FileNotFoundError("Could not locate official evalsorel.py script.")


def evaluate_with_official_evalsorel(
    model: DocumentLevelLCoPExtractor,
    dataset: DocLevelLCoPDataset,
    gold_dir: str,
    output_pred_dir: Optional[str] = None,
    device: Union[torch.device, str] = "cuda",
    batch_size: int = 32,
    threshold: float = 0.5,
    detailed: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates DocumentLevelLCoPExtractor by running full-split inference,
    generating BRAT .ann files, and invoking official evalsorel.py.
    """
    if isinstance(device, str):
        device = torch.device(device)

    model.eval()
    eval_script = find_evalsorel_script()

    # Use a fast local temporary directory if output_pred_dir is not explicitly specified
    # or to avoid Google Drive FUSE latency during intermediate per-epoch dev evaluations.
    is_temp_dir = False
    if output_pred_dir is None:
        work_pred_dir = tempfile.mkdtemp(prefix="lsd600_eval_")
        is_temp_dir = True
    else:
        work_pred_dir = output_pred_dir
        if os.path.exists(work_pred_dir):
            try:
                shutil.rmtree(work_pred_dir)
            except Exception:
                pass
        os.makedirs(work_pred_dir, exist_ok=True)

    try:
        # Copy .txt and gold entity annotations (T lines) into work_pred_dir
        for fname in os.listdir(gold_dir):
            if fname.endswith(".txt"):
                shutil.copy(os.path.join(gold_dir, fname), os.path.join(work_pred_dir, fname))
            elif fname.endswith(".ann"):
                src_ann = os.path.join(gold_dir, fname)
                dst_ann = os.path.join(work_pred_dir, fname)
                with open(src_ann, "r", encoding="utf-8") as f_in, open(dst_ann, "w", encoding="utf-8") as f_out:
                    for line in f_in:
                        if line.startswith("T"):
                            f_out.write(line)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_lcop_batch,
        )

        doc_relations: Dict[str, List[Tuple[str, str, str]]] = {}
        use_cuda_fp16 = (device.type == "cuda")

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating split", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                try:
                    autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda_fp16)
                except Exception:
                    autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda_fp16)

                with autocast_ctx:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = out["probabilities"].cpu().numpy()

                for i in range(len(batch["doc_ids"])):
                    doc_id = batch["doc_ids"][i]
                    e1_id = batch["e1_ids"][i]
                    e2_id = batch["e2_ids"][i]
                    p_row = probs[i]

                    for rel_idx, p_val in enumerate(p_row):
                        if p_val >= threshold:
                            rel_name = RELATION_CLASSES[rel_idx]
                            doc_relations.setdefault(doc_id, []).append((rel_name, e1_id, e2_id))

        # Append predicted relations (R1, R2, ...) to the predicted .ann files
        for doc_id, rels in doc_relations.items():
            ann_path = os.path.join(work_pred_dir, f"{doc_id}.ann")
            rel_counter = 1
            with open(ann_path, "a", encoding="utf-8") as f_out:
                for rel_name, e1_id, e2_id in rels:
                    f_out.write(f"R{rel_counter}\t{rel_name} Arg1:{e1_id} Arg2:{e2_id}\n")
                    rel_counter += 1

        # Run official evalsorel.py with all 8 relation classes
        relations_str = ",".join(RELATION_CLASSES)
        cmd = [
            sys.executable,
            eval_script,
            "--entities", "lifestyle_factor,disease",
            "--relations", relations_str,
            gold_dir,
            work_pred_dir,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)

        summary: Dict[str, Any] = {
            "raw_output": res.stdout,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "micro_p": 0.0,
            "micro_r": 0.0,
            "micro_f1": 0.0,
        }

        # Parse TOTAL line from evalsorel stdout:
        # e.g.: TOTAL: precision 77.3% (231/299) recall 61.6% (231/375) F 68.5%
        for line in res.stdout.splitlines():
            line_str = line.strip()
            if line_str.startswith("TOTAL:"):
                try:
                    p = float(line_str.split("precision ")[1].split("%")[0])
                    r = float(line_str.split("recall ")[1].split("%")[0])
                    f = float(line_str.split("F ")[1].split("%")[0])
                    summary["precision"] = p
                    summary["recall"] = r
                    summary["f1"] = f
                    summary["micro_p"] = p
                    summary["micro_r"] = r
                    summary["micro_f1"] = f
                except Exception:
                    pass

        # Per-class breakdown for test / final evaluation
        if detailed:
            summary["per_class"] = {}
            for rel in RELATION_CLASSES:
                rel_cmd = [
                    sys.executable,
                    eval_script,
                    "--entities", "lifestyle_factor,disease",
                    "--relations", rel,
                    gold_dir,
                    work_pred_dir,
                ]
                rel_res = subprocess.run(rel_cmd, capture_output=True, text=True)
                for line in rel_res.stdout.splitlines():
                    if line.strip().startswith("TOTAL:"):
                        try:
                            p_val = float(line.split("precision ")[1].split("%")[0])
                            r_val = float(line.split("recall ")[1].split("%")[0])
                            f_val = float(line.split("F ")[1].split("%")[0])
                            summary["per_class"][rel] = {
                                "precision": p_val,
                                "recall": r_val,
                                "f1": f_val,
                            }
                        except Exception:
                            pass

            # Coarse breakdown: Direct Causal/Therapeutic vs. Statistical Association
            direct_rels = "causes,controls,prevents,treats"
            assoc_rels = "statistical_association,positive_statistical_association,negative_statistical_association,no_statistical_association"

            cmd_direct = [sys.executable, eval_script, "--entities", "lifestyle_factor,disease", "--relations", direct_rels, gold_dir, work_pred_dir]
            res_d = subprocess.run(cmd_direct, capture_output=True, text=True)
            for line in res_d.stdout.splitlines():
                if line.strip().startswith("TOTAL:"):
                    summary["direct_causal"] = {
                        "precision": float(line.split("precision ")[1].split("%")[0]),
                        "recall": float(line.split("recall ")[1].split("%")[0]),
                        "f1": float(line.split("F ")[1].split("%")[0]),
                    }

            cmd_assoc = [sys.executable, eval_script, "--entities", "lifestyle_factor,disease", "--relations", assoc_rels, gold_dir, work_pred_dir]
            res_a = subprocess.run(cmd_assoc, capture_output=True, text=True)
            for line in res_a.stdout.splitlines():
                if line.strip().startswith("TOTAL:"):
                    summary["statistical_association"] = {
                        "precision": float(line.split("precision ")[1].split("%")[0]),
                        "recall": float(line.split("recall ")[1].split("%")[0]),
                        "f1": float(line.split("F ")[1].split("%")[0]),
                    }

        return summary

    finally:
        if is_temp_dir and os.path.exists(work_pred_dir):
            try:
                shutil.rmtree(work_pred_dir)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Train Document-Level LCoP Model on LSD600")
    parser.add_argument("--train-dir", type=str, default="data/raw/LSD600/train")
    parser.add_argument("--dev-dir", type=str, default="data/raw/LSD600/dev")
    parser.add_argument("--test-dir", type=str, default="data/raw/LSD600/test")
    parser.add_argument("--model-path", type=str, default="models/pretrained_roberta_lsd600")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints_doc_level_lcop")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--pooling-mode", type=str, default="hybrid", choices=["attention_bridge", "bilinear_query", "hybrid"])
    parser.add_argument("--start-epoch", type=int, default=None, help="Explicit starting epoch number (e.g. 2 to train only epoch 2)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--fp16", action="store_true", default=True, help="Enable mixed precision on CUDA")
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=True, help="Automatically resume from latest_checkpoint.pt if available")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--save-all-epochs", action="store_true", default=False, help="Save individual checkpoint for every epoch (consumes ~1.4GB per epoch)")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save intra-epoch checkpoint every N steps to avoid losing progress")
    parser.add_argument("--save-interval-minutes", type=float, default=20.0, help="Save intra-epoch checkpoint every N minutes")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader worker processes for faster training")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience (epochs without Dev F1 improvement)")
    parser.add_argument("--max-hours", type=float, default=None, help="Maximum training time budget in hours before graceful early stopping")
    args = parser.parse_args()

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    use_cuda_fp16 = (device.type == "cuda") and args.fp16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda_fp16)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_fp16)

    print(f"Using execution device: {device} (CUDA FP16: {use_cuda_fp16})")
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    # Initialize LCoP Model
    lcop_config = LCoPConfig(
        pretrained_model_name_or_path=args.model_path,
        max_seq_length=args.max_seq_length,
        pooling_mode=args.pooling_mode,
    )
    model = DocumentLevelLCoPExtractor(lcop_config)
    os.makedirs(args.output_dir, exist_ok=True)

    # If eval only
    if args.eval_only:
        ckpt_to_load = args.checkpoint_path or os.path.join(args.output_dir, "best_lcop_model.pt")
        if not os.path.exists(ckpt_to_load):
            latest_candidate = os.path.join(args.output_dir, "latest_checkpoint.pt")
            if os.path.exists(latest_candidate):
                ckpt_to_load = latest_candidate
            else:
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_to_load}. Please ensure model has been trained.")

        print(f"Loading checkpoint for evaluation: {ckpt_to_load}")
        ckpt_obj = torch.load(ckpt_to_load, map_location="cpu")
        if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
            model.load_state_dict(ckpt_obj["model_state_dict"])
        else:
            model.load_state_dict(ckpt_obj)

        model.to(device)

        eval_split = args.test_dir if os.path.exists(args.test_dir) else args.dev_dir
        print(f"Running official evalsorel evaluation on {eval_split}...")
        eval_ds = DocLevelLCoPDataset(eval_split, tokenizer, max_seq_length=args.max_seq_length)
        pred_dir = os.path.join(args.output_dir, "test_predictions_brat")

        res = evaluate_with_official_evalsorel(
            model=model,
            dataset=eval_ds,
            gold_dir=eval_split,
            output_pred_dir=pred_dir,
            device=device,
            batch_size=args.eval_batch_size,
            threshold=args.threshold,
            detailed=True,
        )

        print("\n" + "=" * 78)
        print("OFFICIAL EVALSOREL EVALUATION RESULT")
        print("=" * 78)
        print(res.get("raw_output", "").strip())
        print("=" * 78)
        print(f"OVERALL METRICS -> Precision: {res['precision']:.1f}% | Recall: {res['recall']:.1f}% | F1: {res['f1']:.1f}%")

        if "per_class" in res:
            print("\n" + "-" * 78)
            print(f"{'Relation Type':<36} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            print("-" * 78)
            for rel, m in res["per_class"].items():
                print(f"{rel:<36} | {m['precision']:>8.1f}% | {m['recall']:>8.1f}% | {m['f1']:>8.1f}%")
            print("-" * 78)

        # Save test results JSON
        out_json_path = os.path.join(args.output_dir, "test_results.json")
        with open(out_json_path, "w", encoding="utf-8") as f_out:
            json.dump({
                "precision": res["precision"],
                "recall": res["recall"],
                "f1": res["f1"],
                "per_class": res.get("per_class", {}),
                "direct_causal": res.get("direct_causal", {}),
                "statistical_association": res.get("statistical_association", {}),
            }, f_out, indent=2)
        print(f"Saved evaluation results to: {out_json_path}")
        return

    # Prepare datasets
    print("Preparing datasets with full document context (up to 512 tokens)...")
    train_ds = DocLevelLCoPDataset(args.train_dir, tokenizer, max_seq_length=args.max_seq_length, is_training=True)
    dev_ds = DocLevelLCoPDataset(args.dev_dir, tokenizer, max_seq_length=args.max_seq_length, is_training=False) if os.path.exists(args.dev_dir) else None
    test_ds = DocLevelLCoPDataset(args.test_dir, tokenizer, max_seq_length=args.max_seq_length, is_training=False) if os.path.exists(args.test_dir) else None

    print(f"Dataset candidate pairs: Train={len(train_ds)}, Dev={len(dev_ds) if dev_ds else 0}, Test={len(test_ds) if test_ds else 0}")

    num_workers = args.num_workers if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_lcop_batch,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    effective_batch = args.batch_size * args.grad_accum_steps
    print(f"Batch configuration: micro_batch={args.batch_size}, grad_accum={args.grad_accum_steps} (Effective batch={effective_batch}, Total batches={len(train_loader)})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    start_epoch = 1
    start_step = 0
    best_f1 = 0.0
    history: List[Dict[str, Any]] = []
    history_file = os.path.join(args.output_dir, "training_history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f_h:
                history = json.load(f_h)
                if history:
                    best_f1 = max(h.get("dev_f1", 0.0) for h in history)
        except Exception:
            pass
    epochs_without_improvement = 0

    # Auto-resume logic
    latest_ckpt_path = os.path.join(args.output_dir, "latest_checkpoint.pt")
    resume_target = None

    # Priority: When resuming, prefer latest_checkpoint.pt as it holds exact mid-epoch steps and optimizer states
    if args.resume and os.path.exists(latest_ckpt_path):
        resume_target = latest_ckpt_path
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        resume_target = args.checkpoint_path
    elif os.path.exists(latest_ckpt_path):
        resume_target = latest_ckpt_path

    if resume_target:
        print(f"Resuming training session from: {resume_target}")
        ckpt = torch.load(resume_target, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception as e:
                    print(f"Note: Could not restore optimizer state: {e}")
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as e:
                    print(f"Note: Could not restore scheduler state: {e}")
            if "scaler_state_dict" in ckpt:
                try:
                    scaler.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    pass

            saved_epoch = ckpt.get("epoch", 1)
            saved_step = ckpt.get("step", 0)
            best_f1 = ckpt.get("best_f1", 0.0)
            history = ckpt.get("history", [])

            # Check if this was a mid-epoch checkpoint or an epoch-completion checkpoint
            if saved_step > 0 and saved_step < len(train_loader):
                start_epoch = saved_epoch
                start_step = saved_step
                print(f"✅ Successfully resumed MID-EPOCH! Resuming at Epoch {start_epoch}, Step {start_step}/{len(train_loader)} (Current Best Dev F1: {best_f1:.1f}%)")
            else:
                start_epoch = saved_epoch + 1
                start_step = 0
                print(f"✅ Successfully resumed! Next epoch: {start_epoch}, Current Best Dev F1: {best_f1:.1f}%")
        else:
            model.load_state_dict(ckpt)
            print("Loaded model weights from checkpoint.")

    # Guard: If user explicitly supplied --start-epoch matching the in-progress epoch, keep the mid-epoch step!
    if args.start_epoch is not None:
        if resume_target and args.start_epoch == start_epoch:
            print(f"--> [Mid-Epoch Guard] Keeping mid-epoch resume step {start_step}/{len(train_loader)} for Epoch {start_epoch}")
        else:
            start_epoch = args.start_epoch
            start_step = 0

    model.to(device)

    print("=" * 80)
    print(f"STARTING DOCUMENT-LEVEL LCoP FINE-TUNING (Epochs={start_epoch} to {args.epochs}, LR={args.lr}, Mode={args.pooling_mode})")
    print("=" * 80)

    train_start_time = time.time()
    last_save_time = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        current_start_step = start_step if epoch == start_epoch else 0
        if current_start_step > 0:
            print(f"--> Fast-forwarding to step {current_start_step}/{len(train_loader)} (skipping first {current_start_step} steps already done)...")
            batches_to_run = itertools.islice(train_loader, current_start_step, None)
            pbar = tqdm(batches_to_run, total=len(train_loader), initial=current_start_step, desc=f"Epoch {epoch}/{args.epochs}")
        else:
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(pbar, start=current_start_step):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_cuda_fp16)
            except Exception:
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_cuda_fp16)

            with autocast_ctx:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out["loss"] / args.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            raw_loss = loss.item() * args.grad_accum_steps
            total_loss += raw_loss
            pbar.set_postfix({"loss": f"{raw_loss:.4f}"})

            # Intra-epoch periodic saving to guarantee zero loss of progress
            time_since_save = (time.time() - last_save_time) / 60.0
            should_save_intra_epoch = (
                (args.save_steps and (step + 1) % args.save_steps == 0) or
                (args.save_interval_minutes and time_since_save >= args.save_interval_minutes)
            )

            if should_save_intra_epoch and (step + 1) < len(train_loader):
                mid_ckpt_data = {
                    "epoch": epoch,
                    "step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_f1": best_f1,
                    "history": history,
                }
                torch.save(mid_ckpt_data, latest_ckpt_path)
                last_save_time = time.time()
                tqdm.write(f"\n--> 💾 Auto-saved mid-epoch checkpoint at Epoch {epoch}, Step {step + 1}/{len(train_loader)} ({time_since_save:.1f} mins since last save)")

        start_step = 0  # reset for subsequent epochs

        avg_loss = total_loss / max(1, len(train_loader) - current_start_step)
        epoch_duration = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} finished in {epoch_duration/60:.1f} mins. Average BCE Loss = {avg_loss:.4f}")

        # Evaluate on dev split using official evalsorel.py
        dev_p, dev_r, dev_f1 = 0.0, 0.0, 0.0
        if dev_ds:
            res_dev = evaluate_with_official_evalsorel(
                model=model,
                dataset=dev_ds,
                gold_dir=args.dev_dir,
                device=device,
                batch_size=args.eval_batch_size,
                threshold=args.threshold,
                detailed=False,
            )
            dev_p = res_dev["precision"]
            dev_r = res_dev["recall"]
            dev_f1 = res_dev["f1"]
            print(f"[DEV EVAL Epoch {epoch}] Official evalsorel -> Precision: {dev_p:.1f}%, Recall: {dev_r:.1f}%, F1: {dev_f1:.1f}%")

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                epochs_without_improvement = 0
                best_model_path = os.path.join(args.output_dir, "best_lcop_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"--> ⭐ NEW BEST MODEL SAVED to {best_model_path} (F1 = {best_f1:.1f}%)")
            else:
                epochs_without_improvement += 1
                print(f"Dev F1 did not improve ({dev_f1:.1f}% vs best {best_f1:.1f}%). Patience: {epochs_without_improvement}/{args.patience}")

        # Update training history
        epoch_record = {
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "dev_precision": round(dev_p, 2),
            "dev_recall": round(dev_r, 2),
            "dev_f1": round(dev_f1, 2),
            "best_f1_so_far": round(best_f1, 2),
            "duration_seconds": round(epoch_duration, 1),
        }
        history.append(epoch_record)

        # Save training history JSON
        history_file = os.path.join(args.output_dir, "training_history.json")
        with open(history_file, "w", encoding="utf-8") as f_hist:
            json.dump(history, f_hist, indent=2)

        # Save full resumable checkpoint (overwrites latest_checkpoint.pt to prevent storage overflow)
        latest_checkpoint_data = {
            "epoch": epoch,
            "step": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_f1": best_f1,
            "history": history,
        }
        torch.save(latest_checkpoint_data, latest_ckpt_path)
        last_save_time = time.time()
        print(f"--> Saved resumable checkpoint to {latest_ckpt_path}")

        # Optional: Save individual checkpoint if explicitly requested
        if args.save_all_epochs:
            epoch_ckpt = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save(model.state_dict(), epoch_ckpt)

        # Early stopping check
        if dev_ds and epochs_without_improvement >= args.patience:
            print(f"\n⏹️ Early stopping triggered: no improvement on Dev F1 for {args.patience} epochs.")
            break

        # Time budget check
        if args.max_hours is not None:
            total_elapsed = time.time() - train_start_time
            if total_elapsed >= args.max_hours * 3600:
                print(f"\n⏰ Time budget reached ({total_elapsed/3600:.2f} hrs >= {args.max_hours} hrs). Stopping gracefully to run test evaluation.")
                break

    # Final evaluation on held-out test set
    if test_ds:
        print("\n" + "=" * 80)
        print("RUNNING FINAL EVALUATION ON HELD-OUT TEST SET WITH OFFICIAL EVALSOREL.PY")
        print("=" * 80)
        best_model_path = os.path.join(args.output_dir, "best_lcop_model.pt")
        if os.path.exists(best_model_path):
            print(f"Loading best checkpoint weights: {best_model_path}")
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            print("Using current model weights for test evaluation.")

        pred_dir = os.path.join(args.output_dir, "test_predictions_brat")
        res_test = evaluate_with_official_evalsorel(
            model=model,
            dataset=test_ds,
            gold_dir=args.test_dir,
            output_pred_dir=pred_dir,
            device=device,
            batch_size=args.eval_batch_size,
            threshold=args.threshold,
            detailed=True,
        )

        print("\n" + "=" * 78)
        print("FINAL TEST RESULTS (Official evalsorel.py):")
        print("=" * 78)
        print(res_test.get("raw_output", "").strip())
        print("=" * 78)
        print(f"TEST OVERALL: Precision: {res_test['precision']:.1f}% | Recall: {res_test['recall']:.1f}% | F1: {res_test['f1']:.1f}%")

        if "per_class" in res_test:
            print("\n" + "-" * 78)
            print(f"{'Relation Type':<36} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
            print("-" * 78)
            for rel, m in res_test["per_class"].items():
                print(f"{rel:<36} | {m['precision']:>8.1f}% | {m['recall']:>8.1f}% | {m['f1']:>8.1f}%")
            print("-" * 78)

        # Save test results JSON and TXT report
        test_json_path = os.path.join(args.output_dir, "test_results.json")
        with open(test_json_path, "w", encoding="utf-8") as f_out:
            json.dump({
                "best_dev_f1": best_f1,
                "test_precision": res_test["precision"],
                "test_recall": res_test["recall"],
                "test_f1": res_test["f1"],
                "per_class": res_test.get("per_class", {}),
                "direct_causal": res_test.get("direct_causal", {}),
                "statistical_association": res_test.get("statistical_association", {}),
            }, f_out, indent=2)

        report_txt_path = os.path.join(args.output_dir, "test_evaluation_report.txt")
        with open(report_txt_path, "w", encoding="utf-8") as f_txt:
            f_txt.write("LSD600 Document-Level LCoP Evaluation Report\n")
            f_txt.write("=" * 60 + "\n")
            f_txt.write(f"Best Dev F1: {best_f1:.1f}%\n")
            f_txt.write(f"Test Precision: {res_test['precision']:.1f}%\n")
            f_txt.write(f"Test Recall: {res_test['recall']:.1f}%\n")
            f_txt.write(f"Test F1: {res_test['f1']:.1f}%\n\n")
            f_txt.write("Raw evalsorel output:\n")
            f_txt.write(res_test.get("raw_output", "") + "\n")

        print(f"\n✅ All results and evaluation reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
