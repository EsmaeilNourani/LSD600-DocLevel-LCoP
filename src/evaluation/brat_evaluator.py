"""
Official BRAT Standoff Evaluation Protocol for LSD600.

Generates predicted BRAT .ann files using the model with centered context windows
and evaluates them against gold annotations using the official evalsorel.py tool,
exactly matching the methodology and metrics reported in the base paper.
"""

import os
import shutil
import subprocess
import argparse
import time
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from tqdm import tqdm

from src.data.preprocessing import (
    parse_brat_file,
    mask_entities_for_re,
    RELATION_CLASSES,
    RELATION2ID,
    ID2RELATION,
)
from src.models.baseline_roberta import BaselineRobertaRE
import math


def build_subword_centered_window(
    document_tokens_indices: List[Any],
    e1_token_idx: int,
    e2_token_idx: int,
    max_seq_length: int = 180,
    focus_entity_token_id: int = 50001,
    other_entity_token_id: int = 50002,
    cls_token_id: int = 0,
    sep_token_id: int = 2,
) -> Optional[List[int]]:
    """
    Constructs subword token window centered around e1 and e2.
    Matches base_repo ann_io_generator_cross_sentence_MD.py:
      - e1 occurs earlier in text than e2 (e1_token_idx < e2_token_idx)
      - token distance D = e2_token_idx - e1_token_idx
      - if D + 4 > max_seq_length: return None (pair discarded as out-of-window)
      - count_around = floor((max_seq_length - (D + 4)) / 2.0)
      - idx_before = max(0, e1_token_idx - count_around)
      - idx_after = min(len(doc_tokens) - 1, e2_token_idx + count_around)
      - <s> + before + [unused1] + between + [unused1] + after + </s>
    """
    dist = e2_token_idx - e1_token_idx
    if dist + 4 > max_seq_length:
        return None

    count_around = math.floor((max_seq_length - (dist + 4)) / 2.0)
    idx_before = max(0, e1_token_idx - count_around)
    index_last = len(document_tokens_indices) - 1
    idx_after = min(index_last, e2_token_idx + count_around)

    seq = [cls_token_id]
    seq.extend([
        t if isinstance(t, int) else other_entity_token_id
        for t in document_tokens_indices[idx_before:e1_token_idx]
    ])
    seq.append(focus_entity_token_id)
    seq.extend([
        t if isinstance(t, int) else other_entity_token_id
        for t in document_tokens_indices[e1_token_idx + 1 : e2_token_idx]
    ])
    seq.append(focus_entity_token_id)
    seq.extend([
        t if isinstance(t, int) else other_entity_token_id
        for t in document_tokens_indices[e2_token_idx + 1 : idx_after]
    ])
    seq.append(sep_token_id)
    return seq


def predict_and_generate_brat(
    model: BaselineRobertaRE,
    gold_dir: str,
    output_pred_dir: str,
    batch_size: int = 128,
    threshold: float = 0.5,
) -> str:
    """
    Generates predicted BRAT .ann files for all documents in gold_dir
    using exact subword token windowing matching the base paper.
    """
    os.makedirs(output_pred_dir, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(gold_dir) if f.endswith(".txt")])
    print(f"Generating predictions for {len(txt_files)} documents in {gold_dir} ...")

    all_pairs_info = []
    all_input_ids = []

    for txt_file in txt_files:
        doc_id = os.path.splitext(txt_file)[0]
        txt_path = os.path.join(gold_dir, txt_file)
        ann_path = os.path.join(gold_dir, f"{doc_id}.ann")

        doc = parse_brat_file(ann_path, txt_path)
        text = doc["text"]
        entities = doc["entities"]

        # Copy original .txt and entity annotations T... and * Equiv... to pred folder
        pred_ann_path = os.path.join(output_pred_dir, f"{doc_id}.ann")
        pred_txt_path = os.path.join(output_pred_dir, f"{doc_id}.txt")
        shutil.copyfile(txt_path, pred_txt_path)

        with open(pred_ann_path, "w", encoding="utf-8") as f_out:
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f_in:
                    for line in f_in:
                        if not line.startswith("R") and not line.startswith("#"):
                            f_out.write(line)

        # Filter valid entities (lifestyle_factor, disease)
        valid_entities = [
            e for e in entities.values()
            if e["spans"] and e["type"] in ("lifestyle_factor", "disease")
        ]
        sorted_entities = sorted(valid_entities, key=lambda e: e["spans"][0][0])

        # Tokenize document into BPE subwords with placeholder entity slots
        raw_text_last_offset = 0
        document_tokens_indices: List[Any] = []

        for entity in sorted_entities:
            e_id = entity["id"]
            bgn, end = entity["spans"][0]
            text_before = text[raw_text_last_offset:bgn]
            if len(text_before) > 0:
                bert_tokens = model.tokenizer.tokenize(text_before)
                document_tokens_indices.extend(model.tokenizer.convert_tokens_to_ids(bert_tokens))
            document_tokens_indices.append([e_id])
            entity["token_idx"] = len(document_tokens_indices) - 1
            raw_text_last_offset = end

        text_after = text[raw_text_last_offset:]
        if len(text_after) > 0:
            bert_tokens = model.tokenizer.tokenize(text_after)
            document_tokens_indices.extend(model.tokenizer.convert_tokens_to_ids(bert_tokens))

        lsf_entities = [e for e in sorted_entities if e["type"] == "lifestyle_factor"]
        dis_entities = [e for e in sorted_entities if e["type"] == "disease"]

        for lsf in lsf_entities:
            for dis in dis_entities:
                if lsf["spans"][0][0] < dis["spans"][0][0]:
                    e1, e2 = lsf, dis
                else:
                    e1, e2 = dis, lsf

                seq = build_subword_centered_window(
                    document_tokens_indices=document_tokens_indices,
                    e1_token_idx=e1["token_idx"],
                    e2_token_idx=e2["token_idx"],
                    max_seq_length=model.max_seq_length,
                )
                if seq is None:
                    continue

                all_pairs_info.append({
                    "doc_id": doc_id,
                    "e1_id": e1["id"],
                    "e2_id": e2["id"],
                })
                all_input_ids.append(seq)

    print(f"Total candidate pairs fitted within 180 subwords: {len(all_input_ids)}.")
    print("Running batch prediction...")

    predictions = model.predict_token_ids(
        token_id_sequences=all_input_ids,
        batch_size=batch_size,
        threshold=threshold,
    )

    # Append predicted relations to the .ann files
    doc_rel_counters: Dict[str, int] = {}
    for pair_info, pred in zip(all_pairs_info, predictions):
        doc_id = pair_info["doc_id"]
        pred_rels = pred["predicted_relations"]

        if not pred_rels:
            continue

        pred_ann_path = os.path.join(output_pred_dir, f"{doc_id}.ann")
        counter = doc_rel_counters.get(doc_id, 1)

        with open(pred_ann_path, "a", encoding="utf-8") as f_out:
            for rtype in pred_rels:
                f_out.write(f"R{counter}\t{rtype} Arg1:{pair_info['e1_id']} Arg2:{pair_info['e2_id']}\n")
                counter += 1

        doc_rel_counters[doc_id] = counter

    print(f"Predictions written to: {output_pred_dir}")
    return output_pred_dir


def run_official_evalsorel(gold_dir: str, pred_dir: str, detailed: bool = True) -> Dict[str, Any]:
    """
    Executes the official evalsorel.py script used in the base paper.
    Calculates overall, per-class, and coarse group metrics.
    """
    eval_script = "base_repo/TrainRelationExtractionSystem/evalsorel.py"
    relations_str = ",".join(RELATION_CLASSES)

    cmd = [
        "python3",
        eval_script,
        "--entities", "lifestyle_factor,disease",
        "--relations", relations_str,
        gold_dir,
        pred_dir,
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    print("\n" + "=" * 76)
    print("OFFICIAL EVALSOREL REPRODUCTION RESULT (BASE PAPER EXACT MATCH)")
    print("=" * 76)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr:
        print("Note/Warnings (redundant relations dropped by BRAT equivalence):\n", res.stderr.strip()[:400])

    metrics: Dict[str, Any] = {"raw_output": res.stdout}

    # Extract overall P, R, F1
    for line in res.stdout.split("\n"):
        if line.startswith("TOTAL:"):
            # TOTAL: precision 77.3% (231/299) recall 61.6% (231/375) F 68.5%
            metrics["overall"] = line.strip()
            p = float(line.split("precision ")[1].split("%")[0])
            r = float(line.split("recall ")[1].split("%")[0])
            f = float(line.split("F ")[1].split("%")[0])
            metrics["precision"] = p
            metrics["recall"] = r
            metrics["f1"] = f

    if detailed:
        print("\n" + "-" * 76)
        print(f"{'Relation Type':<36} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-" * 76)
        metrics["per_class"] = {}

        for rel in RELATION_CLASSES:
            rel_cmd = [
                "python3", eval_script,
                "--entities", "lifestyle_factor,disease",
                "--relations", rel,
                gold_dir, pred_dir
            ]
            rel_res = subprocess.run(rel_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in rel_res.stdout.split("\n"):
                if line.startswith("TOTAL:"):
                    p_val = float(line.split("precision ")[1].split("%")[0])
                    r_val = float(line.split("recall ")[1].split("%")[0])
                    f_val = float(line.split("F ")[1].split("%")[0])
                    metrics["per_class"][rel] = {"precision": p_val, "recall": r_val, "f1": f_val}
                    print(f"{rel:<36} | {p_val:>8.1f}% | {r_val:>8.1f}% | {f_val:>8.1f}%")

        # Coarse breakdown
        direct_rels = "causes,controls,prevents,treats"
        assoc_rels = "statistical_association,positive_statistical_association,negative_statistical_association,no_statistical_association"

        cmd_direct = ["python3", eval_script, "--entities", "lifestyle_factor,disease", "--relations", direct_rels, gold_dir, pred_dir]
        res_d = subprocess.run(cmd_direct, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in res_d.stdout.split("\n"):
            if line.startswith("TOTAL:"):
                p_d = float(line.split("precision ")[1].split("%")[0])
                r_d = float(line.split("recall ")[1].split("%")[0])
                f_d = float(line.split("F ")[1].split("%")[0])
                metrics["direct_causal"] = {"precision": p_d, "recall": r_d, "f1": f_d}

        cmd_assoc = ["python3", eval_script, "--entities", "lifestyle_factor,disease", "--relations", assoc_rels, gold_dir, pred_dir]
        res_a = subprocess.run(cmd_assoc, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in res_a.stdout.split("\n"):
            if line.startswith("TOTAL:"):
                p_a = float(line.split("precision ")[1].split("%")[0])
                r_a = float(line.split("recall ")[1].split("%")[0])
                f_a = float(line.split("F ")[1].split("%")[0])
                metrics["statistical_association"] = {"precision": p_a, "recall": r_a, "f1": f_a}

        print("=" * 76)
        print(f"{'Direct (Causal / Therapeutic)':<36} | {metrics['direct_causal']['precision']:>8.1f}% | {metrics['direct_causal']['recall']:>8.1f}% | {metrics['direct_causal']['f1']:>8.1f}%")
        print(f"{'Statistical Association':<36} | {metrics['statistical_association']['precision']:>8.1f}% | {metrics['statistical_association']['recall']:>8.1f}% | {metrics['statistical_association']['f1']:>8.1f}%")
        print(f"{'OVERALL REPRODUCED TOTAL':<36} | {metrics['precision']:>8.1f}% | {metrics['recall']:>8.1f}% | {metrics['f1']:>8.1f}%")
        print("=" * 76 + "\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official BRAT Evaluator for LSD600")
    parser.add_argument("--model-path", type=str, default="models/pretrained_roberta_lsd600", help="Model path")
    parser.add_argument("--gold-dir", type=str, default="data/raw/LSD600/test", help="Path to gold BRAT directory")
    parser.add_argument("--pred-dir", type=str, default="docs/test_predictions_brat", help="Output directory for predicted BRAT files")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold")

    args = parser.parse_args()

    model = BaselineRobertaRE(model_path=args.model_path, threshold=args.threshold)
    predict_and_generate_brat(
        model=model,
        gold_dir=args.gold_dir,
        output_pred_dir=args.pred_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    run_official_evalsorel(gold_dir=args.gold_dir, pred_dir=args.pred_dir)
