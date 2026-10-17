"""
Evaluation Pipeline for LSD600 Relation Extraction with Live Progress Monitoring.

Features:
- Live batch progress monitoring with throughput (pairs/sec), elapsed time, and ETA.
- Sample benchmarking mode to estimate runtime before executing full evaluation.
- Intermediate checkpointing to resume instantly if interrupted.
- Fine-grained (8 classes) and coarse-grained evaluation metrics.
"""

import os
import argparse
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch

from src.data.preprocessing import (
    load_brat_dataset,
    RELATION_CLASSES,
    RELATION2ID,
    ID2RELATION,
)
from src.evaluation.metrics import (
    compute_multilabel_metrics,
    compute_coarse_metrics,
    format_metrics_table,
)
from src.models.baseline_roberta import BaselineRobertaRE


def evaluate_model_on_brat(
    model: BaselineRobertaRE,
    data_dir: str,
    batch_size: int = 128,
    threshold: float = 0.5,
    sample_limit: Optional[int] = None,
    checkpoint_file: str = "docs/eval_checkpoint.pt",
    save_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluates model on a BRAT dataset directory with live monitoring and checkpointing.
    """
    print("=" * 70)
    print("LSD600 MODEL EVALUATION & MONITORING")
    print("=" * 70)
    print(f"Data directory:    {data_dir}")
    print(f"Device:            {model.device}")
    print(f"Batch size:        {batch_size}")
    print(f"Decision threshold:{threshold}")
    
    t0 = time.time()
    candidates = load_brat_dataset(data_dir)
    if sample_limit and sample_limit < len(candidates):
        print(f"Note: Running in benchmark mode limited to first {sample_limit} candidate pairs.")
        candidates = candidates[:sample_limit]

    total_candidates = len(candidates)
    positive_candidates = [c for c in candidates if c["has_relation"]]
    print(f"Total candidate pairs: {total_candidates}")
    print(f"Ground truth positive relation pairs: {len(positive_candidates)}")
    print(f"Dataset preparation time: {time.time()-t0:.2f}s")
    print("=" * 70)

    # Check for existing checkpoint
    predictions = []
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            ckpt = torch.load(checkpoint_file)
            if ckpt.get("total_candidates") == total_candidates:
                predictions = ckpt.get("predictions", [])
                start_idx = len(predictions)
                print(f"Found existing checkpoint! Resuming from pair {start_idx}/{total_candidates} ({start_idx*100/total_candidates:.1f}%)...")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting fresh.")

    masked_texts = [c["masked_text"] for c in candidates]
    y_true = np.array([c["labels"] for c in candidates], dtype=int)

    total_batches = (total_candidates + batch_size - 1) // batch_size
    current_batch = start_idx // batch_size

    print(f"\nStarting inference: {total_batches - current_batch} remaining batches...")
    t_start = time.time()

    for idx in range(start_idx, total_candidates, batch_size):
        b_num = idx // batch_size + 1
        batch = masked_texts[idx : idx + batch_size]
        t_b0 = time.time()
        batch_preds = model.predict_texts(
            masked_texts=batch,
            batch_size=batch_size,
            threshold=threshold,
        )
        predictions.extend(batch_preds)
        t_b_cost = time.time() - t_b0

        # Periodic status update & checkpoint every 5 batches or on final batch
        is_final = (idx + batch_size) >= total_candidates
        if b_num % 5 == 0 or is_final:
            elapsed = time.time() - t_start
            done_count = len(predictions) - start_idx
            speed = done_count / max(elapsed, 0.001)
            pct = (len(predictions) / total_candidates) * 100
            remaining_pairs = total_candidates - len(predictions)
            eta_sec = remaining_pairs / max(speed, 0.001) if not is_final else 0.0

            eta_min = int(eta_sec // 60)
            eta_rem_sec = int(eta_sec % 60)
            print(
                f"[Batch {b_num:3d}/{total_batches:3d} ({pct:5.1f}%)] "
                f"Pairs: {len(predictions):5d}/{total_candidates:5d} | "
                f"Speed: {speed:5.1f} pairs/s ({t_b_cost:.2f}s/batch) | "
                f"Elapsed: {elapsed/60:4.1f}m | ETA: {eta_min:02d}m {eta_rem_sec:02d}s"
            )

            # Save checkpoint
            os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
            torch.save({
                "total_candidates": total_candidates,
                "predictions": predictions,
            }, checkpoint_file)

    total_inf_time = time.time() - t_start
    print(f"\nInference complete! Total inference time: {total_inf_time/60:.2f} minutes.")

    # Remove checkpoint upon completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    y_pred = np.array([p["binary_vector"] for p in predictions], dtype=int)

    metrics_fine = compute_multilabel_metrics(y_true, y_pred, class_names=RELATION_CLASSES)
    metrics_coarse = compute_coarse_metrics(y_true, y_pred)

    print("\n" + "=" * 70)
    print("REPRODUCED RESULTS: FINE-GRAINED RELATION EXTRACTION (8 CLASSES)")
    print("=" * 70)
    print(format_metrics_table(metrics_fine))

    print("\n" + "=" * 70)
    print("REPRODUCED RESULTS: COARSE-GRAINED (Positive / Negative / Assoc / None)")
    print("=" * 70)
    print(format_metrics_table(metrics_coarse))

    report = {
        "fine_grained": metrics_fine,
        "coarse_grained": metrics_coarse,
        "num_candidates": total_candidates,
        "num_positives": len(positive_candidates),
        "threshold": threshold,
        "total_inference_time_sec": total_inf_time,
    }

    if save_output_path:
        os.makedirs(os.path.dirname(save_output_path) or ".", exist_ok=True)
        with open(save_output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nDetailed reproduction report saved to: {save_output_path}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Relation Extraction Model on LSD600")
    parser.add_argument("--model-path", type=str, default="models/pretrained_roberta_lsd600", help="Path to model directory")
    parser.add_argument("--data-dir", type=str, default="data/raw/LSD600/test", help="Path to BRAT data directory")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional limit on number of candidate pairs for quick benchmarking")
    parser.add_argument("--checkpoint-file", type=str, default="docs/eval_checkpoint.pt", help="Checkpoint file path")
    parser.add_argument("--save-output", type=str, default="docs/reproduced_baseline_results.json", help="Path to save output JSON")

    args = parser.parse_args()

    model = BaselineRobertaRE(model_path=args.model_path, threshold=args.threshold)
    evaluate_model_on_brat(
        model=model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        threshold=args.threshold,
        sample_limit=args.sample_limit,
        checkpoint_file=args.checkpoint_file,
        save_output_path=args.save_output,
    )
