"""
Baseline RoBERTa Model Wrapper for LSD600 Relation Extraction.

Loads the fine-tuned RoBERTa-large-PM-M3-Voc checkpoint from Zenodo
(record 13952449) and provides inference and evaluation interfaces.
"""

import os
from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaForSequenceClassification

from src.data.preprocessing import (
    RELATION_CLASSES,
    RELATION2ID,
    ID2RELATION,
    mask_entities_for_re,
)


class BaselineRobertaRE:
    """
    Wrapper for the best fine-tuned RoBERTa model trained on LSD600.
    """

    def __init__(
        self,
        model_path: str = "models/pretrained_roberta_lsd600",
        device: Optional[str] = None,
        max_seq_length: int = 180,
        threshold: float = 0.5,
    ):
        self.model_path = model_path
        self.max_seq_length = max_seq_length
        self.threshold = threshold

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading tokenizer from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        # Ensure special unused tokens are registered
        unused_tokens = [f"[unused{i}]" for i in range(1, 51)]
        existing_vocab = set(self.tokenizer.get_vocab().keys())
        missing_tokens = [tok for tok in unused_tokens if tok not in existing_vocab]
        if missing_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})

        print(f"Loading model weights from: {self.model_path} onto {self.device}")
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = self.model.config.id2label or ID2RELATION
        self.label2id = self.model.config.label2id or RELATION2ID

    @torch.no_grad()
    def predict_texts(
        self,
        masked_texts: List[str],
        batch_size: int = 16,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on a list of entity-masked texts.

        Args:
            masked_texts: List of strings with [unused1] (focus) and [unused2] (context).
            batch_size: Batch size for tokenization and model inference.
            threshold: Sigmoid probability threshold (defaults to self.threshold).

        Returns:
            list of dicts containing:
                - 'probabilities': dict of class_name -> float probability
                - 'predicted_relations': list of class names with prob >= threshold
                - 'binary_vector': binary numpy array of length 8
        """
        thresh = threshold if threshold is not None else self.threshold
        results = []

        for i in range(0, len(masked_texts), batch_size):
            batch = masked_texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            for p_row in probs:
                prob_dict = {
                    self.id2label.get(idx, str(idx)): float(p_row[idx])
                    for idx in range(len(p_row))
                }
                pred_rels = [
                    self.id2label.get(idx, str(idx))
                    for idx, val in enumerate(p_row)
                    if val >= thresh
                ]
                binary_vec = [1 if val >= thresh else 0 for val in p_row]

                results.append({
                    "probabilities": prob_dict,
                    "predicted_relations": pred_rels,
                    "binary_vector": binary_vec,
                })

        return results

    def predict_candidate_pair(
        self,
        abstract_text: str,
        lsf_span: tuple,
        dis_span: tuple,
        other_spans: Optional[List[tuple]] = None,
    ) -> Dict[str, Any]:
        """
        Predicts relation(s) between a single LSF and Disease mention pair.
        """
        masked = mask_entities_for_re(
            text=abstract_text,
            focus_span_1=lsf_span,
            focus_span_2=dis_span,
            other_spans=other_spans,
        )
        preds = self.predict_texts([masked])
        return preds[0]

    @torch.no_grad()
    def predict_token_ids(
        self,
        token_id_sequences: List[List[int]],
        batch_size: int = 128,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs inference on pre-constructed token ID sequences.
        Directly matches the BPE subword sequence format used in base repo.
        """
        thresh = threshold if threshold is not None else self.threshold
        results = []
        import numpy as np

        for i in range(0, len(token_id_sequences), batch_size):
            batch = token_id_sequences[i : i + batch_size]
            max_len = max(len(seq) for seq in batch)

            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1
            input_ids = np.ones((len(batch), max_len), dtype=np.int64) * pad_id
            attention_mask = np.zeros((len(batch), max_len), dtype=np.int64)

            for j, seq in enumerate(batch):
                input_ids[j, : len(seq)] = seq
                attention_mask[j, : len(seq)] = 1

            b_ids = torch.tensor(input_ids, device=self.device)
            b_mask = torch.tensor(attention_mask, device=self.device)

            outputs = self.model(input_ids=b_ids, attention_mask=b_mask)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()

            for p_row in probs:
                prob_dict = {
                    self.id2label.get(idx, str(idx)): float(p_row[idx])
                    for idx in range(len(p_row))
                }
                pred_rels = [
                    self.id2label.get(idx, str(idx))
                    for idx, val in enumerate(p_row)
                    if val >= thresh
                ]
                binary_vec = [1 if val >= thresh else 0 for val in p_row]
                results.append({
                    "probabilities": prob_dict,
                    "predicted_relations": pred_rels,
                    "binary_vector": binary_vec,
                })

        return results
