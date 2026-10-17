"""
Modular Document-Level Relation Extraction Model.

Supports isolated toggling of:
1. Document-Level / Cross-Sentence Context (on/off)
for clean ablation studies.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaModel, RobertaPreTrainedModel, AutoConfig

from src.data.preprocessing import RELATION_CLASSES, RELATION2ID


@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = "models/pretrained_roberta_lsd600"
    num_classes: int = len(RELATION_CLASSES)
    # Document-Level & Cross-Sentence Modeling
    enable_doc_level: bool = False
    max_seq_length: int = 180           # 180 for baseline, 512 for doc-level
    enable_distance_embedding: bool = False
    max_sentence_distance: int = 10
    # Classifier options
    dropout_prob: float = 0.1
    hidden_dim: int = 1024


class ModularRelationExtractor(nn.Module):
    """
    Modular Relation Extraction Architecture supporting ablation toggles:
    - Baseline (DocLevel=OFF)
    - DocLevel (DocLevel=ON)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.hf_config = AutoConfig.from_pretrained(config.pretrained_model_name_or_path)
        self.roberta = RobertaModel.from_pretrained(config.pretrained_model_name_or_path, config=self.hf_config)

        hidden_size = self.hf_config.hidden_size

        if config.enable_doc_level:
            # Document-Level Bilinear / Multi-feature Classifier:
            # [h_e1; h_e2; |h_e1 - h_e2|; h_e1 * h_e2] -> 4 * hidden_size
            classifier_input_dim = hidden_size * 4
            if config.enable_distance_embedding:
                self.distance_emb = nn.Embedding(config.max_sentence_distance + 1, 64)
                classifier_input_dim += 64
            else:
                self.distance_emb = None

            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_prob),
                nn.Linear(classifier_input_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout_prob),
                nn.Linear(hidden_size, config.num_classes),
            )
        else:
            # Standard sequence classification head on [CLS] token (Baseline)
            self.distance_emb = None
            self.classifier = nn.Sequential(
                nn.Dropout(config.dropout_prob),
                nn.Linear(hidden_size, config.num_classes),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        e1_mask: Optional[torch.Tensor] = None,
        e2_mask: Optional[torch.Tensor] = None,
        sentence_distance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass producing logits of shape (batch_size, num_classes).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, H)

        if self.config.enable_doc_level and e1_mask is not None and e2_mask is not None:
            # Pool entity 1 representation
            e1_lens = torch.clamp(e1_mask.sum(dim=1, keepdim=True), min=1.0)
            h_e1 = (sequence_output * e1_mask.unsqueeze(-1)).sum(dim=1) / e1_lens  # (B, H)

            # Pool entity 2 representation
            e2_lens = torch.clamp(e2_mask.sum(dim=1, keepdim=True), min=1.0)
            h_e2 = (sequence_output * e2_mask.unsqueeze(-1)).sum(dim=1) / e2_lens  # (B, H)

            # Combined multi-feature representation
            diff = torch.abs(h_e1 - h_e2)
            prod = h_e1 * h_e2
            pair_feat = torch.cat([h_e1, h_e2, diff, prod], dim=-1)  # (B, 4*H)

            if self.distance_emb is not None and sentence_distance is not None:
                dist_clamped = torch.clamp(sentence_distance, 0, self.config.max_sentence_distance)
                dist_feat = self.distance_emb(dist_clamped)
                pair_feat = torch.cat([pair_feat, dist_feat], dim=-1)

            logits = self.classifier(pair_feat)
        else:
            # Baseline: pool [CLS] token
            cls_rep = sequence_output[:, 0, :]  # (B, H)
            logits = self.classifier(cls_rep)

        return logits
