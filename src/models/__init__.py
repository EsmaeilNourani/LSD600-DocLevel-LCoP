"""
Model architectures and inference pipelines for Relation Extraction.
"""

from .baseline_roberta import BaselineRobertaRE
from .modular_re_model import ModularRelationExtractor, ModelConfig
from .localized_context_pooling import (
    LocalizedContextPooling,
    DocumentLevelLCoPExtractor,
    LCoPConfig,
)

__all__ = [
    "BaselineRobertaRE",
    "ModularRelationExtractor",
    "ModelConfig",
    "LocalizedContextPooling",
    "DocumentLevelLCoPExtractor",
    "LCoPConfig",
]
