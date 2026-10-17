# Base Paper Summary & Reference Guide

**Paper Title:** LSD600: the first corpus of biomedical abstracts annotated with lifestyle–disease relations  
**Authors:** Esmaeil Nourani, Evangelia-Mantelena Makri, Xiqing Mao, Sampo Pyysalo, Søren Brunak, Katerina Nastou, Lars Juhl Jensen  
**Journal:** *Database (Oxford)*, 2025, baae129  
**DOI:** [10.1093/database/baae129](https://doi.org/10.1093/database/baae129) | **PMC:** [PMC11756709](https://pmc.ncbi.nlm.nih.gov/articles/PMC11756709/)  
**Zenodo Record:** [10.5281/zenodo.13952449](https://zenodo.org/records/13952449)  
**Official GitHub Repository:** [https://github.com/EsmaeilNourani/LSF_Disease_RE](https://github.com/EsmaeilNourani/LSF_Disease_RE)  

---

## 1. Background and Motivation
- **Lifestyle Factors (LSFs)** (diet, exercise, smoking, sleep, alcohol, stress, pollution, etc.) are central determinants in chronic disease prevention, onset, and management.
- Existing biomedical relation extraction (RE) resources primarily focus on:
  - Gene–Disease relations (e.g., DisGeNET, EU-ADR)
  - Chemical–Disease relations (e.g., BC5CDR)
  - Limited nutrition–disease relations (e.g., ND corpus, FoodDisease)
- **LSD600** is the **first comprehensive corpus** covering a wide spectrum of lifestyle factor categories (9 categories) and 8 distinct relation types with diseases.

---

## 2. Dataset Overview & Splits

| Metric | Train (60%) | Dev (20%) | Test (20%) | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Abstracts** | 360 | 120 | 120 | 600 |
| **Annotated Relations** | 1,140 | 380 | 380 | ~1,900 |
| **Diseases Mentions** | - | - | - | 5,027 |
| **LSF Mentions** | - | - | - | 6,930 |
| **Cross-sentence Relations**| ~16% | ~16% | ~16% | ~16% |

- **Format:** BRAT standoff format (`.txt` and `.ann` files) + Consolidated TSV tables.
- **Inter-Annotator Agreement (IAA):** F1-score of **82.1%**.
- **Cross-sentence Relations:** 16% of all relations span sentence boundaries (much higher than typical BioNLP RE benchmarks which have <5%).

---

## 3. Relation Taxonomy (8 Relation Types)

The relation schema is structured hierarchically into associative and causal/preventive classes:

```
└── Statistical Association
    ├── Positive Statistical Association
    │   └── Causes
    ├── Negative Statistical Association
    │   └── Controls
    │       ├── Prevents
    │       └── Treats
    └── No Statistical Association
```

1. **Statistical Association:** Statistically significant relationship requiring an appropriate control group and statistical test.
2. **Positive Statistical Association:** Positive effect clearly stated (e.g., increased risk).
3. **Causes:** Causal relationship explicitly demonstrated or implied.
4. **Negative Statistical Association:** Negative association clearly stated (e.g., inverse correlation).
5. **Controls:** Beneficial impact of the LSF on the disease.
6. **Prevents:** LSF hinders disease development or occurrence.
7. **Treats:** LSF has an explicit therapeutic or clinical curative effect.
8. **No Statistical Association:** Text explicitly documents the lack/absence of statistical association.

*Note:* All relations are annotated as **undirected (nondirectional)** candidate pairs, framed as a **multi-label classification** task.

---

## 4. Base Model & Experimental Setup

- **Pretrained Backbone:** `RoBERTa-large-PM-M3-Voc` (RoBERTa-large pretrained on PubMed, PMC, and MIMIC-III with a PubMed-specific BPE vocabulary; BioLM by Facebook Research).
- **Entity Masking Technique:** Candidate entity mentions in the abstract are masked using unused tokens (e.g., `<unused1>`, `<unused2>`) to prevent memorization of specific entity names and force contextual reasoning.
- **Classification Head:** Linear projection layer + Sigmoid activation for multi-label binary cross-entropy loss.
- **Best Hyperparameters:**
  - Maximum Sequence Length (MSL): **180**
  - Learning Rate: **5e-5** (reported as 5e-6 in filename / 5e-5 in text)
  - Epochs: **75**
- **Test Set Performance:**
  - **Precision:** 77.3%
  - **Recall:** 61.6%
  - **F1-Score:** 68.5%
  - *Coarse-grained F1 (Positive vs. Negative vs. None):* 73.4%
- **External Zero-Shot Validation:**
  - ND (Nutrition-Disease) Corpus: 70.7% F1
  - FoodDisease Corpus: 80.7% F1

---

## 5. Key Error Analysis & Limitations (Opportunities for New Ideas)

The paper's manual error analysis identified critical bottlenecks in the base system:
1. **Cross-Sentence Relations (40% of all False Negatives):**
   - The base model failed to extract 61 out of 77 cross-sentence relations.
   - Standard transformer windowing and naive entity masking struggle to bridge long-distance dependencies across abstract paragraphs.
2. **Co-reference Resolution & Anaphora:**
   - Mentions referred to via pronouns or determiners ("this factor", "such lifestyle changes") were frequently missed.
3. **Ambiguous & Rare Keywords:**
   - Polysemous terms or rarely encountered phrasing in specialized domains led to false negatives.
4. **Label Confusion between Hierarchical Levels:**
   - 24% of all errors involved confusion between parent/child classes (e.g., *Prevents* confused with *Negative Statistical Association*, or *Causes* confused with *Positive Statistical Association*).
5. **Model Architecture Limitations:**
   - Naive entity masking with standard RoBERTa pooling; no explicit relation hierarchy loss, no graph reasoning, and no prompt-based reasoning.

---

## 6. High-Potential Thesis Extension Directions (2026 Roadmap)

The research plan has transitioned from generic pipeline baselines to two core innovation vectors defined in [MASTER_THESIS_PROPOSALS.md](file:///Users/dzq660/My%20Drive/Students/AtashBar/Project/docs/MASTER_THESIS_PROPOSALS.md):

1. **Option A (Vector 1): Aligned Contrastive Learning (ACL) & Label-Guided Alignment:**
   - Mitigates the 24% label hierarchy confusion by projecting candidate pairs and semantic label definitions into a shared metric space.
   - Plugs directly on top of the current document-level LCoP architecture (`src/training/train_doc_level_lcop.py`).
2. **Option B (Vector 2): Span-Based End-to-End Extraction with Packed Levitated Markers (PL-Marker):**
   - Resolves the pipeline fragility and gold-entity dependency bottleneck, enabling simultaneous NER and RE across full documents.
3. **Option C (Hybrid): Dense-Retrieval Cascade & Metric Learning:**
   - Specifically targets the 40% cross-sentence false negative rate through inter-sentence evidence retrieval and multi-hop metric classification.

*See full details, experimental designs, and comparison matrix in [docs/MASTER_THESIS_PROPOSALS.md](file:///Users/dzq660/My%20Drive/Students/AtashBar/Project/docs/MASTER_THESIS_PROPOSALS.md).*
