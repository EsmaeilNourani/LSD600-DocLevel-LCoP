# Lifestyle–Disease Relation Extraction (LSD600)

**Master's Thesis Project: Advanced Biomedical Relation Extraction on the LSD600 Corpus**

This repository contains the codebase, dataset resources, baseline model wrappers, and experimental framework for research on extracting relations between **Lifestyle Factors (LSFs)** and **Diseases** from biomedical literature.

---

## 1. Project Overview & Base Paper

- **Base Paper:** *LSD600: the first corpus of biomedical abstracts annotated with lifestyle–disease relations*
- **Journal:** *Database (Oxford)*, 2025, baae129
- **PubMed Central:** [PMC11756709](https://pmc.ncbi.nlm.nih.gov/articles/PMC11756709/)
- **Zenodo Repository:** [Zenodo Record 13952449](https://zenodo.org/records/13952449)
- **Base Authors:** Esmaeil Nourani, Evangelia-Mantelena Makri, Xiqing Mao, Sampo Pyysalo, Søren Brunak, Katerina Nastou, Lars Juhl Jensen (University of Copenhagen & University of Turku)

### The Research Problem
Lifestyle factors (such as diet, smoking, sleep quality, physical activity, alcohol consumption, and air pollution) play a decisive role in disease prevention, etiology, and therapy. However, biomedical literature mining has historically focused almost exclusively on gene-disease or chemical-disease relations. 

LSD600 addresses this critical gap by providing the first manually annotated corpus of 600 abstracts with ~1,900 relations across 8 relation types and 9 lifestyle factor categories.

---

## 2. Repository Structure

```text
.
├── README.md                      # Comprehensive project guide and thesis roadmap
├── requirements.txt               # Python package dependencies
├── docs/                          # In-depth reference documentation
│   ├── BASE_PAPER_SUMMARY.md      # Methodological breakdown, baselines & gaps
│   └── RE_Annotation_Guidelines.pdf # Official annotation manual
├── data/                          # Dataset assets from Zenodo
│   ├── raw/
│   │   ├── LSD600/                # 600 abstracts in BRAT format
│   │   │   ├── train/             # 360 abstracts, 1,099 relations
│   │   │   ├── dev/               # 120 abstracts, 425 relations
│   │   │   └── test/              # 120 abstracts, 376 relations
│   │   └── BRAT_config/           # BRAT visualization & schema configs
│   ├── Consolidated_Relations_Dataset.tsv # Consolidated tabular relations dataset
│   └── LSD600_metadata.tsv        # Abstract-level metadata (PMIDs, counts, entities)
├── models/
│   └── pretrained_roberta_lsd600/ # Fine-tuned RoBERTa-large weights from Zenodo
├── src/                           # Core research framework & modular code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py       # BRAT parser, candidate generator, entity masking
│   ├── models/
│   │   ├── __init__.py
│   │   └── baseline_roberta.py    # PyTorch/Transformers baseline model wrapper
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py             # Multi-label micro/macro/per-class & coarse metrics
│       └── evaluator.py           # Evaluation runner on test/dev sets
├── notebooks/                     # Analysis & prototyping scripts
│   └── eda_lsd600.py              # Exploratory data analysis & statistical summaries
└── base_repo/                     # Cloned original paper codebase for reference
```

---

## 3. Dataset Summary & Taxonomy

### The 8 Relation Types (Hierarchical Schema)

```
Statistical Association (288 relations)
├── Positive Statistical Association (592 relations)
│   └── Causes (293 relations)
├── Negative Statistical Association (281 relations)
│   └── Controls (59 relations)
│       ├── Prevents (125 relations)
│       └── Treats (114 relations)
└── No Statistical Association (148 relations)
```

- **Task Formulation:** Multi-label classification on candidate `(Lifestyle_Factor, Disease)` entity pairs.
- **Cross-Sentence Relations:** **16%** of relations cross sentence boundaries (a key challenge in the corpus).
- **Inter-Annotator Agreement (IAA):** F1-score of **82.1%**.

---

## 4. Base Paper Baseline Performance

- **Model:** `RoBERTa-large-PM-M3-Voc` (PubMed BPE vocab, pretrained on PubMed/PMC/MIMIC-III).
- **Entity Masking:** `[unused1]` masks candidate focus entities, `[unused2]` masks other entities in the abstract.
- **Test Set Results (Held-out 120 abstracts):**
  - **Precision:** 77.3%
  - **Recall:** 61.6%
  - **F1-Score:** 68.5%
  - **Coarse-Grained F1:** 73.4%

---

## 5. Quickstart & Usage

### 1. Environment Setup
Recommended conda environment with Python 3.10 and PyTorch 2.x:
```bash
conda activate env_python310
pip install -r requirements.txt
```

### 2. Exploratory Data Analysis
Run the dataset statistics and distributions script:
```bash
python notebooks/eda_lsd600.py
```

### 3. Evaluate Baseline Model on Test Set
```bash
python -m src.evaluation.evaluator \
  --model-path models/pretrained_roberta_lsd600 \
  --data-dir data/raw/LSD600/test \
  --threshold 0.5 \
  --save-output docs/evaluation_results_test.json
```


