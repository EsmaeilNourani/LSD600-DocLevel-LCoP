# Document-Level Relation Extraction with Localized Context Pooling (LCoP)

## 1. Executive Summary & Problem Formulation

In biomedical relation extraction between lifestyle factors and health outcomes (the **LSD600** benchmark, *Database (Oxford)* 2025), extracting relations across complex, multi-sentence abstracts is the paramount challenge.

The base paper achieved an F1-score of **68.5%** (Precision: 77.3%, Recall: 61.6%) using `RoBERTa-large-PM-M3-Voc`. However, the authors' original pipeline inherited a rigid **180-subword token window** centered around candidate pairs from local protein-protein interaction (PPI) tools. 

### Why the 180-Token Window Bottlenecked Performance:
1. **Abstract Length vs. Window Size:**
   - Biomedical abstracts in LSD600 have a mean length of **335.7 subword tokens** (maximum: 1,574 tokens).
   - A 180-token window arbitrarily cuts off more than half the abstract context.
2. **Severe Candidate & Relation Drop:**
   - With 180 tokens, **38.0% of candidate pairs (4,927 pairs in the test set)** are dropped before the model can even see them.
   - In the base paper's manual error analysis, the authors proved that this naive truncation failed on **61 out of 77 cross-sentence relations (accounting for 40.0% of ALL False Negatives)**.
3. **83.3% of Errors are False Negatives:**
   - Error diagnosis of the reproduced baseline confirms that 83.3% of all errors are False Negatives where the model predicts near-zero probability ($P < 0.094$) because the connecting context was pruned by the 180-token window.

---

## 2. Architecture: Localized Context Pooling (LCoP)

To resolve the cross-sentence bottleneck without introducing noise from irrelevant sentences across the abstract, we introduce **Localized Context Pooling (LCoP)** in an isolated, modular architecture (`src/models/localized_context_pooling.py`).

### 2.1 Full Abstract Context Modeling
- Input sequence length is extended to **512 tokens**, covering **91.04% of full abstracts** and **92.5% of all candidate pairs** (up from 62.0%).
- **Strict Ablation Isolation:** Entity masking is preserved **identically** to the base paper:
  - Focus pair $(e_1, e_2)$ is masked with `[unused1]` (token ID 50001).
  - All other entities in the abstract are masked with `[unused2]` (token ID 50002).
  - No confounding variable changes are introduced.

### 2.2 Mathematical Formulation of LCoP
Let $H = [h_1, h_2, \dots, h_L] \in \mathbb{R}^{L \times d}$ be the contextual hidden states output by the RoBERTa encoder for the full abstract.
Let $A \in \mathbb{R}^{K \times L \times L}$ be the multi-head self-attention tensor from the final transformer layer, with head-averaged attention matrix:
$$\bar{A} = \frac{1}{K} \sum_{k=1}^K A_k \in \mathbb{R}^{L \times L}$$

For candidate entities $e_1$ (token positions $T(e_1)$) and $e_2$ (token positions $T(e_2)$):

#### 1. Entity-to-Token Attention Vectors:
$$A_{e_1}(t) = \frac{1}{|T(e_1)|} \sum_{i \in T(e_1)} \bar{A}_{i, t}, \quad A_{e_2}(t) = \frac{1}{|T(e_2)|} \sum_{j \in T(e_2)} \bar{A}_{j, t}$$

#### 2. Mutual Evidence Bridge:
A context token $t$ serves as relational evidence specifically for the pair $(e_1, e_2)$ if both entities actively attend to it. The mutual attention weight is:
$$a_{e_1, e_2}(t) = A_{e_1}(t) \odot A_{e_2}(t)$$

We mask out padding tokens and the entity tokens themselves to pool strictly linking context:
$$\alpha_t = \frac{a_{e_1, e_2}(t) \cdot \mathbb{I}(t \notin T(e_1) \cup T(e_2))}{\sum_{t'} a_{e_1, e_2}(t') \cdot \mathbb{I}(t' \notin T(e_1) \cup T(e_2)) + \epsilon}$$

#### 3. Localized Context Vector:
$$c_{e_1, e_2}^{\text{attn}} = \sum_{t=1}^L \alpha_t h_t \in \mathbb{R}^d$$

#### 4. Learned Bilinear Pair Query:
To complement transformer self-attention, a pair-specific query vector is generated:
$$q_{e_1, e_2} = W_q [h_{e_1}; h_{e_2}; |h_{e_1} - h_{e_2}|] \in \mathbb{R}^d$$
$$s_t = \frac{q_{e_1, e_2}^\top W_k h_t}{\sqrt{d}}, \quad \beta_t = \text{softmax}(s_t)$$
$$c_{e_1, e_2}^{\text{query}} = \sum_{t=1}^L \beta_t h_t \in \mathbb{R}^d$$

In hybrid mode:
$$c_{e_1, e_2} = [c_{e_1, e_2}^{\text{attn}}; c_{e_1, e_2}^{\text{query}}] \in \mathbb{R}^{2d}$$

### 2.3 Composite Multi-Feature Classifier
The final candidate pair representation is synthesized as:
$$z_{e_1, e_2} = \Big[ h_{e_1} \,\|\, h_{e_2} \,\|\, |h_{e_1} - h_{e_2}| \,\|\, (h_{e_1} \odot h_{e_2}) \,\|\, c_{e_1, e_2} \Big]$$

This rich vector is fed into a two-layer MLP with GELU activation and dropout:
$$\hat{y} = \sigma \left( W_2 \cdot \text{GELU}(W_1 z_{e_1, e_2} + b_1) + b_2 \right) \in [0, 1]^8$$

---

## 3. Modular Code Structure

- **Core LCoP Module:** [`src/models/localized_context_pooling.py`](file:///Users/dzq660/My%20Drive/Students/AtashBar/Project/src/models/localized_context_pooling.py)
  - `LocalizedContextPooling`: Pluggable PyTorch layer implementing attention-map, bilinear query, and hybrid pooling modes.
  - `DocumentLevelLCoPExtractor`: Complete end-to-end relation extraction model.
- **Training & Evaluation Pipeline:** [`src/training/train_doc_level_lcop.py`](file:///Users/dzq660/My%20Drive/Students/AtashBar/Project/src/training/train_doc_level_lcop.py)
  - Constructs 512-token candidate sequences.
  - Directly evaluates checkpoints using official `evalsorel.py`.
- **Google Colab Notebook:** [`notebooks/colab_doc_level_lcop.ipynb`](file:///Users/dzq660/My%20Drive/Students/AtashBar/Project/notebooks/colab_doc_level_lcop.ipynb)
  - GPU-ready notebook for fine-tuning on Colab.

---

## 4. How to Train & Evaluate

### Local / Headless Evaluation:
```bash
python -m src.training.train_doc_level_lcop \
  --test-dir data/raw/LSD600/test \
  --model-path models/pretrained_roberta_lsd600 \
  --max-seq-length 512 \
  --eval-only
```

### Full Fine-Tuning Pipeline (Colab / GPU):
```bash
python -m src.training.train_doc_level_lcop \
  --train-dir data/raw/LSD600/train \
  --dev-dir data/raw/LSD600/dev \
  --test-dir data/raw/LSD600/test \
  --model-path models/pretrained_roberta_lsd600 \
  --output-dir models/checkpoints_doc_level_lcop \
  --max-seq-length 512 \
  --pooling-mode hybrid \
  --epochs 10 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --eval-batch-size 32 \
  --lr 2e-5 \
  --threshold 0.5 \
  --fp16
```

---

## 5. Benchmark Comparison Baseline

| Model Configuration | Context Length | Masking Scheme | Precision | Recall | F1-Score | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Published Base Paper** (*Database* 2025) | 180 tokens | `[unused1]` / `[unused2]` | 77.3% | 61.6% | 68.5% | Published |
| **Reproduced Baseline** | 180 tokens | `[unused1]` / `[unused2]` | 77.3% | 61.6% | 68.5% | Verified (Delta 0.0%) |
| **Document-Level LCoP** (Ours) | **512 tokens** | `[unused1]` / `[unused2]` | *TBD* | *TBD* | *Target > 71%* | In Progress |
