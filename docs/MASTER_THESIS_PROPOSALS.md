# Master's Thesis Research Proposals (2026)
## Advancing Biomedical Relation Extraction on LSD600: Contrastive Learning & End-to-End Architectures

**Student:** AtashBar  
**Project:** Document-Level Lifestyle Factor–Disease Relation Extraction (LSD600 Benchmark)  
**Supervisor / Lab:** Esmaeil Nourani / Advanced NLP & Biomedical Data Science  
**Theoretical Foundation:** *Advancing Biomedical Relation Extraction in 2026: A Comprehensive Research Plan for Contrastive Learning and End-to-End Architectures*

---

## 1. Executive Summary & Paradigm Shift

The baseline paper (*Database (Oxford)* 2025) framed relation extraction as a pipeline classification task on top of masked language models (`RoBERTa-large-PM-M3-Voc`), assuming gold-standard entity boundaries and relying on binary cross-entropy loss. While effective as an initial benchmark (F1 = 68.5%), comprehensive error analysis revealed three foundational vulnerabilities:
1. **Severe Cross-Sentence Degradation:** 40% of all false negatives occur when subject and object span different sentences.
2. **Label Hierarchy Confusion:** 24% of errors stem from boundary confusion within the hierarchical taxonomy (e.g., confounding *Causes* with generic *Positive Statistical Association*, or *Prevents* with *Negative Statistical Association*).
3. **Pipeline Fragility & Gold Entity Dependency:** The base model cannot operate autonomously on raw text; it requires pre-extracted, perfect gold entity boundaries, leading to cascading failures in clinical deployment.

To resolve these challenges, the research plan shifts focus to **two modern vectors of innovation**:
- **Vector 1:** Advanced Contrastive Representation Learning (Aligned Contrastive Learning with Label Embeddings & Label-Guided Contrastive In-Context Learning).
- **Vector 2:** Span-Based End-to-End Architectures (Packed Levitated Markers) and Dense-Retrieval Metric Frameworks.

Below are **three targeted, highly publishable Master's Thesis Options** structured around these vectors.

---

## Option A: Contrastive Learning & Label Alignment for Document-Level RE (Vector 1 Focus)
> **Working Title:** *Aligned Contrastive Learning and Label-Guided Representation Alignment for Hierarchical Biomedical Relation Extraction*

### 1. Research Motivation & Core Idea
Standard cross-entropy and binary multi-label loss treat relation categories as orthogonal, one-hot targets. In reality, the LSD600 taxonomy is hierarchical (e.g., `Prevents` $\subset$ `Controls` $\subset$ `Negative Statistical Association`). This option implements **Aligned Contrastive Learning (ACL)** to project entity-pair contextual representations into a shared semantic space alongside dense label embeddings derived from medical definitions.

### 2. Methodological Components
- **Semantic Label Embeddings:** Encode relation definitions, guidance notes, and hierarchy descriptions into dense anchor vectors using biomedical language models.
- **Aligned Contrastive Loss (Supervised InfoNCE / SupCon):**
  - Pull representations of entity pairs belonging to the same relation cluster closer together.
  - Push representations of contrasting relations (e.g., *Treats* vs. *Prevents*) apart according to their tree distance in the taxonomy.
- **Label-Guided Contrastive In-Context Learning (LC-ICL):**
  - Retrieve hard negative and hard positive exemplar pairs dynamically to calibrate classification boundaries for ambiguous pairs.
- **Backbone Integration:** Extends the existing `train_doc_level_lcop.py` pipeline (combining Localized Context Pooling with contrastive loss).

### 3. Student Tasks & Roadmap
1. Define formal label descriptions and taxonomy distance matrices for all 8 LSD600 relations.
2. Implement a contrastive projection head and multi-task loss ($\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{ACL}}$) within the existing PyTorch codebase.
3. Benchmark against standard LCoP on Dev and Test sets using official `evalsorel.py`.
4. Conduct t-SNE latent space visualizations demonstrating improved separation of hierarchical classes.

### 4. Expected Deliverables & Publication Target
- Elimination of the 24% label confusion error mode.
- High-impact paper submission to **Bioinformatics**, **JAMIA**, or **ACL/EMNLP Findings**.

---

## Option B: Span-Based End-to-End Extraction with Packed Levitated Markers (Vector 2 Focus)
> **Working Title:** *End-to-End Lifestyle Factor and Disease Relation Extraction via Span-Based Packed Levitated Markers*

### 1. Research Motivation & Core Idea
Current research relies on gold entities provided in advance. In real biomedical literature processing, entities must be discovered and classified simultaneously with relations. Traditional pipeline architectures suffer from error propagation (NER errors compound into RE errors). This option implements an **End-to-End (E2E) Span-Based architecture with Packed Levitated Markers (PL-Marker)**.

### 2. Methodological Components
- **Packed Levitated Markers (PL-Marker):**
  - Instead of separate token tagging followed by entity masking, floating marker pairs are packed into the attention mechanism to jointly model span boundaries and directional relational interactions without modifying original input tokens.
- **Joint Span Classification:**
  - Evaluates all candidate spans in the document up to length $L$ for entity types (`lifestyle_factor`, `disease`).
  - Simultaneously scores all valid (LSF, Disease) span pairs for the 8 relation classes.
- **Span Pruning & Filter Network:**
  - Efficient top-$k$ span selection to maintain linear computational complexity across 512-token document lengths.

### 3. Student Tasks & Roadmap
1. Convert BRAT standoff annotations (`.txt` + `.ann`) into span-pair datasets (ground-truth spans + relation links).
2. Implement PL-Marker encoder wrapping biomedical backbones (`RoBERTa-large-PM-M3-Voc` or `BioLinkBERT-large`).
3. Formulate joint objective: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{NER}} + \alpha \mathcal{L}_{\text{RE}}$.
4. Benchmark against gold-entity pipeline baseline and evaluate on raw text (true unassisted extraction).

### 4. Expected Deliverables & Publication Target
- The first truly autonomous, unassisted end-to-end extraction model for the LSD600 corpus.
- High-impact paper submission to **IEEE Journal of Biomedical and Health Informatics (J-BHI)** or **COLING/NAACL**.

---

## Option C: Dense-Retrieval Cascade & Metric-Learning for Cross-Sentence RE (Hybrid Focus)
> **Working Title:** *A Dense Retrieval and Metric-Learning Cascade for Long-Distance Cross-Sentence Biomedical Relation Extraction*

### 1. Research Motivation & Core Idea
40% of missed relations in the baseline paper occurred across sentence boundaries. While LCoP (Localized Context Pooling) captures local evidence, abstracts frequently contain non-contiguous evidence sentences separated by unrelated clinical discussions. This option casts relation extraction into a **Dense Retrieval Cascade**: retrieving relational evidence passages followed by metric ranking.

### 2. Methodological Components
- **Evidence Sentence Retrieval Sub-network:**
  - Given an entity pair $(e_1, e_2)$, a dense bi-encoder ranks sentences in the abstract based on their probability of containing bridging evidence.
- **Metric Learning Relation Head:**
  - Projects the gathered evidence sub-graph into a metric space where multi-hop relation inferences are resolved via prototypical distance.
- **Cascade Decoupling:**
  - Stage 1: High-recall span & evidence candidate selection.
  - Stage 2: Precision-oriented metric classifier.

### 3. Student Tasks & Roadmap
1. Construct evidence path annotations from cross-sentence relation pairs in LSD600.
2. Train a bi-encoder retrieval model to identify inter-sentence evidence paths.
3. Compare metric classification vs. standard MLP heads on long-distance entity pairs.
4. Specifically analyze performance gains on the 16% cross-sentence relation subset.

### 4. Expected Deliverables & Publication Target
- Direct resolution of the largest single error source (cross-sentence relations).
- Target publication: **Database (Oxford)** or **BMC Bioinformatics**.

---

## Comparison Matrix for Master's Thesis Selection

| Dimension | Option A: Aligned Contrastive Learning (ACL) | Option B: Span-Based End-to-End (PL-Marker) | Option C: Dense-Retrieval Cascade |
| :--- | :--- | :--- | :--- |
| **Primary Innovation Vector** | Vector 1 (Contrastive & Semantic Alignment) | Vector 2 (End-to-End Joint Modeling) | Hybrid (Retrieval & Metric Learning) |
| **Target Error Addressed** | Label confusion (24% of errors) & hierarchy | Gold entity dependency & error propagation | Cross-sentence gap (40% of false negatives) |
| **Existing Code Reuse** | **Very High** (plugs directly into current `train_doc_level_lcop.py`) | Medium (requires span candidate generator) | High (extends evidence pooling) |
| **GPU / Training Intensity** | Low–Medium (runs easily in Colab with FP16) | Medium–High (many span permutations) | Medium |
| **Novelty Level** | Very High (novel in biomedical RE) | High (state-of-the-art E2E formulation) | High |
| **Estimated Implementation Time**| 4–6 weeks | 8–10 weeks | 6–8 weeks |

---

## Recommendation & Immediate Action Plan
- **Primary Recommendation:** **Option A (Aligned Contrastive Learning)** is the natural, seamless evolution of the work currently running in `train_doc_level_lcop.py`. Because the student already has the document-level LCoP architecture and Colab pipeline running, adding label-guided contrastive loss builds immediately upon working infrastructure while delivering top-tier scientific novelty.
- **Secondary / Expansion Path:** As the student progresses, Option A can be paired with span levitation markers from Option B to deliver a comprehensive dissertation.
