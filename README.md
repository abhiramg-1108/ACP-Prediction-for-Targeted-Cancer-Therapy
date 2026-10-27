# Anticancer Peptide (ACP) Prediction via Biophysical Feature Engineering & Multi-View Hybrid Ensemble Learning

Machine learning pipeline for predicting therapeutic **Anticancer Peptides (ACPs)** directly from primary amino acid sequences. This project reproduces and extends the recent open-access methodologies from **MLACP 2.0** (*Computational and Structural Biotechnology Journal*, 2022) and **ACPred-BMF / AntiCP 2.0** (*Nature Scientific Reports*, 2022–2023) using **100% of the official, uncut IIIT-Delhi benchmark datasets** (`1,720` Main sequences and `1,939` Alternate sequences). 

To address the biophysical and sequence-order limitations of standard composition-based models, we engineer **84 domain-specific membrane-interaction & sequence-order features** (508 total features) coupled with a **Multi-View Soft-Voting Hybrid Ensemble**, achieving **75.51% Accuracy (0.8353 AUROC, 0.5106 MCC)** on the Main independent test set and **93.30% Accuracy (0.9743 AUROC, 0.8682 MCC)** on the Alternate independent test set.

---

## Base Reference Papers (2022–2023)
1. **MLACP 2.0:** Park, H. W., Pitti, T., Madhavan, T., Jeon, Y. J., & Manavalan, B. (2022). *MLACP 2.0: An updated machine learning tool for anticancer peptide prediction.* Computational and Structural Biotechnology Journal, 20, 4473–4480. (**PMCID:** `PMC9421197`)
2. **ACPred-BMF:** Han, J., Kang, N., & সিদ্দিক, M. (2022). *ACPred-BMF: Bidirectional LSTM with multiple feature representations for anticancer peptide prediction.* Scientific Reports, 12, 21915. (**DOI:** `10.1038/s41598-022-24404-1`)
3. **AntiCP 2.0 Official Benchmark Dataset:** Agrawal, P., Bhagat, D., Mahalwal, M., Sharma, N., & Raghava, G. P. S. (*IIIT-Delhi Official Benchmark Repository*, `https://webs.iiitd.edu.in/raghava/anticp2/`).

---

## Official Benchmark Datasets (100% Uncut)

Unlike general sequence databases that require arbitrary sub-sampling, this project evaluates models on **100% of the official, pre-partitioned IIIT-Delhi AntiCP 2.0 benchmark splits** containing only valid standard amino acid sequences (`A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`):

| Benchmark Dataset | Partition | Positive Class (`Label = 1`) | Negative Class (`Label = 0`) | Total Sequences | Mean Length ($\pm$ SD) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Main Dataset** | Official Training (80%) | 688 Validated ACPs | 689 Non-ACPs (AMPs) | 1,377 | $23.84 \pm 11.05$ |
| **1. Main Dataset** | Independent Test (20%) | 172 Validated ACPs | 171 Non-ACPs (AMPs) | 343 | $23.77 \pm 11.10$ |
| **2. Alternate Dataset** | Official Training (80%) | 775 Validated ACPs | 776 Swiss-Prot Peptides | 1,551 | $24.52 \pm 12.08$ |
| **2. Alternate Dataset** | Independent Test (20%) | 194 Validated ACPs | 194 Swiss-Prot Peptides | 388 | $25.25 \pm 12.38$ |

---

## End-to-End Methodology & Feature Engineering

### Phase 1 & Phase 2: Baseline Sequence & Physicochemical Representation (424 Features)
Each variable-length peptide sequence of length $L$ ($4 \le L \le 50$) is mapped into a 424-dimensional baseline feature vector:
1. **Amino Acid Composition (AAC — 20 Features):** Normalized residue frequencies $\text{AAC}(a_i) = \text{Count}(a_i) / L$.
2. **Dipeptide Composition (DPC — 400 Features):** Overlapping consecutive pair frequencies $\text{DPC}(a_i, a_j) = \text{Count}(a_i a_j) / (L - 1)$.
3. **Global Physicochemical Averages (4 Features):** Sequence-averaged **Kyte-Doolittle Hydropathy (GRAVY)** and **Chou-Fasman** secondary structure conformational propensities ($\alpha$-Helix, $\beta$-Sheet, and Reverse Turn).

---

### Phase 3: Limitations of Baseline Models & Proposed Biophysical Mitigations (+84 Features = 508 Features)

#### Limitation 1: Neglect of Electrostatic & Membrane-Disruption Biophysics
* **The Biological Problem:** Cancer cell membranes overexpress anionic phosphatidylserine and O-glycosylated mucins on their outer leaflet, giving cancer cells a strong **negative surface charge**, whereas healthy mammalian cell membranes are zwitterionic (electrically neutral). Standard `AAC` and `DPC` counts fail to explicitly model the electrostatic attraction and amphiphilic insertion required for an ACP to lyse a cancer membrane.
* **Our Mitigation:** We engineered explicit biophysical descriptors for every peptide:
  * **Net Electrostatic Charge & Charge Density at physiological pH 7.4** (`ACP_Net_Charge_pH74`, `ACP_Charge_Density`).
  * **Eisenberg Amphiphilic Hydrophobic Moment ($\mu_H$):** Measures the vector sum of residue hydrophobicities perpendicular to the peptide axis for both ideal $\alpha$-helices ($\delta = 100^\circ$) and $\beta$-sheets ($\delta = 160^\circ$):
    $$\mu_H = \frac{1}{L} \sqrt{\left(\sum_{k=1}^{L} H_k \sin(k\delta)\right)^2 + \left(\sum_{k=1}^{L} H_k \cos(k\delta)\right)^2}$$
  * **Boman Protein-Binding Index (`ACP_Boman_Index`):** Quantifies membrane-protein interaction affinity via residue free-energy solubility transfer scales.
  * **Functional Residue Group Fractions:** Cationic (`K, R, H`), Anionic (`D, E`), Aromaticity (`F, W, Y`), Aliphatic Index (Ikai scale), and Hydrophobic core ratios.

#### Limitation 2: Loss of Sequence Order & Terminal Pore-Formation Motifs
* **The Problem:** Global composition ignores whether hydrophobic and cationic residues are segregated at the **N-terminus** (which initiates lipid bilayer insertion) or **C-terminus** (which stabilizes pore formation).
* **Our Mitigation:**
  * **N-Terminal & C-Terminal Regional Profiles (20 Features):** Separate 5-residue and 10-residue terminal windows across GRAVY, Helix, Sheet, Turn, and Electrostatic Charge.
  * **Pseudo Amino Acid Composition (PseAAC — 40 Features):** Autocorrelation of normalized biophysical properties across sequence lags $d = 1, 2, \dots, 8$.
  * **Composition of $k$-Spaced Amino Acid Pairs (CKSAAP — 14 Features):** Gapped dipeptide motifs (`gap = 1`, e.g., `K_K`, `K_L`, `K_W`, `F_L`, `R_W`) characteristic of amphipathic helical turns.

#### Limitation 3: Single-Classifier Geometry Mismatch
* **The Problem:** While `SVM-RBF` captures smooth Euclidean boundaries across normalized composition fractions, it struggles to isolate sharp threshold splits on continuous biophysical indices (such as net charge $\ge +3$ or high hydrophobic moment).
* **Our Mitigation:** We apply **Univariate ANOVA $F$-test Feature Selection (`SelectKBest`)** followed by a probability-calibrated **Multi-View Soft-Voting Hybrid Ensemble** integrating **Cost-Sensitive SVM-RBF**, **ExtraTrees**, **Random Forest**, and **Histogram-Based Gradient Boosting**.

---

## Experimental Results on Official Independent Test Sets

### 1. Main Benchmark Dataset (`acp_benchmark_results.csv`)
*Task: Distinguishing Anticancer Peptides (ACPs) from Antimicrobial Peptides (AMPs).*

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score | MCC | AUROC | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline: SVM-Polynomial (424 feats)** | 72.30% | 59.30% | **85.38%** | 0.6823 | 0.4627 | 0.7982 | `0.785s` |
| **2. Baseline: SVM-RBF (424 feats)** | 74.05% | 72.67% | 75.44% | 0.7375 | 0.4813 | 0.8152 | `0.866s` |
| **3. Baseline: Random Forest (424 feats)** | 74.93% | **76.74%** | 73.10% | **0.7543** | 0.4988 | 0.8305 | `0.340s` |
| **4. Mitigation A: Biophysical + PseAAC + Tuned SVM** | 71.43% | 68.02% | 74.85% | 0.7048 | 0.4297 | 0.8113 | `0.820s` |
| **5. Mitigation B: Proposed Multi-View Hybrid Ensemble (508 feats)** | **75.51%** | 73.84% | 77.19% | 0.7515 | **0.5106** | **0.8353** | `5.461s` |

### 2. Alternate Benchmark Dataset (`acp_benchmark_results.csv`)
*Task: Distinguishing Anticancer Peptides (ACPs) from Random Swiss-Prot Peptides.*

| Model | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score | MCC | AUROC | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline: SVM-Polynomial (424 feats)** | 85.31% | 72.68% | **97.94%** | 0.8319 | 0.7298 | 0.9303 | `0.829s` |
| **2. Baseline: SVM-RBF (424 feats)** | 91.75% | 86.08% | 97.42% | 0.9126 | 0.8405 | 0.9579 | `0.785s` |
| **3. Baseline: Random Forest (424 feats)** | 92.01% | 89.18% | 94.85% | 0.9178 | 0.8416 | 0.9680 | `0.352s` |
| **4. Mitigation A: Biophysical + PseAAC + Tuned SVM** | 92.53% | **90.21%** | 94.85% | 0.9235 | 0.8514 | 0.9656 | `0.569s` |
| **5. Mitigation B: Proposed Multi-View Hybrid Ensemble (508 feats)** | **93.30%** | 89.69% | 96.91% | **0.9305** | **0.8682** | **0.9743** | `5.763s` |

---

## Repository Structure & Quickstart

```text
├── download_acp_dataset.py          # Downloads 100% of the official IIIT-Delhi AntiCP 2.0 benchmark files
├── prepare_acp_data.py              # Validates and inspects raw FASTA/sequence partitions
├── run_acp_pipeline.py              # Full pipeline: Feature extraction (424 -> 508) + Baseline & Ensemble training
├── acp_main_dataset_clean.csv       # 1,720 clean sequences (Official Train + Independent Test)
├── acp_alternate_dataset_clean.csv  # 1,939 clean sequences (Official Train + Independent Test)
├── acp_benchmark_results.csv        # Complete evaluation metrics across both datasets
├── acp_ablation_comparison.png      # High-res bar chart comparing Accuracy and AUROC
├── acp_roc_curves.png               # ROC Curves on Main and Alternate Independent Test Sets
└── README.md                        # Project documentation
