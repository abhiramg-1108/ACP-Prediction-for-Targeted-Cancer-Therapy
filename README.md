# ACPred-BMF+: Hybrid Deep Learning (BiLSTM + Self-Attention) & Biophysical Feature Fusion for Explainable Anticancer Peptide Prediction

An end-to-end Bioinformatics Deep Learning and Machine Learning framework for identifying therapeutic **Anticancer Peptides (ACPs)** directly from primary amino acid sequences. 

This project faithfully implements and extends the open-access research paper **ACPred-BMF** (Han et al., *Nature Scientific Reports*, Dec 2022) on **100% of the official, uncut IIIT-Delhi AntiCP 2.0 benchmark datasets** (`1,720` Main sequences and `1,939` Alternate sequences). To overcome the limitations of discrete binary encodings in standalone Recurrent Neural Networks, we propose **ACPred-BMF+**, a hybrid meta-fusion architecture that integrates **36-dimensional Deep Bidirectional LSTM (`BiLSTM`) sequential representations** with **6 non-redundant continuous thermodynamic & 3D membrane-biophysical descriptors**, achieving **77.26% Accuracy (0.8616 AUROC)** on the Main independent test set and **0.9752 AUROC (91.49% Accuracy)** on the Alternate independent test set.

---

## 1. Clinical & Biological Motivation: Why Study Anticancer Peptides (ACPs)?

Traditional cancer treatments—such as chemotherapy and radiation therapy—suffer from two major clinical drawbacks:
1. **Lack of Specificity (Systemic Toxicity):** Conventional small-molecule chemotherapeutics target rapidly dividing cells indiscriminately, destroying healthy human cells (bone marrow, hair follicles, gastrointestinal lining) alongside tumor cells.
2. **Chemoresistance:** Cancer cells frequently mutate intracellular pathways or overexpress efflux pumps (like P-glycoprotein) to pump chemotherapy drugs out of the cell.

### How Anticancer Peptides (ACPs) Solve This
**Anticancer Peptides (ACPs)** are short cationic peptides (typically 5 to 50 amino acids long) that act as targeted "molecular needles":
* **Electrostatic Selectivity:** Healthy mammalian cell membranes are electrically neutral (zwitterionic) on their outer surface. In contrast, cancer cell membranes overexpress negatively charged **phosphatidylserine (PS)**, **O-glycosylated mucins**, and **heparan sulfate** on their outer leaflet. Positively charged ACPs are electrostatically attracted specifically to negatively charged cancer cells while leaving healthy cells unharmed.
* **Membranolytic Mechanism (Pore Formation):** Once bound to the tumor surface, ACPs fold into rigid amphiphilic structures (such as $\alpha$-helices) that physically bore pores into the lipid bilayer (via the barrel-stave, toroidal-pore, or carpet models), causing cell lysis. Because ACPs destroy the physical membrane rather than targeting a single intracellular protein, cancer cells rarely develop resistance.
* **Why AI/DL is Needed:** Wet-lab synthesis and *in vitro* screening of thousands of candidate peptides is slow and expensive. High-accuracy *in silico* Deep Learning predictors drastically accelerate therapeutic peptide discovery.

---

## 2. Base Reference Paper & Official Benchmark Datasets

### Primary Base Paper (Followed 100% Faithfully)
* **ACPred-BMF:** Han, J., Kang, N., & Siddique, M. (2022). *ACPred-BMF: Bidirectional LSTM with multiple feature representations for explainable anticancer peptide prediction.* **Nature Scientific Reports**, 12, 21915. (**DOI:** `10.1038/s41598-022-24404-1` | **PMCID:** `PMC9763336`).
* **Benchmark Dataset Origin (AntiCP 2.0):** Agrawal, P., Bhagat, D., Mahalwal, M., Sharma, N., & Raghava, G. P. S. (2021). *AntiCP 2.0: an updated model for predicting anticancer peptides.* **Briefings in Bioinformatics**, 22(3), bbaa153. (**Official Server:** `https://webs.iiitd.edu.in/raghava/anticp2/download.php`).

### What Exactly Do Our Datasets Contain? (100% Uncut Official Partitions)
No sub-sampling, slicing, or random re-splitting was performed. We evaluate on **100% of the official, pre-partitioned IIIT-Delhi benchmark splits** containing valid standard amino acid sequences ($4 \le L \le 50$):

| Benchmark Dataset | Official Split | Class `1` (Positive) | Class `0` (Negative) | Total Sequences | Mean Length ($\pm$ SD) | Biological Nature of Negative Class |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. MAIN Dataset** (`1,720` total) | **Training Set (80%)** | 688 ACPs | 689 Non-ACPs | **1,377** | $23.84 \pm 11.05$ | Experimentally validated **Antimicrobial Peptides (AMPs)** with *no* anticancer activity. |
| **1. MAIN Dataset** | **Independent Test (20%)** | 172 ACPs | 171 Non-ACPs | **343** | $23.77 \pm 11.10$ | Held-out **AMPs** for strict generalization testing. |
| **2. ALTERNATE Dataset** (`1,939` total) | **Training Set (80%)** | 775 ACPs | 776 Non-ACPs | **1,551** | $24.52 \pm 12.08$ | Random non-secretory peptides extracted from **Swiss-Prot**. |
| **2. ALTERNATE Dataset** | **Independent Test (20%)** | 194 ACPs | 194 Non-ACPs | **388** | $25.25 \pm 12.38$ | Held-out random **Swiss-Prot** peptides. |

---

## 3. Why Accuracy is Lower on the MAIN Dataset (~77%) vs. ALTERNATE Dataset (~92%): The ACP vs. AMP Dilemma

A critical bioinformatics insight observed across both the base paper and our experiments is the ~15% accuracy gap between the **Main Dataset** and the **Alternate Dataset**:

1. **Why the Alternate Dataset achieves ~91.5%–92.3% Accuracy (0.9752 AUROC):**
   * In the **Alternate Dataset**, negative samples (`Label = 0`) are random intracellular/non-secretory protein fragments from Swiss-Prot.
   * Random proteins have balanced electrical charges and low amphiphilicity, whereas Anticancer Peptides (`Label = 1`) are strongly cationic and amphiphilic. The decision boundary is wide and distinct.
2. **Why the Main Dataset achieves ~74.3%–77.3% Accuracy (0.8616 AUROC):**
   * In the **Main Dataset**, negative samples (`Label = 0`) are **Antimicrobial Peptides (AMPs)**—host-defense peptides that kill bacteria.
   * Bacterial membranes, just like cancer membranes, are negatively charged! Therefore, **both AMPs and ACPs are short, positively charged (cationic), and amphiphilic**.
   * Distinguishing an ACP from an AMP requires capturing ultra-subtle differences in **3D helical hydrophobic moment**, **continuous hydropathy (GRAVY)**, and **local sequence grammar** rather than simple charge counts. This makes the Main Dataset a much harder biochemical classification problem, and explains why adding our continuous 3D biophysical features yields the highest accuracy (**77.26%**) and AUROC (**0.8616**) on the Main test set.

---

## 4. Phase-by-Phase Methodology

### Phase 1: Base Paper Residue Encoding (36-Dimensional Sequence Tensor)
Following *ACPred-BMF*, each peptide sequence (up to $L_{\max} = 50$) is encoded residue-by-residue into a **2D Tensor of shape `(50, 36)`** by concatenating three feature views:
1. **Binary Profile Feature (`BPF` — 20D):** One-hot encoding of the 20 standard amino acids (`A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`).
2. **Quantitative Physicochemical Properties (`Quanc` — 6D, Paper Table 1):** $z$-score standardized physical constants per residue:
   * Molecular Weight, Isoelectric Point ($pI$), Dissociation Constants $pK_1$ ($\alpha\text{-COOH}$), $pK_2$ ($\alpha\text{-NH}_3^+$), $pK_a$ (Side Chain), and van der Waals Volume.
3. **Qualitative Physicochemical Categories (`Qualc` — 10D, Paper Table 2):** Exact 10 binary (`1`/`0`) indicator flags across 4 qualitative groups:
   * **Hydrophobicity:** `Y` (`G,A,V,L,I,F,W,Y,H,K,M,T,C`) vs. `N` (`D,N,E,Q,R,S,P`)
   * **Polarity:** `Y` (`W,Y,D,H,N,E,K,Q,R,S,T,C`) vs. `N` (`G,A,V,L,I,F,M,P`)
   * **Charge:** `Negative` (`D,E`), `Positive` (`H,K,R`), vs. `Neutral/N` (`G,A,V,L,I,F,W,Y,N,Q,M,S,T,C,P`)
   * **Aromatic or Aliphatic:** `Aromatic` (`F,W,Y,H`), `Aliphatic` (`L,I,V`), vs. `N` (`G,A,D,N,E,K,Q,M,R,S,T,C,P`)

All four paper combinations are benchmarked: **`BPF` (20D)**, **`BPF+Quanc` (26D)**, **`BPF+Qualc` (30D)**, and **`BPF+Quanc+Qualc / Mix` (36D)**.

---

### Phase 2: Base Paper Deep Learning Architectures (`BMF-Basic` & `BMF-Selfatt`)
Implemented in **PyTorch** using mini-batch stochastic optimization (`AdamW`, `CosineAnnealingLR`, `BCEWithLogitsLoss`, and `Dropout`):
* **`BMF-Basic`:** Feeds the `(50, D)` sequence tensor into a **Bidirectional LSTM (`BiLSTM`)**, reading the peptide simultaneously from N-terminus $\rightarrow$ C-terminus (forward hidden state $\overrightarrow{h_t}$) and C-terminus $\rightarrow$ N-terminus (backward hidden state $\overleftarrow{h_t}$), followed by global pooling and dense classification layers.
* **`BMF-Selfatt`:** Adds a **Scaled Dot-Product Self-Attention layer** ($Q, K, V$ projections) on top of the BiLSTM hidden states, enabling the network to dynamically weight specific amino acid positions relative to all other residues in the chain rather than treating every position equally.

---

### Phase 3: Limitations of the Base Paper & Our 6 Non-Redundant Biophysical Features

#### What the Base Paper Missed
While `Qualc` (Paper Table 2) already includes binary `1`/`0` flags for Aromaticity, Aliphaticity, Charge, Polarity, and Hydrophobicity, it treats hydrophobicity like an on/off light switch (`1` or `0`) and completely misses **3D secondary-structure folding** and **rotational helical amphiphilicity**.

#### Our 6 Non-Redundant Continuous Biophysical Features (Zero Overlap with Table 1 or Table 2)
To avoid feature redundancy, we did **not** re-add aromaticity or charge counts. Instead, we engineered **6 continuous thermodynamic and 3D geometric features** (combined with `20 AAC` + `16 Global Quanc/Qualc` = **42 compact features**):

1. **Chou-Fasman Secondary Structure Propensities (`New_Alpha_Helix`, `New_Beta_Sheet`, `New_Turn` — 3 Features):**
   * *Biological Role:* A floppy, unfolded peptide chain cannot puncture a tough tumor lipid bilayer. Most potent ACPs must fold into a rigid **$\alpha$-helical corkscrew** (or $\beta$-sheet) to mechanically penetrate the cell wall. These three thermodynamic scales quantify the exact conformational folding probability of the sequence.
2. **Continuous Kyte-Doolittle Hydropathy (`New_GRAVY` — 1 Feature):**
   * *Biological Role:* Replaces Table 2's binary `1/0` hydrophobicity flag with an exact thermodynamic water-to-lipid transfer free energy score (ranging from $-4.5$ for Arginine to $+4.5$ for Isoleucine), measuring how effectively the peptide sinks into the fatty core of the cancer cell membrane.
3. **Eisenberg 3D $\alpha$-Helical Hydrophobic Moment (`New_Hydrophobic_Moment` $\mu_H$ — 1 Feature):**
   * *Biological Role:* Measures whether a helical peptide is **"two-faced" (amphiphilic)** by projecting residue hydrophobicities $H_k$ at $100^\circ$ angular increments around a 3D $\alpha$-helical wheel:
     $$\mu_H = \frac{1}{L} \sqrt{\left(\sum_{k=1}^{L} H_k \sin(100^\circ \cdot k)\right)^2 + \left(\sum_{k=1}^{L} H_k \cos(100^\circ \cdot k)\right)^2}$$
   * A high $\mu_H$ proves that hydrophobic residues cluster on one face of the cylinder (to grip the lipid membrane) while hydrophilic/charged residues cluster on the opposite face (to pull water into the pore and lyse the cancer cell).
4. **Boman Membrane-Binding Index (`New_Boman_Index` — 1 Feature):**
   * *Biological Role:* Computes the negative mean free-energy solubility scale across all residues, quantifying the overall binding affinity of the peptide to membrane-bound proteins and lipid bilayers.

---

### Phase 4: Why Our Architecture is a Hybrid Deep Learning + Machine Learning Model (`ACPred-BMF+`)
* **The Deep Learning Branch (`BiLSTM` / `Self-Attention`):** Digests the **2D sequential tensor `(50, 36)`** using hidden recurrent memory states, backpropagation, dropout regularization, and cosine learning-rate scheduling to learn the **sequential grammar** of amino acid motifs.
* **The Machine Learning Branch (`SVM-RBF` + `ExtraTrees` + `RandomForest` + `HistGradientBoosting`):** Digests the **1D global biophysical vector (42 features)** to isolate sharp non-linear thermodynamic thresholds (e.g., high $\mu_H$ + high $\alpha$-helical propensity).
* **Meta-Fusion (`ACPred-BMF+`):** Combines the calibrated probability of the 5-fold cross-validated Deep Learning BiLSTM ($P_{\text{BiLSTM}}$) with the Biophysical Ensemble ($P_{\text{Biophys}}$):
  $$P_{\text{ACPred-BMF+}} = 0.50 \cdot P_{\text{BiLSTM}} + 0.50 \cdot P_{\text{Biophys}}$$

---

### Phase 5: Strict Two-Stage Evaluation Protocol (No Data Leakage)
1. **Hyperparameter Grid Search & 5-Fold Stratified Cross-Validation (Training Split Only):**
   * Conducted strictly on the `80%` Official Training Set (`1,377` Main / `1,551` Alternate) using `StratifiedKFold(n_splits=5)`.
   * Grid search optimizes **BiLSTM Hidden Units $\in \{32, 64, 128\}$** (memory capacity) and **Dropout Rate $\in \{0.2, 0.5\}$** (neuron deactivation rate to prevent overfitting), reproducing **Paper Tables 4 & 5**.
2. **Independent Testing (Held-Out 20% Validation Split):**
   * Evaluates the locked models on the completely unseen `20%` Independent Test Set (`343` Main / `388` Alternate), reproducing **Paper Table 6**.

---

## 5. Experimental Results on Official Independent Test Sets

### Feature Ablation & Hybrid Comparison (`acpred_bmf_comparison.png`)

| Benchmark Dataset | Model / Feature Configuration | Accuracy (%) | AUROC | Key Observation |
| :--- | :--- | :---: | :---: | :--- |
| **MAIN Test Set** | `BMF-Basic` — **BPF Only (20D)** | 74.34% | 0.8446 | Baseline one-hot sequence representation |
| **MAIN Test Set** | `BMF-Basic` — **BPF + Quanc (26D)** | 72.01% | 0.8334 | Quantitative properties alone add slight noise on AMP boundary |
| **MAIN Test Set** | `BMF-Basic` — **BPF + Qualc (30D)** | 76.97% | 0.8553 | Binary physicochemical flags improve AMP vs. ACP separation |
| **MAIN Test Set** | `BMF-Basic` — **BPF + Quanc + Qualc (36D Mix)** | 76.38% | 0.8550 | Full base paper `BMF-Basic` 36D model |
| **MAIN Test Set** | `BMF-Selfatt` — **BPF + Quanc + Qualc (36D Mix)** | — | 0.8404 | Full base paper `BMF-Selfatt` 36D model |
| **MAIN Test Set** | **Proposed `ACPred-BMF+` Hybrid (36D DL + 6 New Biophys)** | **77.26%** | **0.8616** | **Highest Accuracy (+0.88% over 36D Mix) & Highest AUROC (+0.0066)** |
| --- | --- | :---: | :---: | --- |
| **ALTERNATE Test Set** | `BMF-Basic` — **BPF Only (20D)** | 90.72% | 0.9567 | Strong baseline against random Swiss-Prot peptides |
| **ALTERNATE Test Set** | `BMF-Basic` — **BPF + Quanc (26D)** | 90.72% | 0.9629 | Improved probability ranking (+0.0062 AUROC) |
| **ALTERNATE Test Set** | `BMF-Basic` — **BPF + Qualc (30D)** | **92.27%** | 0.9704 | Strong threshold accuracy from qualitative grouping |
| **ALTERNATE Test Set** | `BMF-Basic` — **BPF + Quanc + Qualc (36D Mix)** | 90.72% | 0.9714 | Full base paper `BMF-Basic` 36D model |
| **ALTERNATE Test Set** | `BMF-Selfatt` — **BPF + Quanc + Qualc (36D Mix)** | — | 0.9735 | Full base paper `BMF-Selfatt` 36D model |
| **ALTERNATE Test Set** | **Proposed `ACPred-BMF+` Hybrid (36D DL + 6 New Biophys)** | 91.49% | **0.9752** | **Beats 36D Mix in Accuracy (+0.77%) & achieves #1 overall AUROC (0.9752)** |

---

## 6. SHAP Explainability Analysis: Opening the Deep Learning Black Box (`acpred_bmf_shap_explainability.png`)

To address the "black-box" nature of deep neural networks, we replicated the **SHAP (SHapley Additive exPlanations)** analysis (Paper Figures 4 & 5) on the Alternate Dataset across the 16 quantitative (`Quanc`) and qualitative (`Qualc`) features:

1. **Top 5 Most Influential Features (Figure 5 — Ranked by $\text{mean}(\vert{}\text{SHAP value}\vert{})$):**
   * `1. charge_Negative` (`D, E`)
   * `2. hydrophobicity_N` (`D, N, E, Q, R, S, P`)
   * `3. hydrophobicity_Y` (`G, A, V, L, I, F, W, Y, H, K, M, T, C`)
   * `4. pI` (Isoelectric Point)
   * `5. charge_Positive` (`H, K, R`)
   * Followed by `pKa`, `Aliphatic`, `pK2`, `Aromatic`, and `polarity_Y`.
2. **Directional Biophysics Confirmed by SHAP Beeswarm Plot (Figure 4 — Red = High Value, Blue = Low Value):**
   * **Positive Drivers of Anticancer Activity (`SHAP > 0`, Right Side):** High values (**Red dots**) of **`hydrophobicity_Y`**, **`pI` (Isoelectric Point)**, **`charge_Positive`**, **`pKa`**, and **`Aromatic`** cluster strongly on the positive right side of the zero line. This confirms that cationic (high $pI$, positively charged) and hydrophobic/aromatic peptides are strongly predicted as true ACPs.
   * **Negative Drivers / Repulsion (`SHAP < 0`, Left Side):** Low values (**Blue dots**) of **`charge_Negative`** and **`hydrophobicity_N`** sit on the right (`SHAP > 0`), whereas high values (**Red/Orange dots**) push strongly to the left (`SHAP < 0`). Biologically, acidic/negatively charged (`D, E`) and non-hydrophobic peptides are electrostatically repelled by anionic cancer cell membranes and are therefore rejected as Non-ACPs.

---

## 7. Repository Structure & Execution Guide

```text
├── raw_anticp2_data/                   # 8 official uncut IIIT-Delhi AntiCP 2.0 train/test sequence files
├── download_acp_dataset.py             # Script 1: Downloads official benchmark files from IIIT-Delhi server
├── build_clean_datasets.py             # Script 2: Parses raw files into validated train/test CSVs
├── run_acpred_bmf.py                   # Script 3: Full pipeline (Grid Search, 5-Fold CV, BiLSTM, Hybrid, SHAP)
├── acp_main_dataset_clean.csv          # 1,720 clean Main dataset sequences (1,377 train + 343 test)
├── acp_alternate_dataset_clean.csv     # 1,939 clean Alternate dataset sequences (1,551 train + 388 test)
├── table4_grid_search_5fold_cv.csv     # Paper Table 4 reproduction: BiLSTM Units & Dropout 5-Fold CV Grid Search
├── table5_5fold_cv_results.csv         # Paper Table 5 reproduction: 5-Fold Cross-Validation metrics
├── acpred_bmf_complete_results.csv     # Paper Table 6 reproduction: Independent Test Set metrics
├── acpred_bmf_comparison.png           # Plot 1: Feature Ablation & Hybrid Accuracy/AUROC Bar Chart
├── acpred_bmf_roc.png                  # Plot 2: Independent Test Set ROC Curves (Main & Alternate)
├── acpred_bmf_shap_explainability.png  # Plot 3: Paper Fig. 4 & 5 SHAP Beeswarm & Feature Importance Rank
└── README.md                           # Complete project documentation
# 1. Install dependencies
pip install torch pandas numpy scikit-learn matplotlib seaborn shap

# 2. (Optional) Download raw benchmark files from IIIT-Delhi server
python download_acp_dataset.py

# 3. Build clean train/test CSVs from raw_anticp2_data/
python build_clean_datasets.py

# 4. Run 5-Fold CV, Independent Testing, Hybrid Fusion, and generate all 3 plots
python run_acpred_bmf.py
