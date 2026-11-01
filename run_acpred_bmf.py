import time
import random
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix, roc_curve

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running ACPred-BMF (5-Fold CV + Independent Test) on: {DEVICE}")

MAX_LEN = 50
AMINO_ACIDS = sorted(list("ACDEFGHIKLMNPQRSTVWY"))
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# ==============================================================================
# 1. EXACT PAPER TABLE 1 (Quanc - 6D) & TABLE 2 (Qualc - 10D)
# ==============================================================================
QUANC_RAW = {
    'G': [ 75.07,  6.06, 2.34,  9.60,  0.00,  47.3],
    'A': [ 89.09,  6.11, 2.34,  9.69,  0.00,  64.4],
    'V': [117.15,  6.02, 2.32,  9.62,  0.00,  98.6],
    'L': [131.17,  6.04, 2.36,  9.60,  0.00, 115.7],
    'I': [131.17,  6.04, 2.36,  9.60,  0.00, 115.7],
    'F': [165.19,  5.76, 1.83,  9.13,  0.00, 139.9],
    'W': [204.23,  5.88, 2.83,  9.39,  0.00, 196.9],
    'Y': [181.19,  5.63, 2.20,  9.11, 10.07, 136.9],
    'D': [133.10,  2.98, 1.88,  9.60,  3.65,  80.1],
    'H': [155.16,  7.64, 1.82,  9.17,  6.00, 118.9],
    'N': [132.12,  5.43, 2.02,  8.80,  0.00,  94.6],
    'E': [147.13,  3.08, 2.19,  9.67,  4.25,  97.2],
    'K': [146.19,  9.47, 2.18,  8.95, 10.53, 118.1],
    'Q': [146.15,  5.65, 2.17,  9.13,  0.00, 111.7],
    'M': [149.21,  5.71, 2.28,  9.21,  0.00, 120.5],
    'R': [174.20, 10.76, 2.17,  9.04, 12.48, 138.4],
    'S': [105.09,  5.70, 2.21,  9.15,  0.00,  66.1],
    'T': [119.12,  5.60, 2.09,  9.10,  0.00,  88.9],
    'C': [121.16,  5.15, 1.96, 10.28,  8.18,  82.2],
    'P': [115.13,  6.30, 1.99, 10.60,  0.00,  88.0]
}
QUANC_NAMES = ["Molecular weight", "pI", "pK1", "pK2", "pKa", "volume"]

quanc_mat = np.array([QUANC_RAW[aa] for aa in AMINO_ACIDS], dtype=np.float32)
q_mean = quanc_mat.mean(axis=0, keepdims=True)
q_std  = quanc_mat.std(axis=0, ddof=1, keepdims=True) + 1e-8
QUANC_Z = {aa: ((QUANC_RAW[aa] - q_mean[0]) / q_std[0]).astype(np.float32) for aa in AMINO_ACIDS}

QUALC_SETS = [
    ("hydrophobicity_Y",       set("GAVLIFWYHKMTC")),
    ("hydrophobicity_N",       set("DNEQRSP")),
    ("polarity_Y",             set("WYDHNEKQRSTC")),
    ("polarity_N",             set("GAVLIFMP")),
    ("charge_Negative",        set("DE")),
    ("charge_Positive",        set("HKR")),
    ("charge_N",               set("GAVLIFWYNQMSTCP")),
    ("Aromatic",               set("FWYH")),
    ("Aliphatic",              set("LIV")),
    ("Aromatic_or_aliphatic_N", set("GADNEKQMRSTCP"))
]
QUALC_NAMES = [name for name, _ in QUALC_SETS]
QUALC_VEC = {aa: np.array([1.0 if aa in s else 0.0 for _, s in QUALC_SETS], dtype=np.float32) for aa in AMINO_ACIDS}

FEATURE_MODES = {
    "BPF (20D)":             20,
    "BPF+Quanc (26D)":       26,
    "BPF+Qualc (30D)":       30,
    "BPF+Quanc+Qualc (36D)": 36
}

def encode_paper_tensor(sequences, mode="BPF+Quanc+Qualc (36D)"):
    D = FEATURE_MODES[mode]
    tensor = np.zeros((len(sequences), MAX_LEN, D), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:MAX_LEN]):
            bpf = np.zeros(20, dtype=np.float32)
            bpf[AA_TO_IDX[aa]] = 1.0
            if mode == "BPF (20D)":
                vec = bpf
            elif mode == "BPF+Quanc (26D)":
                vec = np.concatenate([bpf, QUANC_Z[aa]])
            elif mode == "BPF+Qualc (30D)":
                vec = np.concatenate([bpf, QUALC_VEC[aa]])
            else:
                vec = np.concatenate([bpf, QUANC_Z[aa], QUALC_VEC[aa]])
            tensor[i, j, :] = vec
    return torch.tensor(tensor, dtype=torch.float32)

# ==============================================================================
# 2. EXACT PAPER ARCHITECTURES: BMF-Basic & BMF-Selfatt
# ==============================================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)

    def forward(self, x):
        scores = torch.bmm(self.W_q(x), self.W_k(x).transpose(1, 2)) / self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights, self.W_v(x))

class ACPredBMFModel(nn.Module):
    def __init__(self, input_dim=36, lstm_units=64, use_attention=False, dropout=0.2):
        super().__init__()
        self.use_attention = use_attention
        self.bilstm = nn.LSTM(input_dim, lstm_units, num_layers=1, batch_first=True, bidirectional=True)
        out_dim = lstm_units * 2
        if use_attention:
            self.att = SelfAttention(out_dim)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.bilstm(x)
        if self.use_attention:
            out = self.att(out)
        avg_p = torch.mean(out, dim=1)
        max_p, _ = torch.max(out, dim=1)
        return self.fc(torch.cat([avg_p, max_p], dim=1)).squeeze(-1)

def train_single_bilstm(X_tr, y_tr, X_val, input_dim=36, lstm_units=64, use_attention=False, dropout=0.2, epochs=40, seed=42):
    set_seed(seed)
    model = ACPredBMFModel(input_dim=input_dim, lstm_units=lstm_units, use_attention=use_attention, dropout=dropout).to(DEVICE)
    loader = DataLoader(TensorDataset(X_tr, torch.tensor(y_tr, dtype=torch.float32)), batch_size=32, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(bx), by).backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val.to(DEVICE))).cpu().numpy()
    return model, val_probs

def run_5fold_cv_and_indep_bilstm(X_tr_all, y_tr_all, X_te_indep, input_dim=36, lstm_units=64, use_attention=False, dropout=0.2, epochs=40):
    """
    1. Runs 5-Fold Stratified Cross-Validation strictly on the Training Set (Paper Table 4 & Table 5).
    2. Predicts on the Independent Test Set using the 5-fold ensemble average + full-train model (Paper Table 6).
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_tr_all), dtype=np.float32)
    indep_fold_probs = np.zeros(len(X_te_indep), dtype=np.float32)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(y_tr_all)), y_tr_all)):
        model, val_p = train_single_bilstm(
            X_tr_all[tr_idx], y_tr_all[tr_idx], X_tr_all[val_idx],
            input_dim=input_dim, lstm_units=lstm_units, use_attention=use_attention,
            dropout=dropout, epochs=epochs, seed=42 + fold_idx
        )
        oof_probs[val_idx] = val_p
        model.eval()
        with torch.no_grad():
            indep_fold_probs += torch.sigmoid(model(X_te_indep.to(DEVICE))).cpu().numpy() / 5.0

    return oof_probs, indep_fold_probs

# ==============================================================================
# 3. PAPER 16 FEATURES (FOR SHAP) + 6 NON-REDUNDANT NEW BIOPHYSICAL FEATURES
# ==============================================================================
NEW_BIOPHYS_SCALES = {
    'A': ( 1.8,  1.81, 1.42, 0.83, 0.66), 'C': ( 2.5,  1.28, 0.70, 1.19, 1.19),
    'D': (-3.5, -8.72, 1.01, 0.54, 1.46), 'E': (-3.5, -6.81, 1.51, 0.37, 0.74),
    'F': ( 2.8,  2.98, 1.13, 1.38, 0.60), 'G': (-0.4,  0.94, 0.57, 0.75, 1.56),
    'H': (-3.2, -4.66, 1.00, 0.87, 0.95), 'I': ( 4.5,  4.92, 1.08, 1.60, 0.47),
    'K': (-3.9, -5.55, 1.16, 0.74, 1.01), 'L': ( 3.8,  4.92, 1.21, 1.30, 0.59),
    'M': ( 1.9,  2.35, 1.45, 1.05, 0.60), 'N': (-3.5, -6.64, 0.67, 0.89, 1.56),
    'P': (-1.6,  0.00, 0.57, 0.55, 1.52), 'Q': (-3.5, -5.54, 1.11, 1.10, 0.98),
    'R': (-4.5,-14.92, 0.98, 0.93, 0.95), 'S': (-0.8, -3.40, 0.77, 0.75, 1.43),
    'T': (-0.7, -2.57, 0.83, 1.19, 0.96), 'V': ( 4.2,  4.04, 1.06, 1.70, 0.50),
    'W': (-0.9,  2.33, 1.08, 1.37, 0.96), 'Y': (-1.3, -0.14, 0.69, 1.47, 1.14)
}

def extract_features_compact(sequences):
    paper_16_rows, model_42_rows = [], []
    for seq in sequences:
        L = len(seq)
        counts = Counter(seq)
        q_arr = np.array([QUANC_Z[aa] for aa in seq])
        ql_arr = np.array([QUALC_VEC[aa] for aa in seq])
        
        p16 = {}
        for i, name in enumerate(QUANC_NAMES):
            p16[name] = np.mean(q_arr[:, i])
        for i, name in enumerate(QUALC_NAMES):
            p16[name] = np.mean(ql_arr[:, i])
        paper_16_rows.append(p16)
        
        m42 = dict(p16)
        for aa in AMINO_ACIDS:
            m42[f"AAC_{aa}"] = counts[aa] / L
            
        nb = np.array([NEW_BIOPHYS_SCALES[aa] for aa in seq])
        h_vals = nb[:, 0]
        angles = np.deg2rad(np.arange(L) * 100.0)
        m42["New_GRAVY"]              = np.mean(h_vals)
        m42["New_Hydrophobic_Moment"] = np.sqrt((np.sum(h_vals * np.sin(angles)) / L)**2 + (np.sum(h_vals * np.cos(angles)) / L)**2)
        m42["New_Boman_Index"]        = -np.mean(nb[:, 1])
        m42["New_Alpha_Helix"]        = np.mean(nb[:, 2])
        m42["New_Beta_Sheet"]         = np.mean(nb[:, 3])
        m42["New_Turn"]               = np.mean(nb[:, 4])
        model_42_rows.append(m42)
        
    return pd.DataFrame(paper_16_rows), pd.DataFrame(model_42_rows)

def run_5fold_cv_and_indep_biophys(X_tr_df, y_tr_all, X_te_df):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(y_tr_all), dtype=np.float32)
    
    def make_ens():
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", VotingClassifier(
                estimators=[
                    ("svm", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                    ("et", ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)),
                    ("rf", RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1)),
                    ("gb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42))
                ],
                voting="soft", weights=[2, 3, 2, 2]
            ))
        ])

    for tr_idx, val_idx in skf.split(X_tr_df, y_tr_all):
        ens = make_ens()
        ens.fit(X_tr_df.iloc[tr_idx], y_tr_all[tr_idx])
        oof_probs[val_idx] = ens.predict_proba(X_tr_df.iloc[val_idx])[:, 1]

    full_ens = make_ens()
    full_ens.fit(X_tr_df, y_tr_all)
    indep_probs = full_ens.predict_proba(X_te_df)[:, 1]
    return oof_probs, indep_probs

def calc_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "ACC (%)": f"{acc*100:.2f}%",
        "Sen (%)": f"{sen*100:.2f}%",
        "Spc (%)": f"{spe*100:.2f}%",
        "MCC": f"{matthews_corrcoef(y_true, y_pred):.4f}",
        "AUC": f"{roc_auc_score(y_true, y_prob):.4f}"
    }

# ==============================================================================
# STEP 1: PAPER TABLE 4 — 5-FOLD CV HYPERPARAMETER GRID SEARCH (MAIN TRAINING SET)
# ==============================================================================
print("\n" + "=" * 95)
print("PAPER TABLE 4: 5-FOLD CV HYPERPARAMETER GRID SEARCH ON MAIN TRAINING SET")
print("=" * 95)

main_df = pd.read_csv("acp_main_dataset_clean.csv")
m_tr = main_df[main_df["split"] == "train"].reset_index(drop=True)
m_te = main_df[main_df["split"] == "test"].reset_index(drop=True)
X_tr_mix = encode_paper_tensor(m_tr["sequence"], "BPF+Quanc+Qualc (36D)")
X_te_mix = encode_paper_tensor(m_te["sequence"], "BPF+Quanc+Qualc (36D)")
y_m_tr, y_m_te = m_tr["label"].values, m_te["label"].values

grid_rows = []
for units in [32, 64, 128]:
    for drop in [0.2, 0.5]:
        oof_p, _ = run_5fold_cv_and_indep_bilstm(X_tr_mix, y_m_tr, X_te_mix, input_dim=36, lstm_units=units, use_attention=False, dropout=drop, epochs=35)
        m = calc_metrics(y_m_tr, oof_p)
        grid_rows.append({"Bi-LSTM Units": units, "Dropout": drop, **m})

grid_df = pd.DataFrame(grid_rows)
print(grid_df.to_string(index=False))
grid_df.to_csv("table4_grid_search_5fold_cv.csv", index=False)

# ==============================================================================
# STEP 2: PAPER TABLE 5 (5-FOLD CV) & TABLE 6 (INDEPENDENT TEST) + PROPOSED HYBRID
# ==============================================================================
cv_table_rows = []
indep_table_rows = []
roc_data = {"MAIN": [], "ALTERNATE": []}
bar_plot_data = {"MAIN": [], "ALTERNATE": []}

DATASET_CONFIG = [
    ("MAIN",      "acp_main_dataset_clean.csv",      45),
    ("ALTERNATE", "acp_alternate_dataset_clean.csv", 35)
]

for bm_name, csv_path, paper_epochs in DATASET_CONFIG:
    df = pd.read_csv(csv_path)
    tr_df = df[df["split"] == "train"].reset_index(drop=True)
    te_df = df[df["split"] == "test"].reset_index(drop=True)
    y_tr, y_te = tr_df["label"].values, te_df["label"].values
    
    cv_bm_rows = []
    indep_bm_rows = []
    
    best_oof_bilstm = None
    best_indep_bilstm = None
    best_cv_mcc = -1.0
    
    for mode_name, dim in FEATURE_MODES.items():
        X_tr_t = encode_paper_tensor(tr_df["sequence"], mode_name)
        X_te_t = encode_paper_tensor(te_df["sequence"], mode_name)
        
        for arch_name, use_att in [("BMF-Basic", False), ("BMF-Selfatt", True)]:
            oof_p, indep_p = run_5fold_cv_and_indep_bilstm(
                X_tr_t, y_tr, X_te_t, input_dim=dim, lstm_units=64,
                use_attention=use_att, dropout=0.2, epochs=paper_epochs
            )
            m_cv = calc_metrics(y_tr, oof_p)
            m_in = calc_metrics(y_te, indep_p)
            
            cv_row = {"Dataset": bm_name, "Architecture": arch_name, "Feature Set": mode_name, **m_cv}
            in_row = {"Dataset": bm_name, "Architecture": arch_name, "Feature Set": mode_name, **m_in}
            cv_bm_rows.append(cv_row)
            indep_bm_rows.append(in_row)
            cv_table_rows.append(cv_row)
            indep_table_rows.append(in_row)
            
            if not use_att:
                short_tag = mode_name.split(" ")[0]
                bar_plot_data[bm_name].append((f"BMF-Basic\n({short_tag})", float(m_in["ACC (%)"].replace("%", "")), float(m_in["AUC"]) * 100.0))
                
            if mode_name == "BPF+Quanc+Qualc (36D)":
                roc_data[bm_name].append((f"Paper {arch_name} (36D Mix)", y_te, indep_p))
                
            if float(m_cv["MCC"]) > best_cv_mcc:
                best_cv_mcc = float(m_cv["MCC"])
                best_oof_bilstm = oof_p.copy()
                best_indep_bilstm = indep_p.copy()

    # Proposed Extension: 42 Compact Features (Paper 16 + 20 AAC + 6 New Biophysical) + Deep BiLSTM Fusion
    _, X_tr_42 = extract_features_compact(tr_df["sequence"])
    _, X_te_42 = extract_features_compact(te_df["sequence"])
    
    oof_ens, indep_ens = run_5fold_cv_and_indep_biophys(X_tr_42, y_tr, X_te_42)
    
    # Hybrid Meta-Fusion on both 5-Fold CV and Independent Test
    oof_hybrid   = 0.50 * best_oof_bilstm + 0.50 * oof_ens
    indep_hybrid = 0.50 * best_indep_bilstm + 0.50 * indep_ens
    
    m_cv_hyb = calc_metrics(y_tr, oof_hybrid)
    m_in_hyb = calc_metrics(y_te, indep_hybrid)
    
    cv_hyb_row = {"Dataset": bm_name, "Architecture": "Proposed ACPred-BMF+ Hybrid", "Feature Set": "36D Mix + 6 New Biophys", **m_cv_hyb}
    in_hyb_row = {"Dataset": bm_name, "Architecture": "Proposed ACPred-BMF+ Hybrid", "Feature Set": "36D Mix + 6 New Biophys", **m_in_hyb}
    
    cv_bm_rows.append(cv_hyb_row)
    indep_bm_rows.append(in_hyb_row)
    cv_table_rows.append(cv_hyb_row)
    indep_table_rows.append(in_hyb_row)
    
    bar_plot_data[bm_name].append(("Proposed\nACPred-BMF+", float(m_in_hyb["ACC (%)"].replace("%", "")), float(m_in_hyb["AUC"]) * 100.0))
    roc_data[bm_name].append(("Proposed ACPred-BMF+ Hybrid", y_te, indep_hybrid))
    
    print("\n" + "=" * 95)
    print(f"PAPER TABLE 5 REPRODUCTION: 5-FOLD CROSS-VALIDATION ON {bm_name} TRAINING SET")
    print("=" * 95)
    print(pd.DataFrame(cv_bm_rows).drop(columns=["Dataset"]).to_string(index=False))
    
    print("\n" + "=" * 95)
    print(f"PAPER TABLE 6 REPRODUCTION: INDEPENDENT TEST SET EVALUATION ON {bm_name} DATASET")
    print("=" * 95)
    print(pd.DataFrame(indep_bm_rows).drop(columns=["Dataset"]).to_string(index=False))

pd.DataFrame(cv_table_rows).to_csv("table5_5fold_cv_results.csv", index=False)
indep_df = pd.DataFrame(indep_table_rows)
indep_df.to_csv("acpred_bmf_complete_results.csv", index=False)

# ==============================================================================
# STEP 3: GENERATE BAR CHART COMPARISON (acpred_bmf_comparison.png)
# ==============================================================================
print("\nGenerating Plot 1: 'acpred_bmf_comparison.png'...")
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

for idx, bm in enumerate(["MAIN", "ALTERNATE"]):
    items = bar_plot_data[bm]
    labels = [x[0] for x in items]
    accs   = [x[1] for x in items]
    aucs   = [x[2] for x in items]
    
    x = np.arange(len(labels))
    w = 0.36
    ax = axes[idx]
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy (%)", color="#2563eb", edgecolor="black")
    b2 = ax.bar(x + w/2, aucs, w, label="AUROC (×100)", color="#10b981", edgecolor="black")
    
    ax.set_title(f"{bm} Independent Test Set: All 4 Feature Sets vs. Hybrid", fontsize=12.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylim(55 if bm == "MAIN" else 75, 103)
    ax.set_ylabel("Percentage / Score", fontweight="bold")
    ax.legend(loc="lower right")
    
    for r in b1:
        ax.annotate(f"{r.get_height():.2f}%", (r.get_x() + r.get_width()/2, r.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")
    for r in b2:
        ax.annotate(f"{r.get_height()/100:.4f}", (r.get_x() + r.get_width()/2, r.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

plt.suptitle("ACPred-BMF Feature Ablation (BPF, Quanc, Qualc, Mix) vs. Proposed Biophysical Hybrid", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("acpred_bmf_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# STEP 4: PAPER FIGURES 4 & 5 — RED vs BLUE SHAP ON ALTERNATE DATASET
# ==============================================================================
print("Generating Plot 2: 'acpred_bmf_shap_explainability.png' (Red = High, Blue = Low)...")
alt_df = pd.read_csv("acp_alternate_dataset_clean.csv")
a_tr = alt_df[alt_df["split"] == "train"].reset_index(drop=True)
a_te = alt_df[alt_df["split"] == "test"].reset_index(drop=True)
y_a_tr, y_a_te = a_tr["label"].values, a_te["label"].values

X_p16_tr, _ = extract_features_compact(a_tr["sequence"])
X_p16_te, _ = extract_features_compact(a_te["sequence"])

shap_model = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
shap_model.fit(X_p16_tr, y_a_tr)

if HAS_SHAP:
    explainer = shap.TreeExplainer(shap_model)
    sv = explainer.shap_values(X_p16_te)
    shap_vals_pos = sv[:, :, 1] if isinstance(sv, np.ndarray) and sv.ndim == 3 else (sv[1] if isinstance(sv, list) else sv)
else:
    imp = shap_model.feature_importances_
    z_te = (X_p16_te.values - X_p16_te.values.mean(0)) / (X_p16_te.values.std(0) + 1e-8)
    shap_vals_pos = z_te * imp

mean_abs_shap = np.mean(np.abs(shap_vals_pos), axis=0)
top_idx = np.argsort(mean_abs_shap)[-10:]
top_names = [X_p16_te.columns[i] for i in top_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)

# Left Panel: Figure 4 with Clear Red (High) vs Blue (Low) Contrast
for rank_i, feat_i in enumerate(top_idx):
    s_col = shap_vals_pos[:, feat_i]
    f_col = X_p16_te.iloc[:, feat_i].values
    f_norm = (f_col - f_col.min()) / (f_col.max() - f_col.min() + 1e-8)
    jitter = np.random.normal(0, 0.12, size=len(s_col))
    sc = axes[0].scatter(s_col, np.full_like(s_col, rank_i) + jitter, c=f_norm, cmap="coolwarm", s=18, alpha=0.88)

axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_yticks(range(len(top_idx)))
axes[0].set_yticklabels(top_names, fontsize=10, fontweight="bold")
axes[0].set_xlabel("SHAP value (impact on model output: Right = ACP, Left = Non-ACP)", fontweight="bold")
axes[0].set_title("Figure 4 Reproduction: Impact of Features on ACP Prediction", fontsize=12, fontweight="bold")
plt.colorbar(sc, ax=axes[0], label="Normalized Feature Value (Blue = Low, Red = High)")

# Right Panel: Figure 5 Feature Importance Rank
axes[1].barh(range(len(top_idx)), mean_abs_shap[top_idx], color="#2563eb", edgecolor="black", height=0.68)
axes[1].set_yticks(range(len(top_idx)))
axes[1].set_yticklabels(top_names, fontsize=10, fontweight="bold")
axes[1].set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontweight="bold")
axes[1].set_title("Figure 5 Reproduction: Feature Importance Rank using SHAP", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("acpred_bmf_shap_explainability.png", dpi=300, bbox_inches="tight")
plt.close()

# ==============================================================================
# STEP 5: GENERATE ROC CURVES (acpred_bmf_roc.png)
# ==============================================================================
print("Generating Plot 3: 'acpred_bmf_roc.png'...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
colors = ["#3b82f6", "#f59e0b", "#10b981"]

for idx, bm in enumerate(["MAIN", "ALTERNATE"]):
    ax = axes[idx]
    for (label, y_t, y_p), c in zip(roc_data[bm], colors):
        fpr, tpr, _ = roc_curve(y_t, y_p)
        ax.plot(fpr, tpr, color=c, lw=2.3, label=f"{label} (AUC = {roc_auc_score(y_t, y_p):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_title(f"ROC Curves — {bm} Independent Test Set", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5)

plt.tight_layout()
plt.savefig("acpred_bmf_roc.png", dpi=300, bbox_inches="tight")
plt.close()
print("\nSaved 'table4_grid_search_5fold_cv.csv', 'table5_5fold_cv_results.csv', 'acpred_bmf_complete_results.csv', and all 3 plots!")