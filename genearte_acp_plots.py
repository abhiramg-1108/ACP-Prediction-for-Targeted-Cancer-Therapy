import itertools
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve

# 1. Load the CSV files that run_acp_pipeline.py already created
res_df = pd.read_csv("acp_benchmark_results.csv")
main_df = pd.read_csv("acp_main_dataset_clean.csv")
alt_df = pd.read_csv("acp_alternate_dataset_clean.csv")

plt.style.use("seaborn-v0_8-whitegrid")

# =====================================================================
# PLOT 1: Ablation Bar Chart (acp_ablation_comparison.png)
# =====================================================================
print("Generating Plot 1: acp_ablation_comparison.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

short_labels = [
    "1. SVM-Poly\n(424 Base)",
    "2. SVM-RBF\n(424 Base)",
    "3. Random Forest\n(424 Base)",
    "4. Mit. A: Tuned SVM\n(508 Biophysical)",
    "5. Mit. B: Hybrid\nEnsemble (508 Feats)"
]

for idx, bm in enumerate(["MAIN", "ALTERNATE"]):
    sub = res_df[res_df["Dataset"] == bm].copy()
    acc_vals = [float(str(x).replace("%", "")) for x in sub["Accuracy (%)"]]
    auc_vals = [float(x) * 100.0 for x in sub["AUROC"]]
    
    x = np.arange(len(short_labels))
    w = 0.36
    ax = axes[idx]
    b1 = ax.bar(x - w/2, acc_vals, w, label="Accuracy (%)", color="#2563eb", edgecolor="black", alpha=0.9)
    b2 = ax.bar(x + w/2, auc_vals, w, label="AUROC (×100)", color="#10b981", edgecolor="black", alpha=0.9)
    
    ax.set_title(f"{bm} Benchmark Dataset (100% Uncut)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, fontweight="bold")
    ax.set_ylabel("Percentage / Score", fontsize=11, fontweight="bold")
    ax.set_ylim(50 if bm == "MAIN" else 70, 103)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    
    for rect in b1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f"{h/100:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

plt.suptitle("Anticancer Peptide (ACP) Classification: Baseline vs. Biophysical Hybrid Ensemble", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("acp_ablation_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved 'acp_ablation_comparison.png' successfully!")

# =====================================================================
# PLOT 2: ROC Curves on Official Independent Test Sets (acp_roc_curves.png)
# =====================================================================
print("Generating Plot 2: acp_roc_curves.png ...")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACIDS = sorted(list(VALID_AA))
DIPEPTIDES = ["".join(pair) for pair in itertools.product(AMINO_ACIDS, repeat=2)]

BIOPHYS_SCALES = {
    'A': ( 1.8, 1.42, 0.83, 0.66,  0.0,  1.81), 'C': ( 2.5, 0.70, 1.19, 1.19,  0.0,  1.28),
    'D': (-3.5, 1.01, 0.54, 1.46, -1.0, -8.72), 'E': (-3.5, 1.51, 0.37, 0.74, -1.0, -6.81),
    'F': ( 2.8, 1.13, 1.38, 0.60,  0.0,  2.98), 'G': (-0.4, 0.57, 0.75, 1.56,  0.0,  0.94),
    'H': (-3.2, 1.00, 0.87, 0.95,  0.1, -4.66), 'I': ( 4.5, 1.08, 1.60, 0.47,  0.0,  4.92),
    'K': (-3.9, 1.16, 0.74, 1.01,  1.0, -5.55), 'L': ( 3.8, 1.21, 1.30, 0.59,  0.0,  4.92),
    'M': ( 1.9, 1.45, 1.05, 0.60,  0.0,  2.35), 'N': (-3.5, 0.67, 0.89, 1.56,  0.0, -6.64),
    'P': (-1.6, 0.57, 0.55, 1.52,  0.0,  0.00), 'Q': (-3.5, 1.11, 1.10, 0.98,  0.0, -5.54),
    'R': (-4.5, 0.98, 0.93, 0.95,  1.0, -14.92),'S': (-0.8, 0.77, 0.75, 1.43,  0.0, -3.40),
    'T': (-0.7, 0.83, 1.19, 0.96,  0.0, -2.57), 'V': ( 4.2, 1.06, 1.70, 0.50,  0.0,  4.04),
    'W': (-0.9, 1.08, 1.37, 0.96,  0.0,  2.33), 'Y': (-1.3, 0.69, 1.47, 1.14,  0.0, -0.14)
}

def extract_features_for_df(df):
    base_rows, enh_rows = [], []
    for seq in df["sequence"]:
        L = len(seq)
        aa_counts = Counter(seq)
        f_base = {f"AAC_{aa}": aa_counts[aa] / L for aa in AMINO_ACIDS}
        dipeps = [seq[i:i+2] for i in range(L - 1)]
        dp_counts = Counter(dipeps)
        denom = max(L - 1, 1)
        for dp in DIPEPTIDES:
            f_base[f"DPC_{dp}"] = dp_counts[dp] / denom
        props = np.array([BIOPHYS_SCALES[aa] for aa in seq])
        f_base["PHYS_GRAVY"]       = np.mean(props[:, 0])
        f_base["PHYS_Alpha_Helix"] = np.mean(props[:, 1])
        f_base["PHYS_Beta_Sheet"]  = np.mean(props[:, 2])
        f_base["PHYS_Turn"]        = np.mean(props[:, 3])
        base_rows.append(f_base)
        
        f_enh = dict(f_base)
        f_enh["ACP_Net_Charge_pH74"] = np.sum(props[:, 4])
        f_enh["ACP_Charge_Density"]  = np.mean(props[:, 4])
        f_enh["ACP_Boman_Index"]     = -np.mean(props[:, 5])
        angles = np.deg2rad(np.arange(L) * 100.0)
        h_vals = props[:, 0]
        f_enh["ACP_Hydrophobic_Moment_Helix"] = np.sqrt((np.sum(h_vals * np.sin(angles)) / L)**2 + (np.sum(h_vals * np.cos(angles)) / L)**2)
        angles_b = np.deg2rad(np.arange(L) * 160.0)
        f_enh["ACP_Hydrophobic_Moment_Sheet"] = np.sqrt((np.sum(h_vals * np.sin(angles_b)) / L)**2 + (np.sum(h_vals * np.cos(angles_b)) / L)**2)
        f_enh["ACP_Cationic_Frac"]    = sum(aa_counts[c] for c in "KRH") / L
        f_enh["ACP_Anionic_Frac"]     = sum(aa_counts[c] for c in "DE") / L
        f_enh["ACP_Aromaticity"]      = sum(aa_counts[c] for c in "FWY") / L
        f_enh["ACP_Aliphatic_Index"]  = (aa_counts['A'] + 2.9 * aa_counts['V'] + 3.9 * (aa_counts['I'] + aa_counts['L'])) / L * 100.0
        f_enh["ACP_Hydrophobic_Frac"] = sum(aa_counts[c] for c in "AILMFWV") / L
        f_enh["ACP_Length_Log"]       = np.log1p(L)
        for k_win in [5, 10]:
            n_slice = props[:min(L, k_win)]
            c_slice = props[-min(L, k_win):]
            for col_i, col_name in enumerate(["GRAVY", "Helix", "Sheet", "Turn", "Charge"]):
                f_enh[f"NTerm{k_win}_{col_name}"] = np.mean(n_slice[:, col_i])
                f_enh[f"CTerm{k_win}_{col_name}"] = np.mean(c_slice[:, col_i])
        mean_p = np.mean(props[:, :5], axis=0)
        std_p  = np.std(props[:, :5], axis=0) + 1e-8
        norm_p = (props[:, :5] - mean_p) / std_p
        for lag in range(1, 9):
            corr = np.mean(norm_p[:-lag] * norm_p[lag:], axis=0) if L > lag else np.zeros(5)
            for col_i, col_name in enumerate(["GRAVY", "Helix", "Sheet", "Turn", "Charge"]):
                f_enh[f"PseAAC_Lag{lag}_{col_name}"] = corr[col_i]
        gapped_pairs = [seq[i] + seq[i+2] for i in range(L - 2)]
        gp_counts = Counter(gapped_pairs)
        gp_denom = max(L - 2, 1)
        for motif in ["KK", "KL", "LK", "LL", "KW", "WK", "AA", "AK", "KA", "FL", "LF", "WW", "RR", "RW"]:
            f_enh[f"CKSAAP_Gap1_{motif}"] = gp_counts[motif] / gp_denom
        enh_rows.append(f_enh)
    return pd.DataFrame(base_rows), pd.DataFrame(enh_rows)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
colors = ["#94a3b8", "#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"]
lw_list = [1.5, 2.0, 2.0, 2.0, 2.8]

for idx, (bm, df) in enumerate([("MAIN", main_df), ("ALTERNATE", alt_df)]):
    Xb, Xe = extract_features_for_df(df)
    tr = (df["split"] == "train").values
    te = (df["split"] == "test").values
    y_tr, y_te = df.loc[tr, "label"].values, df.loc[te, "label"].values
    
    Xb_tr, Xb_te = Xb.loc[tr], Xb.loc[te]
    Xe_tr, Xe_te = Xe.loc[tr], Xe.loc[te]
    
    models = [
        ("1. Baseline: SVM-Poly", Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="poly", degree=3, probability=True, random_state=42))]), Xb_tr, Xb_te),
        ("2. Baseline: SVM-RBF", Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]), Xb_tr, Xb_te),
        ("3. Baseline: Random Forest", Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]), Xb_tr, Xb_te),
        ("4. Mit. A: Tuned SVM (508 Feats)", Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(score_func=f_classif, k=300)), ("clf", SVC(kernel="rbf", C=2.5, gamma="scale", class_weight="balanced", probability=True, random_state=42))]), Xe_tr, Xe_te),
        ("5. Mit. B: Proposed Hybrid Ensemble", Pipeline([("scaler", StandardScaler()), ("select", SelectKBest(score_func=f_classif, k=350)), ("clf", VotingClassifier(estimators=[("svm", SVC(kernel="rbf", C=2.5, gamma="scale", class_weight="balanced", probability=True, random_state=42)), ("et", ExtraTreesClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)), ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)), ("gb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42))], voting="soft", weights=[2, 3, 2, 2]))]), Xe_tr, Xe_te)
    ]
    
    ax = axes[idx]
    for (name, pipe, X_tr, X_te), c, lw in zip(models, colors, lw_list):
        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc_val = roc_auc_score(y_te, y_prob)
        ax.plot(fpr, tpr, color=c, lw=lw, label=f"{name} (AUC = {auc_val:.4f})")
        
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_title(f"ROC Curves — {bm} Independent Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

plt.tight_layout()
plt.savefig("acp_roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved 'acp_roc_curves.png' successfully!")