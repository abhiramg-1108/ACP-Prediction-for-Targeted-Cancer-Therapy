import os
import time
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    matthews_corrcoef,
    f1_score,
    confusion_matrix
)

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACIDS = sorted(list(VALID_AA))
DIPEPTIDES = ["".join(pair) for pair in itertools.product(AMINO_ACIDS, repeat=2)]

# ==========================================
# PHASE 1: Load 100% of Official Benchmark Files
# ==========================================
def load_seq_file(filepath):
    seqs = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    if any(l.startswith(">") for l in lines):
        curr = []
        for l in lines:
            if l.startswith(">"):
                if curr:
                    s = "".join(curr).upper()
                    if set(s).issubset(VALID_AA) and len(s) >= 4:
                        seqs.append(s)
                    curr = []
            else:
                curr.append(l)
        if curr:
            s = "".join(curr).upper()
            if set(s).issubset(VALID_AA) and len(s) >= 4:
                seqs.append(s)
    else:
        for l in lines:
            s = l.split()[0].upper()
            if set(s).issubset(VALID_AA) and len(s) >= 4:
                seqs.append(s)
    return seqs

def build_benchmark_df(benchmark_type="main"):
    base = "raw_anticp2_data"
    files = {
        ("train", 1): os.path.join(base, f"pos_train_{benchmark_type}"),
        ("train", 0): os.path.join(base, f"neg_train_{benchmark_type}"),
        ("test",  1): os.path.join(base, f"pos_test_{benchmark_type}"),
        ("test",  0): os.path.join(base, f"neg_test_{benchmark_type}"),
    }
    rows = []
    for (split, label), fpath in files.items():
        seqs = load_seq_file(fpath)
        for idx, s in enumerate(seqs):
            rows.append({
                "peptide_id": f"{benchmark_type.upper()}_{split.upper()}_{'POS' if label==1 else 'NEG'}_{idx+1}",
                "sequence": s,
                "length": len(s),
                "split": split,
                "label": label
            })
    df = pd.DataFrame(rows)
    df.to_csv(f"acp_{benchmark_type}_dataset_clean.csv", index=False)
    return df

# ==========================================
# PHASE 2 & 4: Baseline (424) + Biophysical ACP Features (84)
# ==========================================
# Format: 'AA': (GRAVY, Alpha_Helix, Beta_Sheet, Turn, Charge_pH7.4, Boman_Solubility)
BIOPHYS_SCALES = {
    'A': ( 1.8, 1.42, 0.83, 0.66,  0.0,  1.81),
    'C': ( 2.5, 0.70, 1.19, 1.19,  0.0,  1.28),
    'D': (-3.5, 1.01, 0.54, 1.46, -1.0, -8.72),
    'E': (-3.5, 1.51, 0.37, 0.74, -1.0, -6.81),
    'F': ( 2.8, 1.13, 1.38, 0.60,  0.0,  2.98),
    'G': (-0.4, 0.57, 0.75, 1.56,  0.0,  0.94),
    'H': (-3.2, 1.00, 0.87, 0.95,  0.1, -4.66),
    'I': ( 4.5, 1.08, 1.60, 0.47,  0.0,  4.92),
    'K': (-3.9, 1.16, 0.74, 1.01,  1.0, -5.55),
    'L': ( 3.8, 1.21, 1.30, 0.59,  0.0,  4.92),
    'M': ( 1.9, 1.45, 1.05, 0.60,  0.0,  2.35),
    'N': (-3.5, 0.67, 0.89, 1.56,  0.0, -6.64),
    'P': (-1.6, 0.57, 0.55, 1.52,  0.0,  0.00),
    'Q': (-3.5, 1.11, 1.10, 0.98,  0.0, -5.54),
    'R': (-4.5, 0.98, 0.93, 0.95,  1.0, -14.92),
    'S': (-0.8, 0.77, 0.75, 1.43,  0.0, -3.40),
    'T': (-0.7, 0.83, 1.19, 0.96,  0.0, -2.57),
    'V': ( 4.2, 1.06, 1.70, 0.50,  0.0,  4.04),
    'W': (-0.9, 1.08, 1.37, 0.96,  0.0,  2.33),
    'Y': (-1.3, 0.69, 1.47, 1.14,  0.0, -0.14)
}

def extract_features_for_df(df):
    base_rows = []
    enh_rows = []
    
    for seq in df["sequence"]:
        L = len(seq)
        aa_counts = Counter(seq)
        
        # 1. Baseline 20 AAC
        f_base = {f"AAC_{aa}": aa_counts[aa] / L for aa in AMINO_ACIDS}
        
        # 2. Baseline 400 DPC
        dipeps = [seq[i:i+2] for i in range(L - 1)]
        dp_counts = Counter(dipeps)
        denom = max(L - 1, 1)
        for dp in DIPEPTIDES:
            f_base[f"DPC_{dp}"] = dp_counts[dp] / denom
            
        # 3. Baseline 4 Physicochemical
        props = np.array([BIOPHYS_SCALES[aa] for aa in seq])
        f_base["PHYS_GRAVY"]       = np.mean(props[:, 0])
        f_base["PHYS_Alpha_Helix"] = np.mean(props[:, 1])
        f_base["PHYS_Beta_Sheet"]  = np.mean(props[:, 2])
        f_base["PHYS_Turn"]        = np.mean(props[:, 3])
        
        base_rows.append(f_base)
        
        # --- PHASE 4: Anticancer Membrane-Interaction & Sequence-Order Features ---
        f_enh = dict(f_base)
        
        # Electrostatic Net Charge & Charge Density at pH 7.4 (Critical for anionic cancer membranes)
        f_enh["ACP_Net_Charge_pH74"] = np.sum(props[:, 4])
        f_enh["ACP_Charge_Density"]  = np.mean(props[:, 4])
        
        # Boman Protein-Binding Index (negative of average solubility)
        f_enh["ACP_Boman_Index"]     = -np.mean(props[:, 5])
        
        # Eisenberg Alpha-Helical Hydrophobic Moment (Amphiphilicity at 100 degrees per residue)
        angles = np.deg2rad(np.arange(L) * 100.0)
        h_vals = props[:, 0]
        mu_sin = np.sum(h_vals * np.sin(angles)) / L
        mu_cos = np.sum(h_vals * np.cos(angles)) / L
        f_enh["ACP_Hydrophobic_Moment_Helix"] = np.sqrt(mu_sin**2 + mu_cos**2)
        
        # Beta-Sheet Hydrophobic Moment (160 degrees per residue)
        angles_b = np.deg2rad(np.arange(L) * 160.0)
        f_enh["ACP_Hydrophobic_Moment_Sheet"] = np.sqrt(
            (np.sum(h_vals * np.sin(angles_b)) / L)**2 + (np.sum(h_vals * np.cos(angles_b)) / L)**2
        )
        
        # Key Residue Group Fractions (Cationic, Anionic, Aromatic, Aliphatic, Hydrophobic, Polar)
        f_enh["ACP_Cationic_Frac"]    = sum(aa_counts[c] for c in "KRH") / L
        f_enh["ACP_Anionic_Frac"]     = sum(aa_counts[c] for c in "DE") / L
        f_enh["ACP_Aromaticity"]      = sum(aa_counts[c] for c in "FWY") / L
        f_enh["ACP_Aliphatic_Index"]  = (aa_counts['A'] + 2.9 * aa_counts['V'] + 3.9 * (aa_counts['I'] + aa_counts['L'])) / L * 100.0
        f_enh["ACP_Hydrophobic_Frac"] = sum(aa_counts[c] for c in "AILMFWV") / L
        f_enh["ACP_Length_Log"]       = np.log1p(L)
        
        # N-Terminal (first 5 & 10 residues) and C-Terminal (last 5 & 10 residues) Profiles
        for k_win in [5, 10]:
            n_slice = props[:min(L, k_win)]
            c_slice = props[-min(L, k_win):]
            for col_i, col_name in enumerate(["GRAVY", "Helix", "Sheet", "Turn", "Charge"]):
                f_enh[f"NTerm{k_win}_{col_name}"] = np.mean(n_slice[:, col_i])
                f_enh[f"CTerm{k_win}_{col_name}"] = np.mean(c_slice[:, col_i])
                
        # Sequence-Order PseAAC Autocorrelation (Lags 1..8 across 5 properties = 40 features)
        mean_p = np.mean(props[:, :5], axis=0)
        std_p  = np.std(props[:, :5], axis=0) + 1e-8
        norm_p = (props[:, :5] - mean_p) / std_p
        for lag in range(1, 9):
            corr = np.mean(norm_p[:-lag] * norm_p[lag:], axis=0) if L > lag else np.zeros(5)
            for col_i, col_name in enumerate(["GRAVY", "Helix", "Sheet", "Turn", "Charge"]):
                f_enh[f"PseAAC_Lag{lag}_{col_name}"] = corr[col_i]
                
        # Gapped Dipeptide (k-spaced amino acid pairs, gap=1) for top cationic/hydrophobic motifs
        gapped_pairs = [seq[i] + seq[i+2] for i in range(L - 2)]
        gp_counts = Counter(gapped_pairs)
        gp_denom = max(L - 2, 1)
        for motif in ["KK", "KL", "LK", "LL", "KW", "WK", "AA", "AK", "KA", "FL", "LF", "WW", "RR", "RW"]:
            f_enh[f"CKSAAP_Gap1_{motif}"] = gp_counts[motif] / gp_denom
            
        enh_rows.append(f_enh)
        
    return pd.DataFrame(base_rows), pd.DataFrame(enh_rows)

# ==========================================
# PHASE 3 & 5: Evaluate Baseline vs Proposed Models
# ==========================================
def compute_metrics(y_true, y_pred, y_prob, train_time):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1)  # Sensitivity (TPR)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # Specificity (TNR)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred)
    return {
        "Accuracy (%)": f"{acc*100:.2f}%",
        "Sensitivity (%)": f"{sen*100:.2f}%",
        "Specificity (%)": f"{spe*100:.2f}%",
        "F1-Score": f"{f1:.4f}",
        "MCC": f"{mcc:.4f}",
        "AUROC": f"{auc:.4f}",
        "Time (s)": f"{train_time:.3f}s"
    }

all_results = []

for bm_type in ["main", "alternate"]:
    print("\n" + "=" * 95)
    print(f"RUNNING OFFICIAL BENCHMARK: {bm_type.upper()} DATASET (100% UNCUT)")
    print("=" * 95)
    
    df = build_benchmark_df(bm_type)
    X_base, X_enh = extract_features_for_df(df)
    
    train_mask = (df["split"] == "train").values
    test_mask  = (df["split"] == "test").values
    
    y_train = df.loc[train_mask, "label"].values
    y_test  = df.loc[test_mask, "label"].values
    
    Xb_tr, Xb_te = X_base.loc[train_mask], X_base.loc[test_mask]
    Xe_tr, Xe_te = X_enh.loc[train_mask], X_enh.loc[test_mask]
    
    print(f"Total Sequences: {len(df)} | Official Train: {len(y_train)} | Official Independent Test: {len(y_test)}")
    print(f"Baseline Features: {Xb_tr.shape[1]} | Proposed Biophysical+PseAAC Features: {Xe_tr.shape[1]}\n")
    
    models = [
        (
            "1. Baseline: SVM-Polynomial (424 feats)",
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="poly", degree=3, probability=True, random_state=42))]),
            Xb_tr, Xb_te
        ),
        (
            "2. Baseline: SVM-RBF (424 feats)",
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
            Xb_tr, Xb_te
        ),
        (
            "3. Baseline: Random Forest (424 feats)",
            Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]),
            Xb_tr, Xb_te
        ),
        (
            "4. Mitigation A: Biophysical + PseAAC + Tuned SVM-RBF",
            Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=300)),
                ("clf", SVC(kernel="rbf", C=2.5, gamma="scale", class_weight="balanced", probability=True, random_state=42))
            ]),
            Xe_tr, Xe_te
        ),
        (
            "5. Mitigation B: Proposed Multi-View Hybrid Ensemble (508 feats)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("select", SelectKBest(score_func=f_classif, k=350)),
                ("clf", VotingClassifier(
                    estimators=[
                        ("svm", SVC(kernel="rbf", C=2.5, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
                        ("et", ExtraTreesClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
                        ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)),
                        ("gb", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42))
                    ],
                    voting="soft",
                    weights=[2, 3, 2, 2]
                ))
            ]),
            Xe_tr, Xe_te
        )
    ]
    
    bm_rows = []
    for name, pipe, X_tr, X_te in models:
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_train)
        elapsed = time.perf_counter() - t0
        
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        
        m = compute_metrics(y_test, y_pred, y_prob, elapsed)
        row = {"Dataset": bm_type.upper(), "Model": name, **m}
        bm_rows.append(row)
        all_results.append(row)
        
    bm_df = pd.DataFrame(bm_rows).drop(columns=["Dataset"])
    print(bm_df.to_string(index=False))

final_res_df = pd.DataFrame(all_results)
final_res_df.to_csv("acp_benchmark_results.csv", index=False)
print("\nSaved complete benchmark results to 'acp_benchmark_results.csv'!")