import os
import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, 
    matthews_corrcoef, f1_score, confusion_matrix
)

# Import DL modules
from dl_acpred_bmf import extract_dl_features_for_df, KerasDLWrapper

def compute_metrics(y_true, y_pred, y_prob, train_time):
    acc = accuracy_score(y_true, y_pred)
    sen = recall_score(y_true, y_pred, pos_label=1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
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

print("==================================================")
print("Running Expanded ACPred Pipeline with Deep Learning")
print("==================================================")

all_results = []

for bm_type in ["main", "alternate"]:
    print(f"\nProcessing DL Pipeline for {bm_type.upper()} dataset...")
    df = pd.read_csv(f"acp_{bm_type}_dataset_clean.csv")
    
    train_mask = (df["split"] == "train").values
    test_mask  = (df["split"] == "test").values
    
    y_train = df.loc[train_mask, "label"].values
    y_test  = df.loc[test_mask, "label"].values
    
    # 1. Extract DL Features (3D Tensors: N x 50 x 36)
    print(f"Extracting 3D Tensors (Shape: N x 50 x 36)...")
    X_dl = extract_dl_features_for_df(df, max_length=50)
    Xdl_tr, Xdl_te = X_dl[train_mask], X_dl[test_mask]
    
    # 2. Build DL Model Wrapper 
    # Paper parameters: Main dataset uses 45 epochs, Alternate uses 35 epochs
    epochs_val = 45 if bm_type == "main" else 35
    lstm_val = 64 if bm_type == "main" else 128
    
    dl_model = KerasDLWrapper(epochs=epochs_val, lstm_units=lstm_val, batch_size=32)
    
    print(f"Training Bi-LSTM+Attention Model (Epochs: {epochs_val}, LSTM Units: {lstm_val})...")
    t0 = time.perf_counter()
    dl_model.fit(Xdl_tr, y_train)
    elapsed = time.perf_counter() - t0
    
    # 3. Predict & Compute Metrics
    y_pred = dl_model.predict(Xdl_te)
    y_prob = dl_model.predict_proba(Xdl_te)[:, 1]
    
    m = compute_metrics(y_test, y_pred, y_prob, elapsed)
    row = {"Dataset": bm_type.upper(), "Model": "6. ACPred-BMF (Bi-LSTM+Attention)", **m}
    
    print(f"DL Results: ACC={m['Accuracy (%)']} | AUC={m['AUROC']}")
    all_results.append(row)

# Append DL results to a new CSV
dl_df = pd.DataFrame(all_results)
dl_df.to_csv("acp_benchmark_results_with_dl.csv", index=False)
print("\nDeep learning metrics successfully saved to 'acp_benchmark_results_with_dl.csv'")