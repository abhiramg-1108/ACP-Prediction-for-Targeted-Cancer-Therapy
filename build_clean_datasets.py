import os
import pandas as pd

# The 20 standard amino acids
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def load_seq_file(filepath):
    """Reads sequences from official IIIT-Delhi AntiCP 2.0 text/FASTA files."""
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
    base_dir = "raw_anticp2_data"
    files = {
        ("train", 1): os.path.join(base_dir, f"pos_train_{benchmark_type}"),
        ("train", 0): os.path.join(base_dir, f"neg_train_{benchmark_type}"),
        ("test",  1): os.path.join(base_dir, f"pos_test_{benchmark_type}"),
        ("test",  0): os.path.join(base_dir, f"neg_test_{benchmark_type}"),
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
    out_csv = f"acp_{benchmark_type}_dataset_clean.csv"
    df.to_csv(out_csv, index=False)
    
    print(f"=== Created '{out_csv}' (Total: {len(df)} sequences) ===")
    print(df.groupby(["split", "label"])["length"].agg(["count", "min", "max", "mean"]).round(2))
    print()
    return df

if __name__ == "__main__":
    build_benchmark_df("main")
    build_benchmark_df("alternate")