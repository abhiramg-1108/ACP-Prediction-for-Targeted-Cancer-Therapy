import os
import re
import urllib.request
import ssl
import zipfile
import tarfile
import pandas as pd

# Bypass SSL certificate expiration issues common on university/academic servers
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

print("Connecting to IIIT-Delhi Official AntiCP 2.0 Benchmark Server...")

# 1. Scrape the exact dataset file links from https://webs.iiitd.edu.in/raghava/anticp2/download.php
base_url = "https://webs.iiitd.edu.in/raghava/anticp2/"
download_page_url = base_url + "download.php"

req = urllib.request.Request(download_page_url, headers={"User-Agent": "Mozilla/5.0"})
html = urllib.request.urlopen(req, context=ssl_ctx, timeout=20).read().decode("utf-8", errors="ignore")

# Find all dataset links (.zip, .tar.gz, .txt, .fasta, .fa) on the download page
hrefs = re.findall(r'href=[\'"]?([^\'" >]+)', html)
data_links = [h for h in hrefs if any(ext in h.lower() for ext in [".txt", ".fa", ".fasta", ".zip", ".tar.gz", "dataset", "main", "alt"])]

print("Found official links on IIIT-Delhi page:", data_links)

os.makedirs("raw_anticp2_data", exist_ok=True)

for link in set(data_links):
    if link.startswith("http") and "iiitd.edu.in" not in link:
        continue
    full_url = link if link.startswith("http") else base_url + link.lstrip("/")
    fname = os.path.basename(full_url.split("?")[0])
    if not fname or fname.endswith(".php") or fname.endswith(".html"):
        continue
    out_path = os.path.join("raw_anticp2_data", fname)
    try:
        print(f"Downloading {full_url} -> {out_path} ...")
        req_file = urllib.request.Request(full_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_file, context=ssl_ctx, timeout=30) as resp, open(out_path, "wb") as f:
            f.write(resp.read())
            
        # Extract if archive
        if fname.endswith(".zip"):
            with zipfile.ZipFile(out_path, "r") as z:
                z.extractall("raw_anticp2_data")
        elif fname.endswith(".tar.gz") or fname.endswith(".tgz"):
            with tarfile.open(out_path, "r:gz") as t:
                t.extractall("raw_anticp2_data")
    except Exception as e:
        print(f"Skipped {fname}: {e}")

print("\nDownloaded files in 'raw_anticp2_data/':")
for root, dirs, files in os.walk("raw_anticp2_data"):
    for file in files:
        print(" -", os.path.join(root, file))