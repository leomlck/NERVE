import os
import re
import torch
from scipy.io import loadmat
import pandas as pd

# ---------------- Configuration ----------------
s = 400
ext = ''
input_dir = f"PATH/TO/NCANDA/FC_results/NCANDA_schaefer400_17Net_order_Kong2022_additional_gsr_on_400"+ext
output_dir = f"data/ncanda-data-hr"+ext

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex to match files like "sub-colornest001_combined_fc.mat"
pattern = re.compile(r"sub-(NCANDAS\d+)_ses-baseline_fc\.mat$")

# List all *_fc.mat files
files = [f for f in os.listdir(input_dir) if f.endswith('_fc.mat')]
files.sort()

# Track files with NaNs
nan_files = []

# ---------------- Processing ----------------
for i, fname in enumerate(files):
    match = pattern.search(fname)
    if not match:
        print(f"Skipping file with unexpected name: {fname}")
        continue

    subject_id = match.group(1)
    mat_path = os.path.join(input_dir, fname)

    try:
        data = loadmat(mat_path)
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        continue

    if 'fc' not in data:
        print(f"File {fname} does not contain key 'fc'. Skipping.")
        continue

    fc_mat = data['fc']
    if fc_mat.shape != (s, s):
        print(f"Warning: {fname} has shape {fc_mat.shape}, expected (s,s). Skipping.")
        continue

    fc_tensor = torch.tensor(fc_mat, dtype=torch.float32)

    if torch.isnan(fc_tensor).any():
        print(f"Warning: NaN values detected in {fname}.")
        nan_files.append({
            "subject_id": subject_id,
            "filename": fname,
            "path": mat_path
        })
        continue

    out_fname = f"sub-{subject_id}_fc.pt"
    out_path = os.path.join(output_dir, out_fname)
    torch.save(fc_tensor, out_path)

    print(f"Processed {i+1}/{len(files)}: {fname} -> {out_fname}")

# ---------------- Save NaN Report ----------------
if nan_files:
    nan_csv_path = os.path.join(output_dir, "nan_files.csv")
    pd.DataFrame(nan_files).to_csv(nan_csv_path, index=False)
    print(f"CSV file with NaN filenames created: {nan_csv_path}")
else:
    print("No files with NaN values detected.")

print("✅ All valid FC matrices saved as .pt files.")
