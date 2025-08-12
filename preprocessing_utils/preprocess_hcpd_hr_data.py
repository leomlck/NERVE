import os
import re
import torch
from scipy.io import loadmat
import pandas as pd

s = 100
input_dir = f"/ministorage/HCP_D/FC_results/HCP_D_schaefer{s}_17Net_order_AP_only"
output_dir = f"/midtier/sablab/scratch/lem4012/data/hcpd-data-hr-{s}/"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex pattern to match files like "sub-HCD1227740_combined_fc.mat"
# and extract the subject ID (e.g., HCD1227740)
pattern = re.compile(r"sub-(HCD\d+)_combined_fc\.mat$")

# List all *_fc.mat files
files = [f for f in os.listdir(input_dir) if f.endswith('_fc.mat')]
files.sort()

# Track files with NaNs
nan_files = []

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

# Save list of files with NaNs
if nan_files:
    nan_csv_path = os.path.join(output_dir, "nan_files.csv")
    pd.DataFrame(nan_files).to_csv(nan_csv_path, index=False)
    print(f"CSV file with NaN filenames created: {nan_csv_path}")
else:
    print("No files with NaN values detected.")

print("✅ All valid FC matrices saved as .pt files.")
