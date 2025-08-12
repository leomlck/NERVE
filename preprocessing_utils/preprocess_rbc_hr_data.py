import os
import re
import torch
from scipy.io import loadmat
import pandas as pd

# Define input and output directories
s = 100
ext = '_Kong22'
input_dir = f"/ministorage/RBC/FC_results/PNC_schaefer{s}_17Net_order"+ext
output_dir = f"/midtier/sablab/scratch/lem4012/data/rbc-data-hr-{s}"+ext

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Regex pattern to match files like "sub-4211615439_ses-PNC1_task-rest_acq-singleband_fc.mat"
# and capture the subject ID (digits following "sub-")
pattern = re.compile(r"sub-(\d+)_.*_fc\.mat$")

# List all files in input_dir that end with '_fc.mat'
files = [f for f in os.listdir(input_dir) if f.endswith('_fc.mat')]
files.sort()

# List to keep track of files that contain NaN values
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

    # Convert the matrix to a PyTorch tensor (float32)
    fc_tensor = torch.tensor(fc_mat, dtype=torch.float32)

    # Check for NaN values in the tensor
    if torch.isnan(fc_tensor).any():
        print(f"Warning: NaN values detected in {fname}.")
        nan_files.append({
            "subject_id": subject_id,
            "filename": fname,
            "path": mat_path
        })
        continue

    # Define the output filename and path, e.g. "sub-4211615439_fc.pt"
    out_fname = f"sub-{subject_id}_fc.pt"
    out_path = os.path.join(output_dir, out_fname)

    # Save the tensor as a .pt file
    torch.save(fc_tensor, out_path)
    print(f"Processed {i+1}/{len(files)}: {fname} -> {out_fname}")

# After processing, save the list of files with NaNs to a CSV
if nan_files:
    nan_csv_path = os.path.join(output_dir, "nan_files.csv")
    df_nan = pd.DataFrame(nan_files)
    df_nan.to_csv(nan_csv_path, index=False)
    print(f"CSV file with NaN filenames created: {nan_csv_path}")
else:
    print("No files with NaN values detected.")

print("All valid FC matrices saved as .pt files.")
