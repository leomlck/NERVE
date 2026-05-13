import os
import re
import torch
import pandas as pd

# ---------------- Configuration ----------------
# Input directory where the CCNP .pt files are stored.
data_path = 'data/ncanda-data-hr/'

# Output configuration: folder to save the feature CSV and the CSV file name.
path_to_features = 'outputs/features'
#output_csv_filename = 'features_nerve_CCNP_FC_FC_enc_FC_dec_nw_p1_CCNP_avg.csv'
output_csv_filename = 'rs_fmri_triu_400_ncanda.csv'

# Create the output directory if it does not exist.
os.makedirs(path_to_features, exist_ok=True)

# ----------------- Setup -----------------
# Regex pattern: expecting filenames like "sub-colornest001_fc.pt" to extract the subject ID.
pattern = re.compile(r"sub-(NCANDAS\d+)_fc\.pt$")

# Get all .pt files in the data path.
files = [f for f in os.listdir(data_path) if f.endswith('.pt')]
files.sort()

# Precompute upper triangular indices (excluding diagonal) for a 400x400 matrix.
triu_indices = torch.triu_indices(400, 400, offset=1)

# List to hold each subject's feature dictionary.
entries = []

# ---------------- Process Each File ----------------
for i, fname in enumerate(files):
    match = pattern.search(fname)
    if not match:
        print(f"Skipping file with unexpected name: {fname}")
        continue

    subject_id = match.group(1)
    pt_path = os.path.join(data_path, fname)

    try:
        fc_tensor = torch.load(pt_path)
    except Exception as e:
        print(f"Error loading {fname}: {e}")
        continue

    if fc_tensor.shape != (400, 400):
        print(f"Warning: {fname} has shape {fc_tensor.shape}, expected (400,400). Skipping.")
        continue

    # Extract the upper triangular part (excluding diagonal)
    upper_tri = fc_tensor[triu_indices[0], triu_indices[1]]
    upper_tri_np = upper_tri.cpu().numpy()

    # Create dictionary: subject id + features
    feature_dict = {'src_subject_id': subject_id}
    for j, value in enumerate(upper_tri_np):
        feature_dict[str(j)] = value

    entries.append(feature_dict)
    print(f"Processed {i+1}/{len(files)}: {fname}")

# ---------------- Save to CSV ----------------
df_features = pd.DataFrame(entries)
csv_output_path = os.path.join(path_to_features, output_csv_filename)
df_features.to_csv(csv_output_path, index=False)
print(f"✅ Feature CSV saved to {csv_output_path}")
