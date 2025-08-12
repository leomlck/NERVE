import os
import re
import torch
import pandas as pd

# ---------------- Configuration ----------------
# Input paths
data_path = '/midtier/sablab/scratch/lem4012/data/abcd-data-hr/'
save_folder = 'preprocessed'
save_filename = 'rs_fc.pt'

# Output CSV settings
path_to_features = '/midtier/sablab/scratch/lem4012/save/fc_mae_features'
output_csv_filename = 'features_fc_mae_ABCD_FC_FC_enc_FC_dec_nw_p1_ABCD_avg.csv'

# Create the output directory if it doesn't exist.
os.makedirs(path_to_features, exist_ok=True)

# Regex to filter subject folders (e.g., those containing "NDAR_")
regexp = re.compile(r'NDAR_')

# List all subject folders in the data path
subjects = [f for f in os.listdir(data_path) if regexp.search(f)]
subjects.sort()

# Precompute upper triangular indices for a 400x400 matrix.
# The number of upper triangular elements is 400*(400+1)//2.
triu_indices = torch.triu_indices(400, 400, offset=1)

# List to hold feature dictionaries (each will be one row in the CSV)
entries = []

# ---------------- Process Each Subject ----------------
for i, subject_id in enumerate(subjects):
    print(f'Processing {i+1}/{len(subjects)}: {subject_id}')
    
    # Build the path to the .pt file for this subject.
    pt_path = os.path.join(data_path, subject_id, save_folder, save_filename)
    
    # Check whether the file exists.
    if not os.path.exists(pt_path):
        print(f"Warning: {pt_path} does not exist. Skipping {subject_id}.")
        continue

    # Load the FC matrix tensor.
    try:
        fc_tensor = torch.load(pt_path)
    except Exception as e:
        print(f"Error loading {pt_path}: {e}")
        continue

    # Check that the tensor is 400x400.
    if fc_tensor.shape != (400, 400):
        print(f"Warning: {pt_path} has shape {fc_tensor.shape} (expected (400,400)). Skipping.")
        continue

    # Extract the upper triangular values (including diagonal)
    upper_tri_values = fc_tensor[triu_indices[0], triu_indices[1]]
    
    # Convert the values to a numpy array (ensuring the tensor is on CPU)
    upper_tri_values_np = upper_tri_values.cpu().numpy()
    
    # Create a dictionary entry for this subject.
    feature_dict = {'src_subject_id': subject_id}
    # Populate the dictionary with flattened vector values.
    for j, value in enumerate(upper_tri_values_np):
        feature_dict[str(j)] = value
        
    entries.append(feature_dict)

# ---------------- Save to CSV ----------------
# Create DataFrame from the list of feature dictionaries.
df_features = pd.DataFrame(entries)

# Define the full output CSV path.
csv_output_path = os.path.join(path_to_features, output_csv_filename)
df_features.to_csv(csv_output_path, index=False)
print(f"Feature CSV saved to {csv_output_path}.")
