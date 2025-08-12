import os
import torch
import numpy as np
from scipy.io import loadmat
import re

# Paths
data_path_400 = '/midtier/sablab/scratch/lem4012/data/abcd-data-hr/'
data_path_100 = '/midtier/sablab/scratch/lem4012/data/abcd-data-hr-100_Kong22/'
mapping_mat_path = '/midtier/sablab/scratch/lem4012/data/abcd-data-hr/100_K_400_K_mapping.mat'
struct_names_mat_path = '/midtier/sablab/scratch/lem4012/data/abcd-data-hr/struct_names.mat'

# Load struct_names (list of 400 ROI labels)
fc_lbls = loadmat(struct_names_mat_path)['struct_names']
fc_lbls = [s[0][0] for s in fc_lbls]

# Load 100->400 mapping
mapping = loadmat(mapping_mat_path)['table_names_after_mapping']  # shape (100, 2)
selected_400_labels = [entry[1][0] for entry in mapping]  # second column has 400 ROIs to select

# Convert selected 400 ROI names to indices in fc_lbls
selected_indices = [fc_lbls.index(label) for label in selected_400_labels]

# Regex to filter subject folders
regexp = re.compile(r'NDAR_')
subjects = [f for f in os.listdir(data_path_400) if regexp.search(f)]
subjects.sort()

print(f"Reducing {len(subjects)} subjects' FC matrices to 100x100...")

for i, subject_id in enumerate(subjects):
    print(f"[{i+1}/{len(subjects)}] Processing: {subject_id}")

    # Load 400x400 FC matrix
    mat_path = os.path.join(data_path_400, subject_id, 'rs_fc.mat')
    fc_400 = loadmat(mat_path)['fc_mat']

    # Extract 100x100 submatrix
    fc_100 = fc_400[np.ix_(selected_indices, selected_indices)]

    # Convert to PyTorch tensor
    fc_100_tensor = torch.tensor(fc_100, dtype=torch.float32)

    # Save reduced matrix in corresponding folder in new directory
    save_dir = os.path.join(data_path_100, subject_id, 'preprocessed')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'rs_fc.pt')
    torch.save(fc_100_tensor, save_path)

print("✅ All FC matrices saved in 100x100 format.")
