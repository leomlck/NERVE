import os
import torch
from scipy.io import loadmat
import re

# Paths
data_path = '/midtier/sablab/scratch/lem4012/data/abcd-data-hr/'
save_folder = 'preprocessed'
save_filename = 'rs_fc.pt'

# Regex to filter subject folders
regexp = re.compile(r'NDAR_')

# List all subject folders
subjects = [f for f in os.listdir(data_path) if regexp.search(f)]
subjects.sort()

# Process each subject
for i, subject_id in enumerate(subjects):
    print(f'Processing {i+1}/{len(subjects)}: {subject_id}')

    # Load FC matrix from .mat file
    mat_path = os.path.join(data_path, subject_id, 'rs_fc.mat')
    rs_fc = loadmat(mat_path)['fc_mat']  # Extract FC matrix

    # Convert to PyTorch tensor
    rs_fc_tensor = torch.tensor(rs_fc, dtype=torch.float32)

    # Save as .pt file
    save_path = os.path.join(data_path, subject_id, save_folder)
    os.makedirs(save_path, exist_ok=True)
    torch.save(rs_fc_tensor, os.path.join(save_path, save_filename))

print("✅ All FC matrices saved in .pt format.")
