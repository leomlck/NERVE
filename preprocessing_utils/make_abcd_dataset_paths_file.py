import os
import re
import pandas as pd

# Paths and filenames
data_path = 'data/abcd-data-hr/'
save_folder = 'preprocessed'
save_filename = 'rs_fc.pt'
csv_filename = 'ABCD_dataset.csv'  # CSV file to create

# Regex to filter subject folders (e.g., folders starting with "NDAR_")
regexp = re.compile(r'NDAR_')

# List all subject folders
subjects = [f for f in os.listdir(data_path) if regexp.search(f)]
subjects.sort()

# List to hold CSV entries
entries = []

# Process each subject
for i, subject_id in enumerate(subjects):
    print(f'Processing {i+1}/{len(subjects)}: {subject_id}')

    # Construct the full path to the .pt file for this subject
    pt_path = os.path.join(data_path, subject_id, save_folder, save_filename)

    # Check if the file exists. Skip if not found.
    if not os.path.exists(pt_path):
        print(f"Warning: {pt_path} does not exist. Skipping {subject_id}.")
        continue

    # Construct the path to the prompt embedding
    prompt_path = os.path.join(data_path, "prompts", subject_id, "text_embed.npy")

    # Append subject id, file path, and prompt path to the entries list
    entries.append({'id': subject_id, 'path': pt_path, 'prompt': prompt_path})

# Create a DataFrame from the entries list
df = pd.DataFrame(entries)

# Save the DataFrame as a CSV file in the data_path
csv_path = os.path.join(data_path, csv_filename)
df.to_csv(csv_path, index=False)

print(f"CSV file created: {csv_path}")
