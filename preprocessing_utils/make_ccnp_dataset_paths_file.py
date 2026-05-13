import os
import re
import pandas as pd

# Define the directory containing the .pt files
s = 100
ext = '_Kong22'
data_path = f"data/ccnp-data-hr-{s}"+ext
csv_filename = 'CCNP_dataset.csv'

# Regex pattern to match files like "sub-colornest001_fc.pt" and extract the subject ID
pattern = re.compile(r"sub-(colornest\d+)_fc\.pt$")

# List to hold CSV entries
entries = []

# List all files in the directory
for fname in os.listdir(data_path):
    if fname.endswith('.pt'):
        match = pattern.search(fname)
        if match:
            subject_id = match.group(1)
            file_path = os.path.join(data_path, fname)
            entries.append({'id': subject_id, 'path': file_path})
        else:
            print(f"File {fname} does not match the expected pattern. Skipping.")

# Create a DataFrame from the entries
df = pd.DataFrame(entries)

# Save the DataFrame as a CSV file in the same directory
csv_path = os.path.join(data_path, csv_filename)
df.to_csv(csv_path, index=False)

print(f"CSV file created: {csv_path}")
