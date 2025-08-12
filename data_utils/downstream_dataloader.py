import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class FCDownstreamDataset(Dataset):
    def __init__(self, df, targets, perm=None):
        """
        Args:
            df (pd.DataFrame): DataFrame with 'id', 'path', and target columns.
            targets (list of str): Names of columns with target variables.
            perm (Tensor, optional): Permutation tensor for FC matrix.
        """
        self.df = df
        self.targets = targets
        self.perm = perm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fc = torch.load(row['path'])  # Load FC matrix

        if self.perm is not None:
            fc = fc[self.perm, :][:, self.perm]

        target_values = torch.tensor([row[t] for t in self.targets], dtype=torch.float32)
        return fc, target_values

def get_downstream_loaders(args):
    """
    For a single dataset, return train/eval DataLoaders for downstream prediction tasks.
    FC matrices and target values (from args.targets) are loaded per sample.

    Assumes:
        - Dataset-specific config JSON in dataset_configs/{dataset}_config.json
        - {dataset}_dataset.csv with FC paths and 'id' column.
        - {dataset}_variables.csv with 'src_subject_id' and target columns.
    """
    ds = args.dataset.strip()
    config_path = os.path.join("dataset_configs", f"{ds}_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = config["data_path"]
    csv_file = config.get("csv_file", os.path.join(data_path, f"{ds}_dataset.csv"))
    vars_file = os.path.join(data_path, f"{ds}_variables.csv")

    # Load dataset CSV and variables CSV
    df_fc = pd.read_csv(csv_file)
    df_vars = pd.read_csv(vars_file, usecols=["src_subject_id"] + args.targets)

    # Merge on subject id
    merged_df = df_fc.merge(df_vars, left_on="id", right_on="src_subject_id")
    print(f"[{ds}] Merged data has {len(merged_df)} samples after joining FC and targets.")

    # Optional permutation of FC matrix
    perm = None
    if args.permutation:
        gen = torch.Generator()
        gen.manual_seed(args.permutation)
        perm = torch.randperm(args.input_dim, generator=gen)
        print("*** Permutation of FC matrices is set.")

    # Train/test split
    train_df, eval_df = train_test_split(
        merged_df, test_size=args.val_size, random_state=args.seed
    )
    print(f"[{ds}] Train size: {len(train_df)} | Eval size: {len(eval_df)}")

    # Create datasets
    train_dataset = FCDownstreamDataset(train_df, args.targets, perm=perm)
    eval_dataset = FCDownstreamDataset(eval_df, args.targets, perm=perm)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1
    )

    return train_loader, eval_loader
