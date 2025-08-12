import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split

class FC_Dataset(Dataset):
    def __init__(self, data_info, return_subject_id=False, perm=None):
        """
        Args:
            data_info (str or pd.DataFrame): Path to a CSV file or a DataFrame containing:
                - 'id': unique subject identifier
                - 'path': full (or relative) path to the preprocessed .pt file
            return_subject_id (bool): Whether to return the subject id and settings with the data.
            perm (Tensor, optional): Permutation to apply to the FC matrix.
        """
        # If data_info is a string, assume it's a CSV file path and load it.
        if isinstance(data_info, str):
            self.data_info = pd.read_csv(data_info)
        else:
            self.data_info = data_info

        self.return_subject_id = return_subject_id
        self.perm = perm

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        subject_id = row['id']
        file_path = row['path']

        # Load the preprocessed .pt file
        try:
            rs_fmri = torch.load(file_path)
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")

        # Optionally apply a permutation (for example, to shuffle rows and columns)
        if self.perm is not None:
            rs_fmri = rs_fmri[self.perm, :][:, self.perm]

        if self.return_subject_id:
            return rs_fmri, subject_id
        return rs_fmri

def dataset_setup(args):
    # Build path to the JSON config file
    datasets = [ds.strip() for ds in args.dataset.split('-')]
    ds = datasets[0]
    config_path = os.path.join("./dataset_configs", f"{ds}_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # The data_path should be defined in the JSON file.
    args.data_path = config["data_path"]
    return args

def get_multi_loaders(args):
    """
    Creates train and evaluation DataLoaders by concatenating multiple datasets.

    This function assumes:
      - args.datasets is a comma-separated string (e.g., "ABCD,RBC")
      - For each dataset, a JSON configuration file exists in "dataset_configs" with the name
        "<dataset>_config.json". Each JSON must include at least:
            "data_path": base path for the dataset,
            "val_size": fraction for validation,
            "input_dim": FC matrix dimension,
         Optionally, the JSON can also provide "csv_file" (the file name or full path for the
         CSV with subject info). If not provided, a default is assumed (e.g., "<dataset>_dataset.csv" inside data_path).
      - The CSV for each dataset must have at least columns 'id' and 'path'.
      - You use your FC_Dataset to load the data.

    The function loads each dataset (splitting it into train and eval using train_test_split),
    then concatenates all train datasets and all eval datasets using ConcatDataset, and finally creates DataLoaders.
    """
    dataset_names = [ds.strip() for ds in args.dataset.split('-')]
    train_datasets = []
    eval_datasets = []

    for ds in dataset_names:
        # Load the dataset-specific JSON config.
        config_path = os.path.join("dataset_configs", f"{ds}_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Get the base data path.
        data_path = config["data_path"]
        # Determine the CSV file for subject information.
        csv_file = config.get("csv_file", os.path.join(data_path, f"{ds}_dataset.csv"))

        # Load the CSV.
        df = pd.read_csv(csv_file)
        print(f"Dataset {ds}: Found {len(df)} subjects.")

        # Split into train and validation sets.
        train_df, eval_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
        if args.return_subject_id:
            # Optionally, if desired, one might use the entire CSV.
            train_df = df
        print(f"Dataset {ds}: Train size {len(train_df)}, Eval size {len(eval_df)}.")

        # Create a permutation, if specified in config.
        perm = None
        if args.permutation:
            gen = torch.Generator()
            gen.manual_seed(args.permutation)
            perm = torch.randperm(args.input_dim, generator=gen)
            print("*** Permutation of FC matrices is set.")

        # Create dataset objects.
        train_dataset = FC_Dataset(train_df, return_subject_id=args.return_subject_id, perm=perm)
        eval_dataset = FC_Dataset(eval_df, return_subject_id=args.return_subject_id, perm=perm)

        train_datasets.append(train_dataset)
        eval_datasets.append(eval_dataset)

    # Concatenate all train and evaluation datasets.
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_eval_dataset = ConcatDataset(eval_datasets)

    # Create DataLoaders.
    train_loader = DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    eval_loader = DataLoader(combined_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print("Total combined train samples:", len(combined_train_dataset))
    print("Total combined eval samples:", len(combined_eval_dataset))

    return train_loader, eval_loader


