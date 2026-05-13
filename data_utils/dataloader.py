import json
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class FC_Dataset(Dataset):
    def __init__(self, data_info, return_subject_id=False, perm=None):
        """
        Dataset for precomputed FC matrices stored as PyTorch tensors.

        Args:
            data_info (str or pd.DataFrame): CSV path or DataFrame containing:
                - 'id': subject identifier
                - 'path': path to the preprocessed .pt file
            return_subject_id (bool): Whether to return the subject id with the matrix.
            perm (Tensor, optional): ROI permutation to apply to the FC matrix.
        """
        self.data_info = pd.read_csv(data_info) if isinstance(data_info, str) else data_info
        self.return_subject_id = return_subject_id
        self.perm = perm

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        subject_id = row['id']
        file_path = row['path']

        try:
            fc_matrix = torch.load(file_path)
        except FileNotFoundError as exc:
            raise ValueError(f"File not found: {file_path}") from exc

        if self.perm is not None:
            fc_matrix = fc_matrix[self.perm, :][:, self.perm]

        if self.return_subject_id:
            return fc_matrix, subject_id
        return fc_matrix


def dataset_setup(args):
    datasets = [ds.strip() for ds in args.dataset.split('-')]
    ds = datasets[0]
    config_path = os.path.join("./dataset_configs", f"{ds}_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    args.data_path = config["data_path"]
    return args


def _load_dataset_dataframe(ds):
    config_path = os.path.join("dataset_configs", f"{ds}_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = config["data_path"]
    csv_file = config.get("csv_file", os.path.join(data_path, f"{ds}_dataset.csv"))
    df = pd.read_csv(csv_file)

    if 'src_subject_id' in df.columns:
        df = df.rename(columns={'src_subject_id': 'id'})

    required_cols = {'id', 'path'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset CSV for {ds} is missing columns: {missing}")

    return df, config


def get_multi_loaders(args):
    dataset_names = [ds.strip() for ds in args.dataset.split('-')]
    train_datasets, eval_datasets = [], []

    for ds in dataset_names:
        df, config = _load_dataset_dataframe(ds)
        print(f"Dataset {ds}: {len(df)} subjects.")

        if args.k_fold >= 0:
            cv_file = config.get("cv_splits")
            if cv_file is None:
                raise ValueError(
                    f"k_fold={args.k_fold} requested, but no 'cv_splits' entry was found in {ds}_config.json"
                )
            folds_df = pd.read_csv(cv_file)
            if 'src_subject_id' in folds_df.columns:
                folds_df = folds_df.rename(columns={'src_subject_id': 'id'})
            assert 'fold' in folds_df.columns, "CV split file must contain a 'fold' column"
            df = df.merge(folds_df[['id', 'fold']], on='id', how='inner')
            train_df = df[df['fold'] != args.k_fold].copy()
            eval_df = df[df['fold'] == args.k_fold].copy()
            print(f"[CV MODE] Dataset {ds}: using fold {args.k_fold} from {cv_file}")
        elif args.val_size > 0:
            train_df, eval_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
        else:
            train_df = df.copy()
            eval_df = df.sample(frac=min(0.1, 1.0), random_state=args.seed).copy()

        if args.return_subject_id:
            train_df = df.copy()
            eval_df = df.copy()

        print(f"Dataset {ds}: Train size {len(train_df)}, Eval size {len(eval_df)}")

        perm = None
        if args.permutation:
            gen = torch.Generator()
            gen.manual_seed(args.permutation)
            perm = torch.randperm(args.input_dim, generator=gen)
            print("*** Permutation of FC matrices is set.")

        train_datasets.append(FC_Dataset(train_df, return_subject_id=args.return_subject_id, perm=perm))
        eval_datasets.append(FC_Dataset(eval_df, return_subject_id=args.return_subject_id, perm=perm))

    combined_train_dataset = ConcatDataset(train_datasets)
    combined_eval_dataset = ConcatDataset(eval_datasets)

    train_loader = DataLoader(combined_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    eval_loader = DataLoader(combined_eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    print("Total combined train samples:", len(combined_train_dataset))
    print("Total combined eval samples:", len(combined_eval_dataset))

    return train_loader, eval_loader
