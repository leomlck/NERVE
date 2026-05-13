import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def dataset_setup(args):
    datasets = [ds.strip() for ds in args.dataset.split('-')]
    ds = datasets[0]
    config_path = os.path.join("./dataset_configs", f"{ds}_config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    args.data_path = config["data_path"]
    return args


def make_cv_splits(args):
    # -----------------------------
    # Load dataset config
    # -----------------------------
    datasets = [ds.strip() for ds in args.dataset.split('-')]
    ds = datasets[0]
    config_path = os.path.join("dataset_configs", f"{ds}_config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    data_path = config["data_path"]

    # -----------------------------
    # Load subject CSV
    # -----------------------------
    csv_file = config.get(
        "csv_file",
        os.path.join(data_path, f"{ds}_dataset.csv")
    )
    df = pd.read_csv(csv_file)

    # -----------------------------
    # Load variable CSV
    # -----------------------------
    var_file = config.get('variables_csv', os.path.join(data_path, f'{ds}_variables.csv'))
    vars_df = pd.read_csv(var_file)

    vars_df = vars_df[['src_subject_id', args.var]]
    vars_df = vars_df.rename(columns={
        'src_subject_id': 'id',
        args.var: 'y'
    })

    # -----------------------------
    # Merge + clean
    # -----------------------------
    df = df.merge(vars_df, on='id', how='inner')
    df = df[['id', 'y']].dropna().reset_index(drop=True)

    # -----------------------------
    # Stratification for continuous y
    # -----------------------------
    # Bin y into quantiles for stratified CV
    n_bins = min(5, len(df) // args.k)
    df['y_bin'] = pd.qcut(
        df['y'],
        q=n_bins,
        labels=False,
        duplicates='drop'
    )

    # -----------------------------
    # K-fold split
    # -----------------------------
    skf = StratifiedKFold(
        n_splits=args.k,
        shuffle=True,
        random_state=args.seed
    )

    df['fold'] = -1
    for fold, (_, test_idx) in enumerate(skf.split(df, df['y_bin'])):
        df.loc[test_idx, 'fold'] = fold

    assert (df['fold'] >= 0).all(), "Some subjects were not assigned a fold."

    # -----------------------------
    # Save split CSV
    # -----------------------------
    out_csv = os.path.join(
        data_path,
        f"{ds}_{args.var}_k{args.k}_splits.csv"
    )

    df[['id', 'y', 'fold']].to_csv(out_csv, index=False)

    # -----------------------------
    # Write path back to config
    # -----------------------------
    config['cv_splits'] = out_csv
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[OK] CV splits saved to: {out_csv}")
    print(f"[OK] Updated config: {config_path}")
    print(df.groupby('fold').size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--var", type=str, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args = dataset_setup(args)
    make_cv_splits(args)
