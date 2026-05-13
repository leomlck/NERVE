# NERVE: Network-Aware Representations of Brain Functional Connectivity via Bilinear Tokenization

NERVE is a self-supervised learning framework for resting-state functional connectivity (FC) representation learning. It adapts masked autoencoding to FC matrices by redefining the tokenization step: FC matrices are partitioned into intra- and inter-network connectivity blocks, and each block is embedded as a token using a structured bilinear, network-aware factorization. This preserves large-scale network identity while keeping the parameterization efficient. In the MICCAI 2026 paper, NERVE is evaluated across developmental cohorts for behavior and psychopathology prediction, showing more stable and transferable representations than structurally agnostic MAE variants and graph-based self-supervised baselines.

![Overview of NERVE](figures/fig_overview.png)

**Overview of NERVE.** A. The functional connectivity (FC) matrix is partitioned into patches defined by pairs of functional brain networks. B. **Network-aware Bilinear Tokenization.** Each functional network is assigned learnable network-specific weights at initialization, and patch tokens are computed through structured bilinear interactions between network weights during forward. C. **MAE Framework.** We apply a standard MAE framework to the proposed network-aware tokens, thereby introducing a functionally informed inductive bias over connectivity structure.

---

## Repository structure

```text
.
├── train.py                         # train NERVE / MAE models
├── extract_features.py              # extract encoder features from a trained checkpoint
├── submit_train_slurm.py            # example SLURM launcher for training
├── submit_extract_features_slurm.py # example SLURM launcher for feature extraction
├── dataset_configs/                 # dataset-specific JSON configs
├── data_utils/                      # dataloaders and network/parcellation utilities
├── models/                          # MAE, patch tokenization, patch decoding modules
├── train_utils/                     # checkpointing, metrics, schedulers, misc helpers
└── preprocessing_utils/             # dataset-specific preprocessing helpers/examples
```

---

## Installation

Create a Python environment and install the required packages. A minimal example is:

```bash
conda create -n nerve python=3.9
conda activate nerve
pip install torch torchvision torchaudio
pip install timm einops pandas numpy scipy scikit-learn tqdm wandb
```

---

## Data format

NERVE expects precomputed FC matrices saved as PyTorch tensors (`.pt`). Each tensor should contain one subject-level FC matrix of shape:

```text
R x R
```

where `R` is the number of brain regions, e.g. `400` for Schaefer-400. The matrix should be ordered consistently with the atlas label file used to define functional networks.

Each dataset should have a dataset CSV file with at least two columns:

```csv
id,path
sub001,data/ABCD/fc/sub001.pt
sub002,data/ABCD/fc/sub002.pt
```

The column `id` is the subject identifier. The column `path` is the path to the `.pt` FC matrix.

---

## Dataset configuration files

Each dataset has a JSON configuration file under `dataset_configs/`. For example:

```json
{
  "data_path": "data/ABCD",
  "csv_file": "data/ABCD/ABCD_dataset.csv",
  "id_col": "src_subject_id",
  "cv_splits": "data/ABCD/ABCD_k10_splits.csv"
}
```

For training and feature extraction, the most important entries are:

- `data_path`: dataset folder. This folder should contain the atlas/network label file used by `fetch_network_list`, by default `struct_names.mat`.
- `csv_file`: CSV listing subject IDs and FC matrix paths.
- `cv_splits`: optional CSV containing precomputed fold assignments.

---

## Network definitions and tokenization schemes

Network partitions are defined in:

```text
data_utils/network_utils.py
```

The `--network_method` argument controls how FC patches are defined:

- `p1`: Schaefer network labels at the large-scale network level, e.g. 17 networks in the paper.
- `p2`: finer network labels using additional label components.
- `p3`: coarser grouping.
- `vanilla`: image-style contiguous `16 x 16` matrix blocks for the structurally agnostic MAE baseline.

Patch tokenization modules are implemented in:

```text
models/patch_embeddings.py
```

Main options include:

- `outer`: proposed bilinear network-aware tokenization.
- `linear`: patch-specific linear tokenization.
- `linear-shared`: shared linear tokenization with padding to a common patch size.
- Additional experimental variants are also implemented, including MLP, additive, concatenation, and GCN-based tokenizers.

Patch decoding modules are implemented in:

```text
models/patch_decoding.py
```

Main decoding options include:

- `outer`: bilinear network-aware decoding.
- `linear`: patch-specific linear decoding.
- `linear-shared`: shared linear decoding.

In the main NERVE configuration, the proposed bilinear formulation is selected with:

```bash
--embedding_type outer --decoding_type outer --network_method p1
```

---

## Training

To train NERVE on one dataset:

```bash
python train.py \
  --dataset ABCD \
  --wandb_id test_run \
  --description nerve_abcd_outer \
  --output_dir outputs/checkpoints \
  --wandb_project nerve \
  --input_dim 400 \
  --embedding_type outer \
  --decoding_type outer \
  --network_method p1 \
  --n_layers_enco 4 \
  --nhead_enco 4 \
  --d_model_enco 256 \
  --n_layers_deco 1 \
  --nhead_deco 2 \
  --d_model_deco 64 \
  --mask_ratio 0.5 \
  --batch_size 1024 \
  --optimizer AdamW \
  --learning_rate 1e-2 \
  --weight_decay 1e-2 \
  --num_epochs 4000 \
  --warmup_epochs 400 \
  --use_amp 1
```

To train on multiple datasets, join dataset names with `-`:

```bash
python train.py --dataset ABCD-RBC ...
```

This loads and concatenates the datasets listed in `dataset_configs/ABCD_config.json` and `dataset_configs/RBC_config.json`.

---

## Training with SLURM

An example SLURM launcher is provided:

```bash
python submit_train_slurm.py
```

Before using it, edit the cluster-specific placeholders in `submit_train_slurm.py`:

```bash
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
```

and update the environment setup section, for example:

```bash
# module load anaconda3
# module load cuda/11.8.0
# conda activate nerve
```

---

## Feature extraction

After training, extract encoder representations with:

```bash
python extract_features.py \
  --dataset ABCD \
  --wandb_id <wandb_id> \
  --pooling avg \
  --output_dir outputs/checkpoints \
  --features_dir outputs/features \
  --wandb_project nerve
```

Pooling options:

- `avg`: average over patch tokens, excluding CLS.
- `cls`: use the CLS token.
- `avg_wcls`: average over all tokens including CLS.
- `all`: save one feature CSV per token.

The output CSV contains one row per subject and one column per feature dimension:

```csv
src_subject_id,0,1,2,...
sub001,...
sub002,...
```

---

## Citation

If you use this code, please cite: TBD

```bibtex
```
