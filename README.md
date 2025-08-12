# NERVE: Network-Aware Representations of Brain Functional Connectivity via Masked AutoEncoders

## Description

### **Model Overview**
![Model Overview](figures/overview.png)

Predicting behavioral traits from resting-state functional MRI (rs-fMRI) remains a central challenge in neuroscience due to the complex, high-dimensional, and noisy nature of brain connectivity data. **NERVE** is a novel self-supervised learning framework tailored for brain connectivity data, adapting the masked autoencoder paradigm to high-resolution functional connectivity (FC) matrices with neurobiologically informed patching based on network parcellations.  
Instead of uniform spatial masking, NERVE reconstructs connectivity between functional networks, encouraging biologically meaningful representations. The model supports multiple patch embedding strategies and incorporates a teacher–student distillation scheme to improve semantic consistency.  
We evaluate NERVE on large-scale developmental rs-fMRI datasets (ABCD, PNC, CCNP), showing superior behavioral prediction accuracy compared to classical methods and recent SSL baselines, while maintaining strong generalization and interpretability.

### **Patch Embedding Strategies**
Our embedding strategies can be found in [`models/patch_embeddings.py`](models/patch_embeddings.py).  
![Patch Embeddings](figures/patch_embeddings.ong)

## Code Repository Usage

We use **Weights & Biases** ([wandb](https://wandb.ai/)) to log and track all experiments.

### **Training**
Use `slurm_train.py` to submit jobs that run `train.py` with different configurations:
```bash
python slurm_train.py --config configs/your_config.json
```

### **Inference / Feature Extraction**

Use `slurm_inference.py` to generate features with `inference.py` for a specific dataset and a specific trained model (identified by its wandb run ID):
```bash
python slurm_inference.py --dataset <DATASET_NAME> --model_id <wandb_run_id>
```

## **Data / Use Your Own Data**

To use your own dataset with NERVE, follow these steps using the scripts in the `preprocessing_utils` folder:

1. **Save Functional Connectivity (FC) Matrices**  
    Save each subject's FC matrix to a `.pt` tensor file.  
    Example:
    ```bash
    python preprocessing_utils/preprocess_abcd_hr_data.py --input_path <raw_data_path> --output_path <processed_data_path>
    ```

2.	**Generate Dataset Summary CSV**
    Create a CSV file listing subject IDs and paths to their corresponding .pt files.
    Example:
    ```bash
    python preprocessing_utils/make_abcd_dataset_paths_file.py --data_path <processed_data_path> --output_csv <dataset_csv_path>
    ```

3.	Update Dataset Configuration
    Edit or create the dataset configuration JSON file in dataset_configs/{dataset}_config.json.
    Example:
    ```bash
    nano dataset_configs/ABCD_config.json
    ```

Make sure to set the correct paths and parameters for your dataset.

## **Citation**

If you use this code or refer to our article, please cite:

TODO: Add final citation text here.

BibTeX:


