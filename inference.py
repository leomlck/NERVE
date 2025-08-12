# Adapted from github.com/jeonsworld/ViT-pytorch/blob/main/train.py

import logging
import argparse
import os
import random
import numpy as np
import time
import pandas as pd
#from collections import OrderedDict
import wandb

from tqdm import tqdm 

import torch 
import torch.nn as nn 
import torch.distributed as dist 
import torch.nn.functional as F 

from models.mae import MaskedAutoencoder
from data_utils.dataloader import dataset_setup, get_multi_loaders
from data_utils.network_utils import fetch_network_list 
from train_utils.misc import count_parameters, set_seed

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Inference script for MAE-based fMRI analysis")
    
    # Specific inference args
    parser.add_argument("--pooling", type=str, default='avg', choices=['avg', 'cls', 'avg_wcls', 'all'],
                        help="Method for pooling features from the transformer encoder.")
    parser.add_argument('--wandb_id', type=str, default='test', help="Short run ID for logging")

    parser.add_argument('--dataset', type=str, default="ABCD", help="Dataset name")
    return parser.parse_args()

def convert_patient_id(pid):
    # If it's a tensor:
    if torch.is_tensor(pid):
        # If it's a single-element tensor, get its value as a Python type.
        if pid.numel() == 1:
            return pid.item()
        # If it has multiple elements, convert to a list.
        else:
            return pid.tolist()
    # Otherwise, return as is.
    return pid

def model_setup(args):
    # Get network list
    network_list = fetch_network_list(args.data_path, method=args.network_method)

    # Prepare model
    model = MaskedAutoencoder(input_dim=args.input_dim, embedding_type=args.embedding_type, decoding_type=args.decoding_type, network_list=network_list,
                 embed_dim=args.d_model_enco, depth=args.n_layers_enco, num_heads=args.nhead_enco,
                 decoder_embed_dim=args.d_model_deco, decoder_depth=args.n_layers_deco, decoder_num_heads=args.nhead_deco,
                 norm_pix_loss=args.norm_pix_loss)
    model.to(args.device)

    # Load pretrained
    checkpoint = torch.load(os.path.join(args.output_dir, args.wandb_id, '{}_best.bin'.format(args.wandb_id)))
    model.load_state_dict(checkpoint['state_dict'])

    num_params = count_parameters(model)    
   
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model
 
def infer(args, model):
    """ Train the model """
    # Prepare dataset
    train_loader,_  = get_multi_loaders(args)
 
    # Train!
    logger.info("***** Running inference *****")
    
    set_seed(args)  
    model.eval()
    epoch_iterator = tqdm(train_loader,
                          desc="Inference (X / X Steps)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False) 
    all_features = []
    all_patient_ids = []

    for batch in epoch_iterator:
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            input_rfmr, patient_id = batch
            input_rfmr = input_rfmr.float().to(args.device)

            # Forward through model to get features from encoder.
            feats, _, _ = model.forward_encoder(input_rfmr, mask_ratio=args.mask_ratio)

            # Pooling strategies:
            if args.pooling == 'cls':
                feats = feats[:, 0, :]             # use the [CLS] token only
            elif args.pooling == 'avg_wcls':
                feats = feats.mean(dim=1)            # average over all tokens (including CLS)
            elif args.pooling == 'avg':
                feats = feats[:, 1:, :].mean(dim=1)   # average over non-CLS tokens
            elif args.pooling == 'all':
                # No pooling: keep full embeddings with shape (batch, num_tokens, embed_dim)
                pass
            else:
                raise ValueError(f"Unknown pooling option: {args.pooling}")

            # Collect patient IDs and features
            patient_id = convert_patient_id(patient_id)
            all_patient_ids.append(patient_id)  # Assuming patient_id is iterable; otherwise, use append()
            all_features.append(feats.detach().cpu().numpy())

    # End inference processing.
    if args.pooling == 'all':
        # Create an output folder that will hold one CSV per token.
        output_folder = f"/midtier/sablab/scratch/lem4012/save/fc_mae_features/features_{args.description}_all"
        os.makedirs(output_folder, exist_ok=True)

        # Concatenate features along the first axis (samples). The result shape is (num_subjects, num_tokens, embed_dim)
        all_features_arr = np.concatenate(all_features, axis=0)
        num_subjects, num_tokens, embed_dim = all_features_arr.shape
        logger.info(f"Total subjects: {num_subjects}, Tokens per subject: {num_tokens}, Embedding dimension: {embed_dim}")

        # For each token (i.e., each token index), create a CSV file.
        for token_idx in range(num_tokens):
            # Extract the features for the given token across all subjects: shape (num_subjects, embed_dim)
            token_features = all_features_arr[:, token_idx, :]
            # Create a DataFrame with subject IDs and token features.
            df_token = pd.DataFrame(token_features, columns=[str(i) for i in range(embed_dim)])
            df_token.insert(0, "src_subject_id", all_patient_ids)
            # Save the DataFrame to CSV, e.g., token_0.csv, token_1.csv, etc.
            out_csv = os.path.join(output_folder, f"token_{token_idx}.csv")
            df_token.to_csv(out_csv, index=False)
            logger.info(f"Saved token {token_idx} features to {out_csv}")
    else:
        # For the other pooling options, the features array is 2D: (num_subjects, embed_dim)
        features = np.concatenate(all_features, axis=0)
        df_feats = pd.DataFrame(features)
        df_patients = pd.DataFrame(all_patient_ids, columns=["src_subject_id"])
        df = pd.concat([df_patients, df_feats], axis=1)
        output_path = f"/midtier/sablab/scratch/lem4012/save/fc_mae_features/features_{args.description}_{args.pooling}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved pooled features CSV to {output_path}")

    logger.info("End Inference!")

def fetch_config(run_id, entity='leomlck', project='fc_mae'):
    """
    Fetch the configuration (arguments) of a wandb run given its run_id.
    """
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    return run.config

def main():
    # Required parameters
    args = get_args()
    dataset = args.dataset
    config = fetch_config(args.wandb_id)
    config['return_subject_id'] = 1
    config['batch_size'] = 1
    config['num_epochs'] = 1
    config['mask_ratio'] = 0.
    config['description'] = 'fc_mae_{}_{}_{}_enc_{}_dec_nw_{}_{}'.format(config['dataset'],
                                                                      config['wandb_id'], 
                                                                   config['embedding_type'], 
                                                                   config['decoding_type'],
                                                                   config['network_method'],
                                                                      dataset)
    args.__dict__.update(config)
    args.dataset = dataset

    # Setup CUDA, GPU & distributed training
    args.use_amp = bool(args.use_amp)
    if torch.cuda.is_available():
        device = "cuda"
        args.n_gpu = torch.cuda.device_count()
        args.device = device
    else:
        args.device = "cpu"
        args.n_gpu = 0 
    print('Using device :', args.device)


    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning("Devices: %s, n_gpu: %s" %(args.device, args.n_gpu))

    # Set seed
    set_seed(args)

    # Dataset Setup
    args = dataset_setup(args)

    # Model & Tokenizer Setup
    args, model = model_setup(args)

    # Training
    infer(args, model)


if __name__ == "__main__":
    main()
