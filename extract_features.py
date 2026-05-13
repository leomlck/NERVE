import logging
import argparse
import os
import numpy as np
import pandas as pd
#from collections import OrderedDict
import wandb

from tqdm import tqdm 

import torch 
from models.mae import MaskedAutoencoder
from data_utils.dataloader import dataset_setup, get_multi_loaders
from data_utils.network_utils import fetch_network_list 
from train_utils.misc import count_parameters, set_seed

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract NERVE encoder features from functional connectivity matrices")
    
    # Specific inference args
    parser.add_argument("--pooling", type=str, default='avg', choices=['avg', 'cls', 'avg_wcls', 'all'],
                        help="Method for pooling features from the transformer encoder.")
    parser.add_argument('--wandb_id', type=str, required=True, help='W&B run id to load configuration/checkpoint')
    parser.add_argument('--dataset', type=str, default='ABCD', help='Dataset name used for feature extraction')
    parser.add_argument('--output_dir', type=str, default='outputs/checkpoints', help='Directory containing model checkpoints')
    parser.add_argument('--features_dir', type=str, default='outputs/features', help='Directory where extracted features are saved')
    parser.add_argument('--wandb_project', type=str, default='nerve', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Optional Weights & Biases entity')
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

def build_model(args):
    # Get network list
    network_list = fetch_network_list(args.data_path, method=args.network_method)

    # Prepare model
    model = MaskedAutoencoder(input_dim=args.input_dim, embedding_type=args.embedding_type, decoding_type=args.decoding_type, network_list=network_list,
                 embed_dim=args.d_model_enco, depth=args.n_layers_enco, num_heads=args.nhead_enco,
                 decoder_embed_dim=args.d_model_deco, decoder_depth=args.n_layers_deco, decoder_num_heads=args.nhead_deco,
                 norm_pix_loss=args.norm_pix_loss)
    model.to(args.device)

    # Load pretrained
    checkpoint_path = os.path.join(args.output_dir, args.wandb_id, f'{args.wandb_id}_best.bin')
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

    num_params = count_parameters(model)    
   
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model
 
def extract_features(args, model):
    """Extract encoder features for all subjects in the selected dataset."""
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

            # Collect subject IDs and features
            patient_id = convert_patient_id(patient_id)
            if isinstance(patient_id, (list, tuple)):
                all_patient_ids.extend(patient_id)
            else:
                all_patient_ids.append(patient_id)
            all_features.append(feats.detach().cpu().numpy())

    # End inference processing.
    if args.pooling == 'all':
        # Create an output folder that will hold one CSV per token.
        output_folder = os.path.join(args.features_dir, f'features_{args.description}_all')
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
        os.makedirs(args.features_dir, exist_ok=True)
        output_path = os.path.join(args.features_dir, f'features_{args.description}_{args.pooling}.csv')
        df.to_csv(output_path, index=False)
        logger.info(f"Saved pooled features CSV to {output_path}")

    logger.info("End Inference!")

def fetch_config(run_id, entity=None, project='nerve'):
    """
    Fetch the configuration (arguments) of a wandb run given its run_id.
    """
    api = wandb.Api()
    run_path = f'{entity}/{project}/{run_id}' if entity else f'{project}/{run_id}'
    run = api.run(run_path)
    return run.config

def main():
    # Required parameters
    args = parse_args()
    dataset = args.dataset
    config = fetch_config(args.wandb_id, entity=args.wandb_entity, project=args.wandb_project)
    config['return_subject_id'] = 1
    config['batch_size'] = 1
    config['num_epochs'] = 1
    config['mask_ratio'] = 0.
    config['description'] = 'nerve_{}_{}_{}_enc_{}_dec_nw_{}_{}'.format(config['dataset'],
                                                                      config['wandb_id'], 
                                                                   config['embedding_type'], 
                                                                   config['decoding_type'],
                                                                   config['network_method'],
                                                                      dataset)
    cli_output_dir = args.output_dir
    cli_features_dir = args.features_dir
    args.__dict__.update(config)
    args.dataset = dataset
    args.output_dir = cli_output_dir
    args.features_dir = cli_features_dir

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
    args, model = build_model(args)

    # Training
    extract_features(args, model)


if __name__ == "__main__":
    main()
