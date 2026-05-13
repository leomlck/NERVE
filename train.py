# Adapted from github.com/jeonsworld/ViT-pytorch/blob/main/train.py

import argparse
import logging
import os
import time

import wandb
from tqdm import tqdm

import torch

from models.mae import MaskedAutoencoder
from data_utils.dataloader import dataset_setup, get_multi_loaders
from data_utils.network_utils import fetch_network_list
from train_utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from train_utils.metrics import AverageMeter
from train_utils.checkpoints import save_ckp, load_ckp
from train_utils.misc import count_parameters, set_seed

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train NERVE masked autoencoder on functional connectivity matrices")

    # Data and file handling
    parser.add_argument('--return_subject_id', type=int, default=0,
                        help="Return subject IDs in dataloader (1=True, 0=False)")
    parser.add_argument('--dataset', type=str, default="ABCD", help="Dataset name, dash-separated for multi-dataset training")
    parser.add_argument('--permutation', type=int, default=0, help="Seed for ROI permutation baseline (0 disables permutation)")

    # Output and experiment management
    parser.add_argument('--output_dir', type=str, default='outputs/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_id', type=str, default='test', help='Short run ID for logging/checkpoints')
    parser.add_argument('--description', type=str, default='nerve', help='Run description for logging')
    parser.add_argument('--resume', type=int, default=0, help="Resume training from checkpoint (1=True, 0=False)")

    # Optional split / initialization setup
    parser.add_argument('--k_fold', type=int, default=-1, help="Use precomputed fold index if available; -1 uses random validation split")
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                        help='Optional checkpoint path used to initialize model weights')

    # Model hyperparameters
    parser.add_argument('--embedding_type', type=str, default='outer', choices=['linear', 'linear-shared',
                                                                                 'MLP', 'MLP-shared',
                                                                                 'outer', 'outer-MLP',
                                                                                 'concat-MLP',
                                                                                 'add', 'add-MLP',
                                                                                 'GCN', 'GCN-shared'],
                        help="Type of patch embedding strategy to use")
    parser.add_argument('--decoding_type', type=str, default='linear', choices=['linear', 'linear-shared', 'outer'],
                        help="Type of patch decoding strategy to use")
    parser.add_argument('--network_method', type=str, default='p1', choices=['p1', 'p2', 'p3', 'vanilla'],
                        help="Method to define network partitions")
    parser.add_argument('--input_dim', type=int, default=400, help="Input size (number of brain regions)")
    parser.add_argument('--n_layers_enco', type=int, default=4, help="Number of transformer layers in encoder")
    parser.add_argument('--nhead_enco', type=int, default=4, help="Number of attention heads in encoder")
    parser.add_argument('--d_model_enco', type=int, default=256, help="Dimension of transformer embeddings in encoder")
    parser.add_argument('--n_layers_deco', type=int, default=1, help="Number of transformer layers in decoder")
    parser.add_argument('--nhead_deco', type=int, default=2, help="Number of attention heads in decoder")
    parser.add_argument('--d_model_deco', type=int, default=64, help="Dimension of transformer embeddings in decoder")
    parser.add_argument('--dropout_tsf', type=float, default=0.1, help="Dropout rate for transformers")
    parser.add_argument('--mask_ratio', type=float, default=0.5, help="Masking ratio for MAE")
    parser.add_argument('--norm_pix_loss', type=int, default=0, help="Normalize pixel loss (1=True, 0=False)")

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training")
    parser.add_argument('--eval_every', type=int, default=1, help="Evaluate model every X epochs")
    parser.add_argument('--val_size', type=float, default=0.1, help="Fraction of data used for validation")

    # Optimization and learning rate
    parser.add_argument('--optimizer', choices=['Adam', 'AdamW', 'SGD'], default='AdamW', help="Optimizer selection")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-2, help="Weight decay for regularization")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of total training epochs")
    parser.add_argument('--decay_type', choices=["cosine", "linear"], default="cosine", help="Learning rate decay strategy")
    parser.add_argument('--warmup_epochs', type=int, default=10, help="Number of warmup epochs for learning rate schedule")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Max gradient norm for clipping")

    # Logging, mixed precision, and hardware
    parser.add_argument('--wandb_project', type=str, default='nerve', help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Optional Weights & Biases entity')
    parser.add_argument('--use_amp', type=int, default=1, help="Use automatic mixed precision (1=True, 0=False)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def build_model(args):
    network_list = fetch_network_list(args.data_path, method=args.network_method)

    model = MaskedAutoencoder(
        input_dim=args.input_dim,
        embedding_type=args.embedding_type,
        decoding_type=args.decoding_type,
        network_list=network_list,
        embed_dim=args.d_model_enco,
        depth=args.n_layers_enco,
        num_heads=args.nhead_enco,
        decoder_embed_dim=args.d_model_deco,
        decoder_depth=args.n_layers_deco,
        decoder_num_heads=args.nhead_deco,
        norm_pix_loss=args.norm_pix_loss,
    )
    model.to(args.device)
    num_params = count_parameters(model)

    if args.pretrained_checkpoint is not None:
        print(f"*** Loading pretrained checkpoint: {args.pretrained_checkpoint} ***")
        checkpoint = torch.load(args.pretrained_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def _get_inputs(batch, device):
    """Handle dataloaders that return either inputs or (inputs, subject_id)."""
    if isinstance(batch, (tuple, list)):
        batch = batch[0]
    return batch.float().to(device)


def validate(args, model, eval_loader, wandb_step, global_step, epoch_step):
    eval_losses = AverageMeter()

    logger.info("\n\n***** Running Validation *****")
    logger.info("  Num steps = %d", len(eval_loader))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    epoch_iterator = tqdm(
        eval_loader,
        desc="Validating... (loss=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=False,
    )

    with torch.no_grad():
        for batch in epoch_iterator:
            input_rfmr = _get_inputs(batch, args.device)

            with torch.amp.autocast(args.device, enabled=args.use_amp):
                loss, _, _ = model(input_rfmr, mask_ratio=args.mask_ratio)

            eval_losses.update(loss.item())
            epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)

    wandb.log({
        'validation/loss': eval_losses.avg,
        'validation/mae_loss': eval_losses.avg,
        'global_step': global_step,
        'epoch_step': epoch_step,
    }, step=wandb.run.step + wandb_step + 1)

    return eval_losses.avg


def train_mae(args, model):
    """Train the masked autoencoder."""
    train_loader, eval_loader = get_multi_loaders(args)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6,
                                     weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    args.num_steps = args.num_epochs * len(train_loader)
    args.warmup_steps = args.warmup_epochs * len(train_loader)
    t_total = args.num_steps

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    logger.info("***** Running training *****")
    logger.info("  Total epochs = %d", args.num_epochs)
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Train batch size = %d", args.batch_size)

    model.zero_grad()
    set_seed(args)
    losses = AverageMeter()
    wandb_step, global_step, epoch_step, best_loss = 0, 0, 0, 1e12

    if args.resume:
        model, optimizer, scheduler, wandb_step, global_step, epoch_step, best_loss = load_ckp(
            args, model, optimizer, scheduler
        )
        model.to(args.device)

    while True:
        t = time.time()
        epoch_step += 1

        model.train()
        epoch_iterator = tqdm(
            train_loader,
            desc="Training (X / X Steps) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=False,
        )

        for batch in epoch_iterator:
            input_rfmr = _get_inputs(batch, args.device)

            with torch.amp.autocast(args.device, enabled=args.use_amp):
                loss, _, _ = model(input_rfmr, mask_ratio=args.mask_ratio)

            scaler.scale(loss).backward()
            losses.update(loss.item())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
            )

            # Save model checkpoint every 2 hours if epochs are long.
            if (time.time() - t) / 60 > 120:
                ckp = {
                    'wandb_step': wandb.run.step,
                    'global_step': global_step,
                    'epoch_step': epoch_step,
                    'best_loss': best_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                save_ckp(args, ckp, is_best=False)
                t = time.time()

            if global_step % t_total == 0:
                break

        if epoch_step % args.eval_every == 0:
            eval_loss = validate(args, model, eval_loader, wandb_step, global_step, epoch_step)
            if best_loss >= eval_loss:
                ckp = {'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict()}
                save_ckp(args, ckp, is_best=True)
                best_loss = eval_loss
            model.train()

        wandb.log({
            'train/epoch_loss': losses.avg,
            'train/mae_loss': losses.avg,
            'global_step': global_step,
            'epoch_step': epoch_step,
        }, step=wandb.run.step + wandb_step + 1)
        wandb.log({'train/lr': scheduler.get_last_lr()[0], 'global_step': global_step, 'epoch_step': epoch_step},
                  step=wandb.run.step + wandb_step + 1)

        losses.reset()
        ckp = {
            'wandb_step': wandb.run.step,
            'global_step': global_step,
            'epoch_step': epoch_step,
            'best_loss': best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        save_ckp(args, ckp, is_best=False)

        if global_step % t_total == 0:
            break

    logger.info("Best Eval Loss: \t%f" % best_loss)
    logger.info("End Training!")


def main():
    args = parse_args()

    if args.wandb_id == 'test':
        args.wandb_id = wandb.util.generate_id()

    os.makedirs(os.path.join(args.output_dir, args.wandb_id), exist_ok=True)
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.description, id=args.wandb_id, resume='allow')
    wandb.config.update(args)

    args.use_amp = bool(args.use_amp)
    args.norm_pix_loss = bool(args.norm_pix_loss)

    if torch.cuda.is_available():
        args.device = "cuda"
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device = "cpu"
        args.n_gpu = 0
    print('Using device :', args.device)

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger.warning("Devices: %s, n_gpu: %s" % (args.device, args.n_gpu))

    set_seed(args)
    args = dataset_setup(args)
    args, model = build_model(args)
    train_mae(args, model)


if __name__ == "__main__":
    main()
