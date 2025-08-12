# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

#from timm.models.vision_transformer import PatchEmbed, Block
from models.vision_transformer_w_attn import PatchEmbed, Block

import torch_geometric.utils as tgu
import torch_geometric.nn as tgnn

from models.patch_embeddings import (
    PatchLinearEmbedding,
    PatchLinearSharedEmbedding,
    MLPEmbedding,
    MLPSharedEmbedding,
    OuterProductEmbedding,
    OuterMLPEmbedding,
    ConcatFusionMLPWithPatchMean,
    AdditionFusionWithPatchMean,
    AdditionFusionMLPWithPatchMean,
    SharedGCNEmbedding,
    GCNEmbedding
)
from models.patch_decoding import (
        PatchLinearDecoder,
        PatchLinearSharedDecoder,
        OuterProductDecoder
)

class MaskedAutoencoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, input_dim=400, embedding_type='linear', decoding_type='linear-shared', network_list=None,
                 embed_dim=128, depth=4, num_heads=2,
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=2,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        # Compute max patch size
        self.network_list = network_list
        self.max_patch_size = max(len(idx_a) * len(idx_b) for i, (net_a, idx_a) in enumerate(network_list.items())
                                  for j, (net_b, idx_b) in enumerate(network_list.items()) if i <= j)
        # Compute number of patches dynamically
        num_networks = len(network_list)
        self.num_patches = (num_networks * (num_networks + 1)) // 2  # Includes self-pairs

        # Select embedding layer
        if embedding_type == "linear":
            self.patch_embed = PatchLinearEmbedding(network_list, embed_dim)
        elif embedding_type == "linear-shared":
            self.patch_embed = PatchLinearSharedEmbedding(network_list, embed_dim)
        elif embedding_type == "MLP":
            self.patch_embed = MLPEmbedding(network_list, embed_dim)
        elif embedding_type == "MLP-shared":
            self.patch_embed = MLPSharedEmbedding(network_list, embed_dim)
        elif embedding_type == "outer":
            self.patch_embed = OuterProductEmbedding(network_list, embed_dim)
        elif embedding_type == "outer-MLP":
            self.patch_embed = OuterMLPEmbedding(network_list, embed_dim)
        elif embedding_type == "concat-MLP":
            self.patch_embed = ConcatFusionMLPWithPatchMean(network_list, embed_dim)
        elif embedding_type == "add":
            self.patch_embed = AdditionFusionWithPatchMean(network_list, embed_dim)
        elif embedding_type == "add-MLP":
            self.patch_embed = AdditionFusionMLPWithPatchMean(network_list, embed_dim)
        elif embedding_type == "GCN-shared":
            self.patch_embed = SharedGCNEmbedding(network_list, embed_dim)
        elif embedding_type == "GCN":
            self.patch_embed = GCNEmbedding(network_list, embed_dim)
        else:
            raise ValueError(f"Invalid embedding type: {embedding_type}")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim)) #, requires_grad=False)  # fixed sin-cos embedding        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim)) #, requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=False, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        if decoding_type == "linear":
            self.decoder_pred = PatchLinearDecoder(network_list, decoder_embed_dim) 
        elif decoding_type == "linear-shared":
            self.decoder_pred = PatchLinearSharedDecoder(decoder_embed_dim, self.max_patch_size)
        elif decoding_type == "outer":
            self.decoder_pred = OuterProductDecoder(network_list, decoder_embed_dim)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        
        #pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2]) #, cls_token=True)
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.pos_embed)
        #decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.decoder_pos_embed.shape[-2]) #, cls_token=True)
        #self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        nn.init.xavier_uniform_(self.decoder_pos_embed)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, return_all_attn=False):
        """
        Forward pass through the encoder with optional attention map collection.

        Args:
            x (Tensor): FC matrices (B, N, N)
            mask_ratio (float): Proportion of patches to mask.
            return_all_attn (bool): If True, collects attention maps from all layers.

        Returns:
            x (Tensor): Final encoded representation
            mask (Tensor): Mask applied during patch dropout
            ids_restore (Tensor): Indices to restore original patch order
            all_attn (List[Tensor]): List of attention maps from each block (optional)
        """
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        all_attn = []

        # apply Transformer blocks
        for blk in self.blocks:
            if return_all_attn:
                x, attn = blk(x, return_attn=True)
                all_attn.append(attn)
            else:
                x = blk(x)

        x = self.norm(x)

        if return_all_attn:
            return x, mask, ids_restore, all_attn
        else:
            return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        #x = x[:, 1:, :]

        return x

    def patchify(self, fc_matrices):
        """
        Extracts patches from FC matrices based on `network_list`.

        Args:
            fc_matrices (Tensor): Functional connectivity matrices (B, N, N).

        Returns:
            Tensor: Flattened patches of shape (B, num_patches, max_patch_size).
        """
        B = fc_matrices.shape[0]
        patches = []

        for i, (net_a, idx_a) in enumerate(self.network_list.items()):
            for j, (net_b, idx_b) in enumerate(self.network_list.items()):
                if i > j:  # Skip redundant B-A pairs
                    continue

                patch = fc_matrices[:, idx_a][:, :, idx_b]  # Extract A-B patch
                patch_flat = patch.reshape(B, -1)  # Flatten patch

                # Pad to max_patch_size
                pad_size = self.max_patch_size - patch_flat.shape[1]
                padded_patch = F.pad(patch_flat, (0, pad_size))  # Pad at the end

                patches.append(padded_patch)

        return torch.stack(patches, dim=1)  # Shape: (B, num_patches, max_patch_size)

    def forward_loss(self, fc_matrices, pred_patches, mask):
        """
        Computes the loss only on the masked patches.

        Args:
            fc_matrices (Tensor): Ground-truth FC matrices (B, N, N).
            pred_tokens (Tensor): Decoded patches from `PatchDecoder`.
            mask (Tensor): Mask indicating which patches were removed (B, num_patches).

        Returns:
            Tensor: Masked MSE loss.
        """
        target_patches = self.patchify(fc_matrices)  # Extract target patches

        if self.norm_pix_loss:
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1.e-6) ** 0.5

        loss = (pred_patches - target_patches) ** 2
        loss = loss.mean(dim=-1)  # Compute per-patch loss (B, num_patches)

        loss = (loss * mask).sum() / mask.sum()  # Compute loss on masked patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # (B, N, N)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


