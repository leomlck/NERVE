# Adapted from https://github.com/HobbitLong/SupContrast/blob/master/losses.py

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKSoftSupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss with adaptive top-k positives
    based on |y_i - y_j|.

    Positives = k nearest neighbors in y (per anchor).
    Optional soft weighting inside top-k via exp(-Δy / sigma).
    """

    def __init__(self, temperature=0.1, k=16, sigma=0.3, eps=1e-8, print_stats=False):
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.sigma = sigma
        self.eps = eps
        self.print_stats = print_stats

    def forward(self, z, y):
        """
        z: Tensor [B, D]  - embeddings (CLS token)
        y: Tensor [B]     - continuous targets (standardized)
        """

        device = z.device
        B = z.size(0)

        assert self.k < B, f"k must be < batch size (k={self.k}, B={B})"

        # Normalize embeddings
        z = F.normalize(z, dim=1)

        # Pairwise similarity (fp32 for stability)
        sim = torch.matmul(z, z.T) / self.temperature   # [B, B]

        # Mask self-similarity
        mask = ~torch.eye(B, device=device, dtype=torch.bool)

        # Pairwise label distance
        y = y.view(-1, 1)
        y_diff = torch.abs(y - y.T)                      # [B, B]

        # -----------------------------------------
        # Top-k selection (adaptive positives)
        # -----------------------------------------
        y_diff_masked = y_diff.masked_fill(~mask, float("inf"))
        _, topk_idx = torch.topk(y_diff_masked, self.k, dim=1, largest=False)

        # Build positive mask
        pos_mask = torch.zeros_like(y_diff, dtype=torch.bool)
        pos_mask.scatter_(1, topk_idx, True)

        # -----------------------------------------
        # Soft weights inside top-k (optional)
        # -----------------------------------------
        weights = torch.exp(-y_diff / self.sigma)
        weights = weights * pos_mask

        # Normalize weights per anchor
        weights_sum = weights.sum(dim=1, keepdim=True).clamp_min(self.eps)
        weights = weights / weights_sum

        # -----------------------------------------
        # Diagnostics
        # -----------------------------------------
        if True:
            pos_counts = pos_mask.sum(dim=1)
            print(
                f"[TopK-SupCon] positives per anchor | "
                f"min={pos_counts.min().item()} "
                f"mean={pos_counts.float().mean().item():.1f} "
                f"max={pos_counts.max().item()}"
            )

        # -----------------------------------------
        # Log-softmax over similarities (fp32-safe)
        # -----------------------------------------
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(~mask, float("-inf")).float(),
            dim=1,
            keepdim=True
        ).to(sim.dtype)

        # Weighted contrastive loss
        loss = -(weights * log_prob).sum(dim=1)

        return loss.mean()

class SoftSupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss with soft positives based on |y_i - y_j|.
    """

    def __init__(self, temperature=0.1, sigma=1.0, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.sigma = sigma
        self.eps = eps

    def forward(self, z, y):
        """
        z: Tensor [B, D]  - embeddings (CLS token)
        y: Tensor [B]     - continuous targets (standardized)
        """

        device = z.device
        B = z.size(0)

        # Normalize embeddings
        z = F.normalize(z, dim=1)

        # Pairwise cosine similarity
        sim = torch.matmul(z, z.T) / self.temperature   # [B, B]

        # Remove self-similarity
        mask = ~torch.eye(B, device=device).bool()

        # Pairwise absolute label difference
        y = y.view(-1, 1)
        y_diff = torch.abs(y - y.T)                      # [B, B]

        # Soft positive weights
        weights = torch.exp(-y_diff / self.sigma)        # [B, B]

        # Zero self-pairs
        weights = weights * mask

        # ----------------------------
        # Positive count diagnostics
        # ----------------------------
        if True:
            pos_counts = (weights > 0).sum(dim=1)        # [B]
            print(
                f"[SupCon] positives per anchor | "
                f"min={pos_counts.min().item()} "
                f"mean={pos_counts.float().mean().item():.1f} "
                f"max={pos_counts.max().item()}"
            )

        # Log-softmax over similarities
        log_prob = sim - torch.logsumexp(
            sim.masked_fill(~mask, float('-inf')).float(),
            dim=1,
            keepdim=True
        ).to(sim.dtype)

        # Weighted contrastive loss
        numerator = (weights * log_prob).sum(dim=1)
        denominator = weights.sum(dim=1) + self.eps

        loss = -numerator / denominator

        return loss.mean()

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature 

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (features.get_device() if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss
