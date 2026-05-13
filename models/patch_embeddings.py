import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

class PatchLinearEmbedding(nn.Module):
    def __init__(self, network_list, embed_dim):
        """
        Args:
            network_list (dict): Dictionary where keys are network names and values are lists of indices.
            embed_dim (int): Size of the embedding vector for each patch.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim

        # Define a linear layer for each unique network pair (A, B) where A <= B
        self.patch_embeddings = nn.ModuleDict({
            f"{net_a}_{net_b}": nn.Linear(len(idx_a) * len(idx_b), embed_dim)
            for i, (net_a, idx_a) in enumerate(network_list.items())
            for j, (net_b, idx_b) in enumerate(network_list.items())
            if i <= j  # Ensure only unique (A, B) pairs
        })

    def forward(self, fc_matrices):
        """
        Args:
            fc_matrices (Tensor): Batch of functional connectivity matrices (B, N, N)

        Returns:
            Tensor: Embedded patches of size (B, num_patches, embed_dim)
        """
        B, N, _ = fc_matrices.shape
        embedded_patches = []

        for i, (net_a, idx_a) in enumerate(self.network_list.items()):
            for j, (net_b, idx_b) in enumerate(self.network_list.items()):
                if i > j:  # Skip redundant (B, A) pairs
                    continue

                # Extract patch corresponding to (net_a, net_b)
                patch = fc_matrices[:, idx_a][:, :, idx_b]  # Shape (B, len(idx_a), len(idx_b))
                patch = patch.view(B, -1)  # Flatten patch

                # Pass through corresponding linear layer
                patch_key = f"{net_a}_{net_b}"
                embedded_patch = self.patch_embeddings[patch_key](patch)
                embedded_patches.append(embedded_patch)

        # Stack patches along num_patches dimension
        return torch.stack(embedded_patches, dim=1)  # Shape (B, num_patches, embed_dim)

class PatchLinearSharedEmbedding(nn.Module):
    """
    Patch Linear Embedding with Shared Parameters:
    - Extracts patches from full FC matrices based on `network_list`
    - Pads all patches to `max_patch_size`
    - Applies a shared linear layer to all patches
    """

    def __init__(self, network_list, embed_dim):
        """
        Args:
            network_list (dict): Dictionary where keys are network names, and values are indices.
            embed_dim (int): Dimension of the output patch embeddings.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim
        self.max_patch_size = max(len(indices) ** 2 for indices in network_list.values())  # Max patch size

        self.shared_linear = nn.Linear(self.max_patch_size, embed_dim)  # Shared embedding layer

    def forward(self, fc_matrices):
        """
        Args:
            fc_matrices (Tensor): Full FC matrices of shape (B, N, N).
        Returns:
            patch_embeddings (Tensor): Embedded patches of shape (B, num_patches, embed_dim).
        """
        batch_size = fc_matrices.shape[0]
        patch_embeddings = []

        # Extract and embed each patch
        for net_A, indices_A in self.network_list.items():
            for net_B, indices_B in self.network_list.items():
                if net_A > net_B:  # Ensure only upper triangle patches (A-B, no duplicates)
                    continue

                # Extract patch from FC matrix
                patch = fc_matrices[:, indices_A, :][:, :, indices_B]  # (B, |A|, |B|)
                patch = patch.reshape(batch_size, -1)  # Flatten

                # Pad to max_patch_size
                pad_size = self.max_patch_size - patch.shape[1]
                patch = F.pad(patch, (0, pad_size), "constant", 0)  # Pad with zeros

                # Apply shared embedding layer
                patch_emb = self.shared_linear(patch)  # (B, embed_dim)
                patch_embeddings.append(patch_emb)

        return torch.stack(patch_embeddings, dim=1)  # (B, num_patches, embed_dim)

class MLPEmbedding(nn.Module):
    def __init__(self, network_list, embed_dim):
        """
        Args:
            network_list (dict): Dictionary where keys are network names and values are lists of indices.
            embed_dim (int): Size of the embedding vector for each patch.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim

        # Define an MLP for each unique network pair (A, B) where A <= B
        # The MLP consists of: Linear -> ReLU -> Linear, mapping from the flattened patch dimension to embed_dim.
        self.patch_embeddings = nn.ModuleDict({
            f"{net_a}_{net_b}": nn.Sequential(
                nn.Linear(len(idx_a) * len(idx_b), embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            for i, (net_a, idx_a) in enumerate(network_list.items())
            for j, (net_b, idx_b) in enumerate(network_list.items())
            if i <= j  # Ensure only unique (A, B) pairs
        })

    def forward(self, fc_matrices):
        """
        Args:
            fc_matrices (Tensor): Batch of functional connectivity matrices (B, N, N)

        Returns:
            Tensor: Embedded patches of size (B, num_patches, embed_dim)
        """
        B, N, _ = fc_matrices.shape
        embedded_patches = []

        for i, (net_a, idx_a) in enumerate(self.network_list.items()):
            for j, (net_b, idx_b) in enumerate(self.network_list.items()):
                if i > j:  # Skip redundant (B, A) pairs
                    continue

                # Extract patch corresponding to (net_a, net_b)
                patch = fc_matrices[:, idx_a][:, :, idx_b]  # Shape (B, len(idx_a), len(idx_b))
                patch = patch.view(B, -1)  # Flatten patch

                # Pass through the corresponding MLP
                patch_key = f"{net_a}_{net_b}"
                embedded_patch = self.patch_embeddings[patch_key](patch)
                embedded_patches.append(embedded_patch)

        # Stack patches along num_patches dimension
        return torch.stack(embedded_patches, dim=1)  # Shape (B, num_patches, embed_dim)

class MLPSharedEmbedding(nn.Module):
    """
    Patch Linear Embedding with Shared Parameters:
    - Extracts patches from full FC matrices based on `network_list`
    - Pads all patches to `max_patch_size`
    - Applies a shared linear layer to all patches
    """

    def __init__(self, network_list, embed_dim):
        """
        Args:
            network_list (dict): Dictionary where keys are network names, and values are indices.
            embed_dim (int): Dimension of the output patch embeddings.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim
        self.max_patch_size = max(len(indices) ** 2 for indices in network_list.values())  # Max patch size

        self.shared_linear = nn.Sequential(
                nn.Linear(self.max_patch_size, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
                )# Shared embedding layer

    def forward(self, fc_matrices):
        """
        Args:
            fc_matrices (Tensor): Full FC matrices of shape (B, N, N).
        Returns:
            patch_embeddings (Tensor): Embedded patches of shape (B, num_patches, embed_dim).
        """
        batch_size = fc_matrices.shape[0]
        patch_embeddings = []

        # Extract and embed each patch
        for net_A, indices_A in self.network_list.items():
            for net_B, indices_B in self.network_list.items():
                if net_A > net_B:  # Ensure only upper triangle patches (A-B, no duplicates)
                    continue

                # Extract patch from FC matrix
                patch = fc_matrices[:, indices_A, :][:, :, indices_B]  # (B, |A|, |B|)
                patch = patch.reshape(batch_size, -1)  # Flatten

                # Pad to max_patch_size
                pad_size = self.max_patch_size - patch.shape[1]
                patch = F.pad(patch, (0, pad_size), "constant", 0)  # Pad with zeros

                # Apply shared embedding layer
                patch_emb = self.shared_linear(patch)  # (B, embed_dim)
                patch_embeddings.append(patch_emb)

        return torch.stack(patch_embeddings, dim=1)  # (B, num_patches, embed_dim)


class SharedNetworkEmbedding(nn.Module):
    """
    Shared network-specific weights: Each network has its own weight, 
    and patch embeddings use a fusion of weights.
    """
    def __init__(self, network_list, embed_dim):
        """
        network_list: Dict of networks {name: indices}
        embed_dim: Embedding size
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.network_weights = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), embed_dim))
            for net, indices in network_list.items()
        })
        self.network_pairs = list(itertools.combinations(network_list.keys(), 2))  # All network pairs

    def forward(self, x):
        batch_size = x.shape[0]
        patch_tokens = []

        for net_A, net_B in self.network_pairs:
            row_idx, col_idx = self.network_weights[net_A], self.network_weights[net_B]
            patch = x[:, row_idx, :][:, :, col_idx].reshape(batch_size, -1)

            W_A = self.network_weights[net_A]
            W_B = self.network_weights[net_B]
            W_AB = W_A + W_B  # Shared fusion

            patch_emb = patch @ W_AB
            patch_tokens.append(patch_emb)

        return torch.stack(patch_tokens, dim=1)


class OuterProductEmbedding(nn.Module):
    """
    Outer Product-Based Embedding: Each network has a learnable vector, 
    and patch embeddings are computed using outer products.
    """
    def __init__(self, network_list, embed_dim):
        """
        Args:
            network_list (dict): Dictionary mapping network names to their indices.
            embed_dim (int): Size of the embedding vector for each network.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.network_list = network_list
        
        # Trainable vector embeddings for each network
        self.network_weights = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), embed_dim))
            for net, indices in network_list.items()
        })

        # Generate all unique (A, B) pairs, including self-pairs (A, A)
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b]

    def forward(self, x):
        """
        Args:
            x (Tensor): Functional Connectivity matrix of shape (B, N, N)

        Returns:
            Tensor: Patch embeddings of shape (B, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        patch_tokens = []

        for net_A, net_B in self.network_pairs:
            # Get indices corresponding to networks A and B
            idx_A, idx_B = self.network_list[net_A], self.network_list[net_B]
            
            # Extract FC patch from input matrix
            patch = x[:, idx_A, :][:, :, idx_B].reshape(batch_size, -1)  # (B, |A| * |B|)

            # Get trainable network-specific weight vectors
            W_A = self.network_weights[net_A]  # (|A|, embed_dim)
            W_B = self.network_weights[net_B]  # (|B|, embed_dim)

            # Compute outer product (|A|, embed_dim) ⊗ (|B|, embed_dim) → (|A|, |B|, embed_dim)
            W_AB = torch.einsum('ij,kj->ikj', W_A, W_B).reshape(-1, self.embed_dim)  # (|A| * |B|, embed_dim)

            # Compute final embedding
            patch_emb = patch @ W_AB  # (B, embed_dim)
            patch_tokens.append(patch_emb)

        return torch.stack(patch_tokens, dim=1)  # (B, num_patches, embed_dim)


class OuterMLPEmbedding(nn.Module):
    """
    Factorized Embedding Fusion with Activation.

    For each network, we factorize its parameters into:
      - a base parameter of shape (|indices|, factor_dim)
      - a projection parameter of shape (factor_dim, embed_dim)

    The network embedding is computed as:
        W = activation(base @ proj)

    For a pair of networks (A, B), the fusion is done via an outer product:
        W_AB = outer(W_A, W_B)  → shape (|A|, |B|, embed_dim)
    which is then flattened and used to project the corresponding patch
    from the functional connectivity matrix.
    """
    def __init__(self, network_list, embed_dim, factor_dim=64, activation=nn.ReLU()):
        """
        Args:
            network_list (dict): Dictionary mapping network names to lists of indices.
            embed_dim (int): Final embedding dimension for each patch.
            factor_dim (int): Intermediate dimension for the factorized representation.
            activation (nn.Module): Activation function to apply to each network embedding.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim
        self.factor_dim = factor_dim
        self.activation = activation

        # Initialize factorized parameters for each network.
        # base: (|indices|, factor_dim)
        # proj: (factor_dim, embed_dim)
        self.network_base = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), factor_dim))
            for net, indices in network_list.items()
        })
        self.network_proj = nn.ParameterDict({
            net: nn.Parameter(torch.randn(factor_dim, embed_dim))
            for net in network_list.keys()
        })

        # Create list of unique network pairs (including self-pairs)
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b]

    def forward(self, x):
        """
        Args:
            x (Tensor): Batch of functional connectivity matrices of shape (B, N, N)
        Returns:
            Tensor: Patch embeddings of shape (B, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        patch_tokens = []

        for net_A, net_B in self.network_pairs:
            # Retrieve indices for networks A and B.
            idx_A, idx_B = self.network_list[net_A], self.network_list[net_B]

            # Extract the patch from the FC matrix corresponding to (net_A, net_B).
            # x[:, idx_A, :][:, :, idx_B] → shape: (B, |A|, |B|)
            patch = x[:, idx_A, :][:, :, idx_B].reshape(batch_size, -1)  # (B, |A|*|B|)

            # Compute the factorized network embeddings with activation:
            # W_A: (|A|, embed_dim) and W_B: (|B|, embed_dim)
            W_A = self.activation(self.network_base[net_A] @ self.network_proj[net_A])
            W_B = self.activation(self.network_base[net_B] @ self.network_proj[net_B])

            # Fuse the network embeddings via an outer product:
            # torch.einsum('ij,kj->ikj', ...) computes elementwise products, yielding (|A|, |B|, embed_dim)
            # Then, reshape to flatten the first two dimensions: (|A|*|B|, embed_dim)
            W_AB = torch.einsum('ij,kj->ikj', W_A, W_B).reshape(-1, self.embed_dim)

            # Compute the patch embedding: Multiply the flattened patch (B, |A|*|B|)
            # with the fused weights (|A|*|B|, embed_dim) → (B, embed_dim)
            patch_emb = patch @ W_AB
            patch_tokens.append(patch_emb)

        # Stack all patch embeddings for each network pair along a new dimension.
        # Final shape: (B, num_patches, embed_dim)
        return torch.stack(patch_tokens, dim=1)

class ConcatFusionMLPWithPatchMean(nn.Module):
    """
    Fuse network embeddings by concatenation followed by an MLP,
    and incorporate patch information by modulating the fused vector with
    the mean value of the connectivity patch.
    
    Each network is assigned a learnable embedding matrix of shape (|indices|, embed_dim).
    For each network, we aggregate by taking the mean over indices.
    For a network pair (A, B), we fuse by concatenating their aggregated embeddings
    (resulting in a vector of size 2*embed_dim) and feed this vector to an MLP
    to obtain a fused vector of shape (embed_dim,).
    Finally, we compute the mean of the corresponding FC patch and multiply it with the fused vector.
    """
    def __init__(self, network_list, embed_dim, hidden_dim=128, activation=nn.ReLU()):
        """
        Args:
            network_list (dict): Dictionary mapping network names to lists of indices.
            embed_dim (int): Embedding dimension for each network.
            hidden_dim (int): Hidden dimension for the fusion MLP.
            activation (nn.Module): Activation function to use in the MLP.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim
        self.activation = activation
        
        # Each network gets a learnable embedding matrix (|indices|, embed_dim)
        self.network_weights = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), embed_dim))
            for net, indices in network_list.items()
        })
        
        # Create unique network pairs (including self-pairs)
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b]
        
        # MLP to fuse concatenated aggregated embeddings: (2*embed_dim -> embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Batch of functional connectivity matrices of shape (B, N, N)
        Returns:
            Tensor: Patch embeddings of shape (B, num_pairs, embed_dim)
        """
        batch_size = x.shape[0]
        patch_tokens = []
        
        for net_A, net_B in self.network_pairs:
            # Retrieve indices for networks A and B
            idx_A, idx_B = self.network_list[net_A], self.network_list[net_B]
            
            # Extract the FC patch for the network pair: shape (B, |A|, |B|)
            patch = x[:, idx_A, :][:, :, idx_B]
            # Compute the mean of the patch (scalar per matrix): shape (B,)
            patch_mean = patch.mean(dim=(1,2))
            
            # Compute aggregated (mean) embedding for network A and network B: shape (embed_dim,)
            agg_A = self.network_weights[net_A].mean(dim=0)
            agg_B = self.network_weights[net_B].mean(dim=0)
            
            # Concatenate aggregated embeddings: shape (2*embed_dim,)
            fused_input = torch.cat([agg_A, agg_B], dim=-1)
            # Fuse via MLP: shape (embed_dim,)
            fused_vector = self.fusion_mlp(fused_input)
            
            # Modulate the fused vector by the patch mean: for each example in the batch,
            # multiply the fused vector by the corresponding patch_mean scalar.
            # Resulting shape: (B, embed_dim)
            patch_emb = patch_mean.unsqueeze(1) * fused_vector.unsqueeze(0)
            patch_tokens.append(patch_emb)
        
        # Stack the patch embeddings from all network pairs along a new dimension:
        # Final shape: (B, num_pairs, embed_dim)
        return torch.stack(patch_tokens, dim=1)

class AdditionFusionWithPatchMean(nn.Module):
    """
    Fuse network embeddings by addition and modulate with the patch's mean value.

    For each network, a learnable embedding matrix of shape (|indices|, embed_dim) is defined.
    The aggregated embedding for a network is obtained by taking the mean over its indices.
    For a network pair (A, B), the fusion is performed by adding the aggregated embeddings:

        fused_vector = mean(W_A) + mean(W_B)

    The final patch embedding is obtained by scaling this fused vector with the mean value of the
    corresponding FC patch.
    """
    def __init__(self, network_list, embed_dim):
        """
        Args:
            network_list (dict): Mapping from network names to lists of indices.
            embed_dim (int): Embedding dimension for each network.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim

        # Create a learnable embedding for each network.
        # Each embedding is a parameter of shape (|indices|, embed_dim)
        self.network_weights = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), embed_dim))
            for net, indices in network_list.items()
        })

        # Create list of unique network pairs (including self-pairs)
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b]

    def forward(self, x):
        """
        Args:
            x (Tensor): Batch of FC matrices with shape (B, N, N)
        Returns:
            Tensor: Patch embeddings of shape (B, num_pairs, embed_dim)
        """
        batch_size = x.shape[0]
        patch_tokens = []

        for net_A, net_B in self.network_pairs:
            # Retrieve indices for networks A and B.
            idx_A, idx_B = self.network_list[net_A], self.network_list[net_B]

            # Extract the patch corresponding to the connectivity between networks A and B.
            # Shape: (B, |A|, |B|)
            patch = x[:, idx_A, :][:, :, idx_B].reshape(batch_size, -1)  # Flatten to (B, |A|*|B|)

            # Aggregate the network embeddings by taking the mean over indices.
            agg_A = self.network_weights[net_A].mean(dim=0)  # Shape: (embed_dim,)
            agg_B = self.network_weights[net_B].mean(dim=0)  # Shape: (embed_dim,)

            # Fuse by simple addition.
            fused_vector = agg_A + agg_B  # Shape: (embed_dim,)

            # Compute the mean value of the FC patch (a scalar per sample).
            patch_mean = patch.mean(dim=1)  # Shape: (B,)

            # Modulate the fused vector by the patch mean:
            # For each sample in the batch, scale the fused vector.
            patch_emb = patch_mean.unsqueeze(1) * fused_vector.unsqueeze(0)  # (B, embed_dim)

            patch_tokens.append(patch_emb)

        # Stack patch embeddings along a new dimension for each network pair.
        # Final shape: (B, num_pairs, embed_dim)
        return torch.stack(patch_tokens, dim=1)

class AdditionFusionMLPWithPatchMean(nn.Module):
    """
    Fuse network embeddings by addition and modulate with the patch's mean value.
    
    For each network, a learnable embedding matrix of shape (|indices|, embed_dim) is defined.
    The aggregated embedding for a network is obtained by taking the mean over its indices.
    For a network pair (A, B), the fusion is performed by adding the aggregated embeddings:
    
        fused_vector = mean(W_A) + mean(W_B)
    
    The final patch embedding is obtained by scaling this fused vector with the mean value of the
    corresponding FC patch.
    """
    def __init__(self, network_list, embed_dim, hidden_dim=128, activation=nn.ReLU()):
        """
        Args:
            network_list (dict): Mapping from network names to lists of indices.
            embed_dim (int): Embedding dimension for each network.
        """
        super().__init__()
        self.network_list = network_list
        self.embed_dim = embed_dim
        
        # Create a learnable embedding for each network.
        # Each embedding is a parameter of shape (|indices|, embed_dim)
        self.network_weights = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), embed_dim))
            for net, indices in network_list.items()
        })
        
        # Create list of unique network pairs (including self-pairs)
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b]

        # MLP to fuse concatenated aggregated embeddings: (2*embed_dim -> embed_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): Batch of FC matrices with shape (B, N, N)
        Returns:
            Tensor: Patch embeddings of shape (B, num_pairs, embed_dim)
        """
        batch_size = x.shape[0]
        patch_tokens = []
        
        for net_A, net_B in self.network_pairs:
            # Retrieve indices for networks A and B.
            idx_A, idx_B = self.network_list[net_A], self.network_list[net_B]
            
            # Extract the patch corresponding to the connectivity between networks A and B.
            # Shape: (B, |A|, |B|)
            patch = x[:, idx_A, :][:, :, idx_B].reshape(batch_size, -1)  # Flatten to (B, |A|*|B|)
            
            # Aggregate the network embeddings by taking the mean over indices.
            agg_A = self.network_weights[net_A].mean(dim=0)  # Shape: (embed_dim,)
            agg_B = self.network_weights[net_B].mean(dim=0)  # Shape: (embed_dim,)
            
            # Fuse by simple addition + MLP
            fused_vector = agg_A + agg_B  # Shape: (embed_dim,)
            fused_vector = self.fusion_mlp(fused_vector)
            
            # Compute the mean value of the FC patch (a scalar per sample).
            patch_mean = patch.mean(dim=1)  # Shape: (B,)
            
            # Modulate the fused vector by the patch mean:
            # For each sample in the batch, scale the fused vector.
            patch_emb = patch_mean.unsqueeze(1) * fused_vector.unsqueeze(0)  # (B, embed_dim)
            
            patch_tokens.append(patch_emb)
        
        # Stack patch embeddings along a new dimension for each network pair.
        # Final shape: (B, num_pairs, embed_dim)
        return torch.stack(patch_tokens, dim=1)


class GCNEmbedding(nn.Module):
    """
    Applies a unique GCN to each patch defined by a (network A, network B) pair.
    All (A, B) pairs such that A <= B are included (including self-pairs).
    """
    def __init__(self, network_list, embed_dim):
        super().__init__()
        self.network_list = network_list
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b] 
        self.embed_dim = embed_dim

        self.gcn_modules = nn.ModuleDict({
            f"{a}_{b}": nn.Sequential(
                GCNConv(in_channels=1, out_channels=embed_dim),
                nn.ReLU(),
                GCNConv(in_channels=embed_dim, out_channels=embed_dim)
            )
            for a, b in self.network_pairs
        })

    def forward(self, fc_matrices):
        # fc_matrices: (B, N, N)
        B = fc_matrices.size(0)
        device = fc_matrices.device
        patch_tokens = []

        for net_a, net_b in self.network_pairs:
            idx_a = self.network_list[net_a]
            idx_b = self.network_list[net_b]
            patch_tokens_batch = []

            for b in range(B):
                patch = fc_matrices[b][idx_a][:, idx_b]  # shape: (len_a, len_b)
                x = patch.flatten().unsqueeze(-1).to(device)  # (num_nodes, 1)
                num_nodes = x.size(0)

                # Define a trivial edge_index (fully connected graph could be added)
                edge_index = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)

                data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
                data.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

                gcn = self.gcn_modules[f"{net_a}_{net_b}"]
                out = gcn[0](data.x, data.edge_index)
                out = gcn[1](out)
                out = gcn[2](out, data.edge_index)

                pooled = global_mean_pool(out, data.batch)  # shape: (1, embed_dim)
                patch_tokens_batch.append(pooled)

            patch_tokens_batch = torch.cat(patch_tokens_batch, dim=0).unsqueeze(1)  # (B, 1, embed_dim)
            patch_tokens.append(patch_tokens_batch)

        # Concatenate over all patch tokens → (B, num_patches, embed_dim)
        patch_tokens = torch.cat(patch_tokens, dim=1)
        return patch_tokens

class SharedGCNEmbedding(nn.Module):
    """
    Applies the same GCN to every patch defined by a (network A, network B) pair.
    All (A, B) pairs such that A <= B are included (including self-pairs).
    """
    def __init__(self, network_list, embed_dim):
        super().__init__()
        self.network_list = network_list
        self.network_pairs = [(a, b) for a in network_list.keys() for b in network_list.keys() if a <= b] 
        self.embed_dim = embed_dim

        self.gcn1 = GCNConv(in_channels=1, out_channels=embed_dim)
        self.gcn2 = GCNConv(in_channels=embed_dim, out_channels=embed_dim)

    def forward(self, fc_matrices):
        B = fc_matrices.size(0)
        device = fc_matrices.device
        patch_tokens = []

        for net_a, net_b in self.network_pairs:
            idx_a = self.network_list[net_a]
            idx_b = self.network_list[net_b]
            patch_tokens_batch = []

            for b in range(B):
                patch = fc_matrices[b][idx_a][:, idx_b]  # shape: (len_a, len_b)
                x = patch.flatten().unsqueeze(-1).to(device)  # shape: (num_nodes, 1)
                num_nodes = x.size(0)

                edge_index = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(2, 1)

                data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
                data.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

                out = self.gcn1(data.x, data.edge_index)
                out = torch.relu(out)
                out = self.gcn2(out, data.edge_index)

                pooled = global_mean_pool(out, data.batch)  # shape: (1, embed_dim)
                patch_tokens_batch.append(pooled)

            patch_tokens_batch = torch.cat(patch_tokens_batch, dim=0).unsqueeze(1)  # (B, 1, embed_dim)
            patch_tokens.append(patch_tokens_batch)

        patch_tokens = torch.cat(patch_tokens, dim=1)  # (B, num_patches, embed_dim)
        return patch_tokens





