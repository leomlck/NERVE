import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchLinearDecoder(nn.Module):
    """
    A separate module that takes decoded tokens from the decoder
    and maps them to the appropriate patch size via learned linear projections.
    """
    def __init__(self, network_list, decoder_embed_dim):
        """
        Args:
            network_list (dict): Dictionary mapping network names to their indices.
            decoder_embed_dim (int): Size of the decoder embedding.
        """
        super().__init__()

        # Compute the maximum patch size for padding
        self.max_patch_size = max(len(idx_a) * len(idx_b) for i, (net_a, idx_a) in enumerate(network_list.items())
                                  for j, (net_b, idx_b) in enumerate(network_list.items()) if i <= j)

        # Define a linear mapping for each unique patch (A-B)
        self.patch_decoders = nn.ModuleDict({
            f"{net_a}_{net_b}": nn.Linear(decoder_embed_dim, len(idx_a) * len(idx_b))
            for i, (net_a, idx_a) in enumerate(network_list.items())
            for j, (net_b, idx_b) in enumerate(network_list.items())
            if i <= j  # Only store A-B, not B-A
        })

    def forward(self, pred_tokens):
        """
        Decodes tokens into patches using learned linear projections.

        Args:
            pred_tokens (Tensor): Decoded tokens from the transformer (B, num_patches+1, decoder_embed_dim).

        Returns:
            Tensor: Predicted patches of shape (B, num_patches, max_patch_size).
        """
        pred_patches = []
        patch_idx = 1  # Skip CLS token

        for patch_key, decoder in self.patch_decoders.items():
            decoded_patch = decoder(pred_tokens[:, patch_idx])  # Linear transformation

            # Pad to max_patch_size
            pad_size = self.max_patch_size - decoded_patch.shape[1]
            padded_patch = F.pad(decoded_patch, (0, pad_size))  # Pad at the end

            pred_patches.append(padded_patch)
            patch_idx += 1

        return torch.stack(pred_patches, dim=1)  # Shape: (B, num_patches, max_patch_size)


class PatchLinearSharedDecoder(nn.Module):
    """
    Patch Linear Decoder with Shared Parameters:
    - A single shared linear layer is applied to all decoded patch embeddings.
    - Patches remain at max_patch_size after decoding.
    """
    def __init__(self, embed_dim, max_patch_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_patch_size = max_patch_size
        self.linear = nn.Linear(embed_dim, max_patch_size)  # Shared linear layer

    def forward(self, encoded_patches):
        """
        Args:
            encoded_patches: Tensor of shape (B, num_patches, embed_dim)
        Returns:
            Tensor of shape (B, num_patches, max_patch_size)
        """
        # Ignore CLS token (first token)
        encoded_patches = encoded_patches[:, 1:, :]  # Shape: (B, num_patches, embed_dim)

        decoded_patches = self.linear(encoded_patches)  # (B, num_patches, max_patch_size)
        return decoded_patches  # No trimming, remains at max_patch_size


class OuterProductDecoder(nn.Module):
    """
    Outer Product-Based Patch Decoder:
    
    Instead of having one independent linear layer per network pair (like in PatchLinearDecoder),
    this module assigns each network its own learnable decoding parameter matrix of shape 
    (|indices|, decoder_embed_dim). For each patch corresponding to a network pair (A,B), the module:
      - Retrieves D_A and D_B for networks A and B respectively.
      - Computes their outer product to yield a weight tensor of shape (|A|, |B|, decoder_embed_dim).
      - Flattens this tensor to (|A| * |B|, decoder_embed_dim).
      - Multiplies the corresponding decoder token (from a transformer’s output) with this weight
        matrix to produce a patch prediction of size (|A| * |B|) per subject.
      - Pads the patch to a fixed maximum patch size for consistency across patches.
    """
    def __init__(self, network_list, decoder_embed_dim):
        """
        Args:
            network_list (dict): Dictionary mapping network names to their indices.
            decoder_embed_dim (int): The dimensionality of the decoder embedding.
        """
        super().__init__()
        self.network_list = network_list
        self.decoder_embed_dim = decoder_embed_dim

        # Each network gets a learnable decoding parameter.
        # For each network, we have a parameter of shape (|indices|, decoder_embed_dim).
        self.dec_network_weights = nn.ParameterDict({
            net: nn.Parameter(torch.randn(len(indices), decoder_embed_dim))
            for net, indices in network_list.items()
        })

        # Compute the maximum patch size across all network pairs (for later padding).
        self.max_patch_size = max(
            len(idx_a) * len(idx_b)
            for i, (net_a, idx_a) in enumerate(network_list.items())
            for j, (net_b, idx_b) in enumerate(network_list.items())
            if i <= j
        )

        # Define the order of network pairs: all unique pairs (A, B) with i <= j.
        self.network_pairs = [
            (net_a, net_b)
            for i, (net_a, idx_a) in enumerate(network_list.items())
            for j, (net_b, idx_b) in enumerate(network_list.items())
            if i <= j
        ]

    def forward(self, pred_tokens):
        """
        Args:
            pred_tokens (Tensor): Decoded tokens from the transformer decoder,
                                    shape (B, num_patches+1, decoder_embed_dim).
                                    Assumes token 0 is [CLS] and should be skipped.
            
        Returns:
            Tensor: Reconstructed patches, shape (B, num_patches, max_patch_size).
        """
        B = pred_tokens.shape[0]
        patch_tokens = []
        patch_idx = 1  # Skip the [CLS] token.

        for net_A, net_B in self.network_pairs:
            # Retrieve the per-network decoding parameters.
            D_A = self.dec_network_weights[net_A]  # Shape: (|A|, decoder_embed_dim)
            D_B = self.dec_network_weights[net_B]  # Shape: (|B|, decoder_embed_dim)

            # Compute the outer product: 
            # This produces a tensor of shape (|A|, |B|, decoder_embed_dim)
            W_AB = torch.einsum('ij,kj->ikj', D_A, D_B)
            # Flatten the first two dimensions to form a weight matrix for the patch:
            # Shape: (|A| * |B|, decoder_embed_dim)
            W_AB_flat = W_AB.reshape(-1, self.decoder_embed_dim)

            # Retrieve the corresponding token (for the current patch) from the decoder output.
            # pred_tokens has shape (B, num_patches+1, decoder_embed_dim); we assume tokens[1:] map to patches.
            token = pred_tokens[:, patch_idx]  # Shape: (B, decoder_embed_dim)

            # Decode the patch: project the token through the computed weight matrix.
            decoded_patch = token @ W_AB_flat.T  # Shape: (B, |A|*|B|)

            # Pad the decoded patch to reach max_patch_size.
            pad_size = self.max_patch_size - decoded_patch.shape[1]
            if pad_size > 0:
                decoded_patch = F.pad(decoded_patch, (0, pad_size))
            patch_tokens.append(decoded_patch)
            patch_idx += 1

        # Stack all patch predictions along a new dimension.
        # Final shape: (B, num_patches, max_patch_size)
        return torch.stack(patch_tokens, dim=1)

