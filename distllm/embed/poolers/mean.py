"""Mean Pooler."""

from __future__ import annotations

from typing import Literal

import torch

from distllm.utils import BaseConfig


def average_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    exclude_special_tokens: bool = True,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).
    exclude_special_tokens : bool
        Whether to exclude the first and last tokens (BOS/EOS) from pooling.
        Set to False for models that don't use special tokens (e.g., ESMFold).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Clone attention mask to avoid modifying the original
    mask = attention_mask.clone().float()

    if exclude_special_tokens:
    # Get the sequence lengths
        seq_lengths = attention_mask.sum(dim=1)

        # Set the first token (BOS) to 0 for all sequences
        mask[:, 0] = 0

        # Set the last valid token (EOS) to 0 for each sequence
        # Use proper batch indexing: mask[i, seq_lengths[i] - 1] = 0
        batch_size = mask.shape[0]
        batch_indices = torch.arange(batch_size, device=mask.device)
        # Clamp to avoid negative indices for very short sequences
        last_indices = torch.clamp(seq_lengths - 1, min=0).long()
        mask[batch_indices, last_indices] = 0

    # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
    pool_mask = mask.unsqueeze(-1).expand(embeddings.shape)

    # Sum the embeddings over the sequence length
    sum_embeds = torch.sum(embeddings * pool_mask, dim=1)

    # Count valid positions per sequence
    # Use min=1e-5 to avoid division by zero (FP16 safe - min positive ~6e-8)
    sum_mask = torch.clamp(pool_mask.sum(dim=1), min=1e-5)

    # Compute mean pooled embeddings for each sequence
    pooled = sum_embeds / sum_mask

    # Check for NaN and warn
    if torch.isnan(pooled).any():
        import sys
        nan_count = torch.isnan(pooled).sum().item()
        total = pooled.numel()
        print(
            f'[WARNING] Mean pooling produced {nan_count}/{total} NaN values. '
            f'This may be due to FP16 numerical instability.',
            file=sys.stderr,
        )

    return pooled


class MeanPoolerConfig(BaseConfig):
    """Configuration for the MeanPooler."""

    name: Literal['mean'] = 'mean'  # type: ignore[assignment]
    # Whether to exclude the first and last tokens from pooling
    # Set to False for models without special tokens (e.g., ESMFold)
    exclude_special_tokens: bool = True


class MeanPooler:
    """Mean Pooler.

    Pooler that averages the hidden states using the attention mask.
    Can optionally exclude start/end tokens (for models with BOS/EOS).
    """

    def __init__(self, config: MeanPoolerConfig) -> None:
        """Initialize the pooler with the configuration."""
        self.config = config

    def pool(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            The embeddings to pool.
            (shape: [num_sequences, sequence_length, embedding_size])

        attention_mask : torch.Tensor
            The attention mask.
            (shape: [num_sequences, sequence_length])

        Returns
        -------
        torch.Tensor
            The pooled embeddings.
            (shape: [num_sequences, embedding_size])
        """
        return average_pool(
            embeddings,
            attention_mask,
            exclude_special_tokens=self.config.exclude_special_tokens,
        )
