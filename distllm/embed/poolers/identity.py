"""Identity Pooler for pre-pooled embeddings.

This pooler is designed for use with encoders that already return pooled
embeddings (e.g., NVEmbedEncoder). It simply removes the fake sequence
dimension without any actual pooling operation.
"""

from __future__ import annotations

from typing import Literal

import torch

from distllm.utils import BaseConfig


class IdentityPoolerConfig(BaseConfig):
    """Configuration for the IdentityPooler."""

    name: Literal['identity'] = 'identity'  # type: ignore[assignment]


class IdentityPooler:
    """Identity Pooler for pre-pooled embeddings.

    This pooler is a pass-through that handles embeddings which are
    already pooled but have a fake sequence dimension for interface
    compatibility.

    Expected input shape: [B, 1, D] (pre-pooled with fake seq dim)
    Output shape: [B, D]

    This is designed for use with encoders like NVEmbedEncoder that
    perform pooling internally.
    """

    def __init__(self, config: IdentityPoolerConfig) -> None:
        """Initialize the pooler with the configuration."""
        self.config = config

    def pool(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the embeddings with the sequence dimension removed.

        Parameters
        ----------
        embeddings : torch.Tensor
            The pre-pooled embeddings.
            Expected shape: [num_sequences, 1, embedding_size]
            Also handles: [num_sequences, embedding_size] (pass-through)

        attention_mask : torch.Tensor
            The attention mask (unused, kept for interface compatibility).
            (shape: [num_sequences, sequence_length])

        Returns
        -------
        torch.Tensor
            The embeddings with sequence dimension removed.
            (shape: [num_sequences, embedding_size])
        """
        # Handle both [B, 1, D] and [B, D] inputs
        if embeddings.dim() == 3:
            # Squeeze out the fake sequence dimension
            return embeddings.squeeze(1)
        elif embeddings.dim() == 2:
            # Already [B, D], return as-is
            return embeddings
        else:
            raise ValueError(
                f'IdentityPooler expects 2D or 3D input, got {embeddings.dim()}D '
                f'with shape {embeddings.shape}',
            )
