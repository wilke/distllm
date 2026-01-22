"""Full sequence Embedder."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import Field
from torch.utils.data import DataLoader
from tqdm import tqdm

from distllm.embed.embedders.base import EmbedderResult
from distllm.embed.encoders.base import Encoder
from distllm.embed.poolers.base import Pooler
from distllm.utils import BaseConfig


@torch.no_grad()
def compute_embeddings(
    dataloader: DataLoader,
    encoder: Encoder,
    pooler: Pooler,
    normalize: bool = False,
) -> np.ndarray:
    """Compute pooled hidden embeddings.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    encoder : Encoder
        The encoder to use for inference.
    pooler : Pooler
        The pooler to use for pooling the embeddings.
    normalize : bool, optional
        Whether to normalize the embeddings, by default False.

    Returns
    -------
    np.ndarray
        A numpy array of pooled hidden embeddings.
    """
    # Get the number of embeddings and the embedding size
    num_embeddings = len(dataloader.dataset)

    # Initialize a torch tensor for storing embeddings in host memory
    all_embeddings = torch.empty(
        (num_embeddings, encoder.embedding_size),
        dtype=encoder.dtype,
    )

    # Index for storing embeddings
    idx = 0

    for batch in tqdm(dataloader):
        # Move the batch to the model device
        inputs = batch.to(encoder.device)

        # Get the model outputs with a forward pass
        embeddings = encoder.encode(inputs)

        # Compute the pooled embeddings
        pooled_embeds = pooler.pool(embeddings, inputs.attention_mask)

        # Normalize the embeddings
        if normalize:
            pooled_embeds = F.normalize(pooled_embeds, p=2, dim=-1)

        # Get the batch size
        batch_size = inputs.attention_mask.shape[0]

        # Store the pooled embeddings in the output buffer
        all_embeddings[idx : idx + batch_size, :] = pooled_embeds.cpu()

        # Increment the output buffer index by the batch size
        idx += batch_size

    # Convert to float32 if bfloat16 (numpy doesn't support bf16)
    if all_embeddings.dtype == torch.bfloat16:
        all_embeddings = all_embeddings.float()
    return all_embeddings.numpy()


@torch.no_grad()
def compute_multi_embeddings(
    dataloader: DataLoader,
    encoder: Encoder,
    pooler: Pooler,
    normalize: bool = False,
) -> dict[str, np.ndarray]:
    """Compute pooled hidden embeddings for multi-representation encoders.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to use for batching the data.
    encoder : Encoder
        The encoder to use for inference. Must return a dict of tensors.
    pooler : Pooler
        The pooler to use for pooling the embeddings.
    normalize : bool, optional
        Whether to normalize the embeddings, by default False.

    Returns
    -------
    dict[str, np.ndarray]
        A dict mapping embedding names to numpy arrays of pooled embeddings.
    """
    num_embeddings = len(dataloader.dataset)
    embedding_sizes = encoder.embedding_size  # dict[str, int]

    # Initialize storage for each embedding type
    all_embeddings: dict[str, torch.Tensor] = {
        name: torch.empty((num_embeddings, size), dtype=encoder.dtype)
        for name, size in embedding_sizes.items()
    }

    idx = 0

    for batch in tqdm(dataloader):
        inputs = batch.to(encoder.device)

        # Get dict of embeddings from encoder
        embeddings_dict = encoder.encode(inputs)

        batch_size = inputs.attention_mask.shape[0]

        # Pool each embedding type separately
        for name, embeddings in embeddings_dict.items():
            pooled = pooler.pool(embeddings, inputs.attention_mask)

            if normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)

            all_embeddings[name][idx : idx + batch_size, :] = pooled.cpu()

        idx += batch_size

    # Convert to float32 if bfloat16 (numpy doesn't support bf16)
    result = {}
    for name, arr in all_embeddings.items():
        if arr.dtype == torch.bfloat16:
            arr = arr.float()
        result[name] = arr.numpy()
    return result


class FullSequenceEmbedderConfig(BaseConfig):
    """Configuration for the full sequence embedder."""

    name: Literal['full_sequence'] = 'full_sequence'  # type: ignore[assignment]
    normalize_embeddings: bool = Field(
        False,
        description='Whether to return normalized the embeddings.',
    )


class FullSequenceEmbedder:
    """Embedder for full sequence embeddings."""

    def __init__(self, config: FullSequenceEmbedderConfig) -> None:
        """Initialize the embedder with the configuration."""
        self.config = config

    def embed(
        self,
        dataloader: DataLoader,
        encoder: Encoder,
        pooler: Pooler,
    ) -> EmbedderResult:
        """Embed the sequences.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader to use for batching the data.
        encoder : Encoder
            The encoder to use for inference.
        pooler : Pooler
            The pooler to use for pooling the embeddings.

        Returns
        -------
        EmbedderResult
            Dataclass with the embeddings, text, and optional metadata.
        """
        # Check if encoder returns multiple embeddings (dict embedding_size)
        embedding_size = encoder.embedding_size
        is_multi = isinstance(embedding_size, dict)

        if is_multi:
            # Multi-representation mode: compute each embedding separately
            named_embeddings = compute_multi_embeddings(
                dataloader=dataloader,
                encoder=encoder,
                pooler=pooler,
                normalize=self.config.normalize_embeddings,
            )

            # Use the first embedding as the primary (for backward compat)
            first_name = next(iter(named_embeddings))
            primary_embeddings = named_embeddings[first_name]

            return EmbedderResult(
                embeddings=primary_embeddings,
                text=dataloader.dataset.data,
                metadata=dataloader.dataset.metadata,
                named_embeddings=named_embeddings,
            )

        # Standard single-embedding mode
        embeddings = compute_embeddings(
            dataloader=dataloader,
            encoder=encoder,
            pooler=pooler,
            normalize=self.config.normalize_embeddings,
        )

        return EmbedderResult(
            embeddings=embeddings,
            text=dataloader.dataset.data,
            metadata=dataloader.dataset.metadata,
        )
