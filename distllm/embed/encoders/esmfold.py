"""Encoder for ESMFold structure embeddings.

This encoder extracts embeddings from ESMFold's STRUCTURE MODULE,
not the ESM-2 language model stem. These embeddings encode 3D
structural information learned during structure prediction.
"""

from __future__ import annotations

from typing import Literal

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class EsmFoldEncoderConfig(BaseConfig):
    """Config for the ESMFold structure encoder.

    ESMFold predicts protein 3D structure from sequence. This encoder
    extracts embeddings from the STRUCTURE MODULE (folding trunk),
    which contain learned 3D structural information - NOT the ESM-2
    sequence embeddings.

    Available representations:
    - 'states': Hidden states from the folding trunk (structure module)
    - 's_z': Pairwise residue embeddings (structural relationships)
    - 's_s': ESM-2 stem embeddings (sequence-based, NOT recommended
             if you already have ESM-2 embeddings)
    """

    # The name of the encoder
    name: Literal['esmfold'] = 'esmfold'  # type: ignore[assignment]
    # The model id (HuggingFace model)
    pretrained_model_name_or_path: str = 'facebook/esmfold_v1'
    # Use half precision (recommended for H100s)
    half_precision: bool = True
    # Set the model to evaluation mode
    eval_mode: bool = True
    # Maximum sequence length (ESMFold can handle up to 2048)
    max_length: int = 1024
    # Which representation to extract:
    # - 'states': hidden states from folding trunk (STRUCTURE, recommended)
    # - 's_z': pairwise residue embeddings (STRUCTURE, 2D relationships)
    # - 'combined': BOTH states + s_z concatenated (most complete)
    # - 's_s': ESM-2 stem embeddings (SEQUENCE, not recommended)
    representation: Literal['states', 's_z', 'combined', 's_s'] = 'states'
    # Multi-representation mode: extract BOTH states and s_z as separate
    # embeddings. When True, outputs a dict instead of a single tensor.
    # The embedder will pool each separately and write to separate columns.
    # This overrides the 'representation' setting.
    multi_representation: bool = False
    # Number of recycles for structure prediction
    # Higher values = better structure prediction = better embeddings
    # Recommended: 4 for best quality, 1-2 for speed
    num_recycles: int | None = 4


class EsmFoldEncoder:
    """Encoder for ESMFold STRUCTURE embeddings.

    This encoder extracts embeddings from ESMFold's structure prediction
    module (folding trunk), NOT the ESM-2 language model. These embeddings
    encode learned 3D structural information.

    Available representations:
    - 'states': Hidden states from the folding trunk after structure
                prediction. These encode 3D geometric information learned
                during the iterative refinement process.
    - 's_z': Pairwise residue embeddings that capture structural
             relationships between all residue pairs (contact-like info).
    - 's_s': ESM-2 stem output (sequence-based, not structure).

    For structure embeddings, use 'states' (recommended) or 's_z'.

    Reference: https://www.science.org/doi/10.1126/science.ade2574
    """

    def __init__(self, config: EsmFoldEncoderConfig) -> None:
        """Initialize the ESMFold encoder."""
        from transformers import AutoTokenizer
        from transformers import EsmForProteinFolding

        # Load model
        model = EsmForProteinFolding.from_pretrained(
            config.pretrained_model_name_or_path,
        )

        # Set number of recycles (more recycles = better structure)
        if config.num_recycles is not None:
            model.config.num_recycles = config.num_recycles

        # Convert to half precision if requested
        if config.half_precision:
            model.half()

        # Set to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Load tokenizer (ESMFold uses ESM-2 tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
        )
        tokenizer.model_max_length = config.max_length

        # Store config and model
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._representation = config.representation
        self._multi_representation = config.multi_representation

        # Cache embedding sizes for different representations
        trunk_config = model.config.esmfold_config.trunk
        self._embedding_sizes = {
            'states': trunk_config.sequence_state_dim,  # Folding trunk hidden
            's_z': trunk_config.pairwise_state_dim,     # Pairwise embeddings
            's_s': trunk_config.sequence_state_dim,     # ESM-2 projected
            # Combined = states + s_z concatenated
            'combined': (
                trunk_config.sequence_state_dim + trunk_config.pairwise_state_dim
            ),
        }

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int | dict[str, int]:
        """Get the embedding size based on the selected representation.

        Returns
        -------
        int | dict[str, int]
            If multi_representation=False, returns int.
            If multi_representation=True, returns dict mapping names to sizes.
        """
        if self._multi_representation:
            return {
                'structure': self._embedding_sizes['states'],
                'pairwise': self._embedding_sizes['s_z'],
            }
        return self._embedding_sizes[self._representation]

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self._tokenizer

    def encode(
        self,
        batch_encoding: BatchEncoding,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Encode sequences and extract STRUCTURE embeddings.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequences (containing input_ids
            and attention_mask).

        Returns
        -------
        torch.Tensor | dict[str, torch.Tensor]
            If multi_representation=False: single tensor
                (shape: [num_sequences, sequence_length, embedding_size])
            If multi_representation=True: dict with 'structure' and 'pairwise'
                keys, each containing a tensor.
        """
        # Move inputs to device
        input_ids = batch_encoding['input_ids'].to(self.device)
        attention_mask = batch_encoding['attention_mask'].to(self.device)

        # Forward pass through ESMFold (includes structure prediction)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Multi-representation mode: return both as dict
        if self._multi_representation:
            return {
                'structure': outputs.states,  # [B, L, 1024]
                'pairwise': outputs.s_z.mean(dim=2),  # [B, L, 128]
            }

        # Single representation mode
        if self._representation == 'states':
            # Hidden states from the FOLDING TRUNK (structure module)
            # These encode 3D structural information after iterative refinement
            # Shape: [batch, seq_len, sequence_state_dim]
            embeddings = outputs.states
        elif self._representation == 's_z':
            # Pairwise residue embeddings (structural relationships)
            # Shape: [batch, seq_len, seq_len, pairwise_state_dim]
            # Mean-pool over one dimension to get per-residue embeddings
            # -> [batch, seq_len, pairwise_state_dim]
            embeddings = outputs.s_z.mean(dim=2)
        elif self._representation == 'combined':
            # BOTH structure representations concatenated
            # states: [batch, seq_len, sequence_state_dim]
            # s_z pooled: [batch, seq_len, pairwise_state_dim]
            # -> [batch, seq_len, sequence_state_dim + pairwise_state_dim]
            states = outputs.states
            s_z_pooled = outputs.s_z.mean(dim=2)
            embeddings = torch.cat([states, s_z_pooled], dim=-1)
        elif self._representation == 's_s':
            # ESM-2 stem embeddings (sequence-based, NOT structure)
            # Only use if you specifically want sequence info
            embeddings = outputs.s_s
        else:
            raise ValueError(
                f'Unknown representation: {self._representation}. '
                f'Expected "states", "s_z", "combined", or "s_s".',
            )

        return embeddings
