"""Encoder for ESMFold structure embeddings.

This encoder extracts embeddings from ESMFold's STRUCTURE MODULE,
not the ESM-2 language model stem. These embeddings encode 3D
structural information learned during structure prediction.
"""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig

# Flag to track if patch has been applied in this process
_COMPUTE_TM_PATCHED = False


def _patch_esmfold_compute_tm() -> None:
    """Patch ESMFold's compute_tm to handle half-precision edge cases.

    The original compute_tm function in transformers can fail with
    IndexError when using half precision due to numerical instability
    causing NaN values. This patch adds graceful handling for that case.

    Since we only need embeddings (not pTM scores), returning a default
    value when the computation fails is acceptable.

    This function is idempotent - safe to call multiple times.

    IMPORTANT: We must patch BOTH locations:
    1. transformers.models.esm.openfold_utils.loss.compute_tm
    2. transformers.models.esm.modeling_esmfold.compute_tm
    Because modeling_esmfold imports compute_tm directly at module level.
    """
    global _COMPUTE_TM_PATCHED
    if _COMPUTE_TM_PATCHED:
        return

    try:
        from transformers.models.esm.openfold_utils import loss as openfold_loss
        from transformers.models.esm import modeling_esmfold
    except ImportError:
        return  # Old transformers version, skip patching

    # Check if already patched (in case flag was reset)
    if hasattr(modeling_esmfold.compute_tm, '_is_patched'):
        _COMPUTE_TM_PATCHED = True
        return

    # Store original function (from the source module)
    _original_compute_tm = openfold_loss.compute_tm

    def _safe_compute_tm(
        logits: torch.Tensor,
        max_bin: int = 31,
        no_bins: int = 64,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute TM-score with graceful handling for numerical edge cases."""
        try:
            return _original_compute_tm(
                logits, max_bin=max_bin, no_bins=no_bins, **kwargs,
            )
        except IndexError:
            # Half-precision numerical instability caused empty nonzero result
            # Return a default pTM of 0.0 (we don't use this value anyway)
            batch_size = logits.shape[0]
            return torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)

    # Mark as patched
    _safe_compute_tm._is_patched = True  # type: ignore[attr-defined]

    # Apply the patch to BOTH locations
    # 1. The source module (for any other imports)
    openfold_loss.compute_tm = _safe_compute_tm
    # 2. The modeling_esmfold module (where EsmForProteinFolding.forward calls it)
    modeling_esmfold.compute_tm = _safe_compute_tm

    _COMPUTE_TM_PATCHED = True


class EsmFoldTokenizerWrapper:
    """Wrapper for ESMFold tokenizer that disables special tokens.

    ESMFold requires sequences WITHOUT special tokens (no BOS/EOS).
    This wrapper ensures add_special_tokens=False is always used.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        """Wrap the tokenizer."""
        self._tokenizer = tokenizer

    def __call__(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> BatchEncoding:
        """Tokenize with add_special_tokens=False."""
        # Force no special tokens for ESMFold
        kwargs['add_special_tokens'] = False
        return self._tokenizer(text, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped tokenizer."""
        return getattr(self._tokenizer, name)


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
    # Use half precision for memory efficiency.
    # IMPORTANT: When True, uses BFLOAT16 (not float16) because ESMFold
    # has severe numerical instability with float16 (produces 100% NaN).
    # BF16 has the same memory footprint as FP16 but better numerical range.
    # Requires GPU with BF16 support (e.g., A100, H100, RTX 30/40 series).
    half_precision: bool = True
    # Set the model to evaluation mode
    eval_mode: bool = True
    # Maximum sequence length (ESMFold can handle up to 2048)
    max_length: int = 1024
    # Minimum sequence length (sequences shorter than this will raise error)
    # ESMFold's pTM computation can fail on very short sequences
    min_length: int = 10
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
        # IMPORTANT: Apply the compute_tm patch BEFORE importing ESMFold model
        # This fixes half-precision numerical instability in pTM computation
        _patch_esmfold_compute_tm()

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
        # IMPORTANT: Use bfloat16, NOT float16!
        # ESMFold produces 100% NaN with float16 due to numerical instability.
        # BF16 has the same memory footprint but better numerical range.
        if config.half_precision:
            model.to(torch.bfloat16)

        # Set to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Load tokenizer (ESMFold uses ESM-2 tokenizer)
        # IMPORTANT: ESMFold requires add_special_tokens=False
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
        )
        tokenizer.model_max_length = config.max_length

        # Wrap tokenizer to enforce no special tokens
        wrapped_tokenizer = EsmFoldTokenizerWrapper(tokenizer)

        # Store config and model
        self.config = config
        self.model = model
        self._tokenizer = wrapped_tokenizer
        self._representation = config.representation
        self._multi_representation = config.multi_representation
        self._min_length = config.min_length
        self._num_recycles = config.num_recycles or model.config.num_recycles

        # Cache embedding sizes for different representations
        # NOTE: ESMFold has two different hidden dimensions:
        # - sequence_state_dim (1024): ESM-2 trunk / s_s embeddings
        # - structure_module.sequence_dim (384): folding trunk states
        trunk_config = model.config.esmfold_config.trunk
        structure_config = trunk_config.structure_module  # Nested inside trunk
        self._embedding_sizes = {
            # states comes from structure_module, uses its sequence_dim
            'states': structure_config.sequence_dim,  # 384 for ESMFold v1
            's_z': trunk_config.pairwise_state_dim,   # 128 for pairwise
            's_s': trunk_config.sequence_state_dim,   # 1024 for ESM-2 stem
            # Combined = states + s_z concatenated
            'combined': (
                structure_config.sequence_dim + trunk_config.pairwise_state_dim
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
    def tokenizer(self) -> EsmFoldTokenizerWrapper:
        """Get the tokenizer of the encoder (wrapped to disable special tokens)."""
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

        Raises
        ------
        ValueError
            If any sequence is shorter than min_length.
        RuntimeError
            If ESMFold's pTM computation fails (usually due to numerical
            instability with half precision or edge case sequences).
        """
        # Move inputs to device
        input_ids = batch_encoding['input_ids'].to(self.device)
        attention_mask = batch_encoding['attention_mask'].to(self.device)

        # Check minimum sequence length (ESMFold pTM can fail on short seqs)
        seq_lengths = attention_mask.sum(dim=1)
        min_seq_len = seq_lengths.min().item()
        if min_seq_len < self._min_length:
            raise ValueError(
                f'Sequence length {min_seq_len} is below minimum '
                f'{self._min_length}. ESMFold pTM computation may fail on '
                f'very short sequences. Filter out sequences shorter than '
                f'{self._min_length} residues or set min_length in config.',
            )

        # Forward pass through ESMFold (includes structure prediction)
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        except IndexError as e:
            # Known issue: ESMFold's compute_tm can fail with numerical
            # instability, especially with half precision
            if 'index 0 is out of bounds' in str(e):
                raise RuntimeError(
                    'ESMFold pTM computation failed. This is often caused by '
                    'numerical instability with half precision. Try setting '
                    'half_precision: false in encoder_config, or check for '
                    'problematic sequences (very short, unusual amino acids).',
                ) from e
            raise

        # Extract embeddings from the FINAL recycle iteration (best quality)
        # states has shape [num_recycles, batch, seq_len, hidden_dim]
        # We want the last recycle: states[-1] -> [batch, seq_len, hidden_dim]
        batch_size = attention_mask.shape[0]
        final_states = self._extract_final_recycle_states(
            outputs.states,
            batch_size,
            )

        # Multi-representation mode: return both as dict
        if self._multi_representation:
            structure_emb = final_states  # [B, L, 384]
            pairwise_emb = outputs.s_z.mean(dim=2)  # [B, L, 128]

            # Check for NaN values (can happen with FP16)
            self._check_nan(structure_emb, 'structure')
            self._check_nan(pairwise_emb, 'pairwise')

            return {
                'structure': structure_emb,
                'pairwise': pairwise_emb,
            }

        # Single representation mode
        if self._representation == 'states':
            # Hidden states from the FOLDING TRUNK (structure module)
            # These encode 3D structural information after iterative refinement
            # Shape: [batch, seq_len, 384] (structure_module.sequence_dim)
            embeddings = final_states
        elif self._representation == 's_z':
            # Pairwise residue embeddings (structural relationships)
            # Shape: [batch, seq_len, seq_len, pairwise_state_dim]
            # Mean-pool over one dimension to get per-residue embeddings
            # -> [batch, seq_len, pairwise_state_dim]
            embeddings = outputs.s_z.mean(dim=2)
        elif self._representation == 'combined':
            # BOTH structure representations concatenated
            # states: [batch, seq_len, 384]
            # s_z pooled: [batch, seq_len, 128]
            # -> [batch, seq_len, 512]
            s_z_pooled = outputs.s_z.mean(dim=2)
            embeddings = torch.cat([final_states, s_z_pooled], dim=-1)
        elif self._representation == 's_s':
            # ESM-2 stem embeddings (sequence-based, NOT structure)
            # Only use if you specifically want sequence info
            # Shape: [batch, seq_len, 1024]
            embeddings = outputs.s_s
        else:
            raise ValueError(
                f'Unknown representation: {self._representation}. '
                f'Expected "states", "s_z", "combined", or "s_s".',
            )

        # Check for NaN values (can happen with FP16)
        self._check_nan(embeddings, self._representation)

        return embeddings

    def _check_nan(self, tensor: torch.Tensor, name: str) -> None:
        """Check for NaN values in tensor and log a warning if found.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to check.
        name : str
            Name of the tensor for logging.
        """
        import sys

        if torch.isnan(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            total = tensor.numel()
            print(
                f'[WARNING] ESMFold {name} embeddings contain '
                f'{nan_count}/{total} NaN values. '
                f'This is often caused by FP16 numerical instability. '
                f'Consider using half_precision: false.',
                file=sys.stderr,
            )

    def _extract_final_recycle_states(
        self,
        states: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Extract the final recycle's states from the ESMFold output.

        ESMFold outputs states from ALL recycling iterations. The states
        tensor has shape [num_recycles, batch, seq_len, hidden_dim] or may
        be reshaped to [num_recycles * batch, seq_len, hidden_dim].

        We extract only the final recycle (index -1) which represents the
        most refined structure prediction.

        Parameters
        ----------
        states : torch.Tensor
            The states tensor from ESMFold outputs.
        batch_size : int
            The actual batch size (from attention_mask).

        Returns
        -------
        torch.Tensor
            States from the final recycle iteration.
            Shape: [batch_size, seq_len, hidden_dim]
        """
        # Handle different possible shapes from ESMFold
        if states.dim() == 4:
            # Shape: [num_recycles, batch, seq_len, hidden_dim]
            # Take the last recycle
            return states[-1]
        elif states.dim() == 3:
            # Shape: [num_recycles * batch, seq_len, hidden_dim]
            # Need to reshape and extract final recycle
            total_samples = states.shape[0]
            num_recycles = total_samples // batch_size

            if num_recycles * batch_size != total_samples:
                # If it doesn't divide evenly, states might already be
                # just the final recycle or have unexpected format
                if total_samples == batch_size:
                    # Already just one recycle's worth
                    return states
                raise ValueError(
                    f'Unexpected states shape {states.shape}. Expected '
                    f'[num_recycles * batch, seq_len, hidden] where '
                    f'batch={batch_size}, but got {total_samples} samples.',
                )

            # Reshape to [num_recycles, batch, seq_len, hidden_dim]
            seq_len = states.shape[1]
            hidden_dim = states.shape[2]
            states_reshaped = states.view(
                num_recycles,
                batch_size,
                seq_len,
                hidden_dim,
            )
            # Take the final recycle
            return states_reshaped[-1]
        else:
            raise ValueError(
                f'Unexpected states tensor with {states.dim()} dimensions. '
                f'Expected 3 or 4 dimensions, got shape {states.shape}.',
            )
