"""Encoder for NVIDIA NV-Embed models.

NV-Embed models use a latent attention pooling mechanism that is trained
as part of the model. This encoder uses the model's built-in encode()
method to get properly pooled embeddings.

Reference: https://huggingface.co/nvidia/NV-Embed-v2
Paper: https://arxiv.org/abs/2405.17428

Note: NV-Embed requires specific package versions for compatibility.
Install with: pip install distllm[nvembed]

Or manually:
    pip install transformers==4.42.4 accelerate==0.30.0 einops
"""

from __future__ import annotations

from typing import Literal
from typing import Optional

import torch
from transformers import BatchEncoding
from transformers import PreTrainedTokenizer

from distllm.utils import BaseConfig


class NVEmbedEncoderConfig(BaseConfig):
    """Config for the NV-Embed encoder."""

    # The name of the encoder
    name: Literal['nvembed'] = 'nvembed'  # type: ignore[assignment]
    # The model id (e.g., 'nvidia/NV-Embed-v2')
    pretrained_model_name_or_path: str = 'nvidia/NV-Embed-v2'
    # Optional instruction prefix for retrieval tasks
    # See: https://huggingface.co/nvidia/NV-Embed-v2#1-instruction-template-for-mteb-benchmarks
    instruction: str = ''
    # Maximum sequence length
    max_length: int = 32768
    # Use the model in half precision
    half_precision: bool = False
    # Set the model to evaluation mode
    eval_mode: bool = True
    # Use quantization (4-bit)
    quantization: bool = True


class NVEmbedEncoder:
    """Encoder for NVIDIA NV-Embed models.

    This encoder wraps NV-Embed's built-in encode() method which includes
    the trained latent attention pooling layer. The output embeddings are
    already pooled.

    Note: This encoder returns embeddings of shape [B, 1, D] to maintain
    compatibility with the pooler interface. Use the IdentityPooler to
    squeeze out the fake sequence dimension.
    """

    def __init__(self, config: NVEmbedEncoderConfig) -> None:
        """Initialize the encoder."""
        from transformers import AutoModel
        from transformers import AutoTokenizer

        model_kwargs = {}

        # Use quantization
        if config.quantization:
            from transformers import BitsAndBytesConfig

            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs['quantization_config'] = nf4_config
            # Use device_map for proper quantized model loading
            model_kwargs['device_map'] = 'auto'

        # Load model with trust_remote_code (required for NV-Embed)
        model = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

        # Set the model max length for proper truncation
        tokenizer.model_max_length = config.max_length

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        if config.eval_mode:
            model.eval()

        # Load the model onto the device (if not using quantization)
        if not config.quantization:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu',
            )
            model.to(device)

        # Store config and model
        self.config = config
        self.model = model
        self._tokenizer = tokenizer

    @property
    def dtype(self) -> torch.dtype:
        """Get the data type of the encoder."""
        return self.model.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
        return self.model.device

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the encoder.

        NV-Embed-v2 has 4096-dimensional embeddings.
        """
        return self.model.config.hidden_size

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer of the encoder."""
        return self._tokenizer

    def encode(self, batch_encoding: BatchEncoding) -> torch.Tensor:
        """Encode the sequence using NV-Embed's built-in pooling.

        Parameters
        ----------
        batch_encoding : BatchEncoding
            The batch encoding of the sequence. Note: We decode this back
            to text since NV-Embed's encode() method expects raw strings.

        Returns
        -------
        torch.Tensor
            The pooled embeddings with a fake sequence dimension.
            (shape: [num_sequences, 1, embedding_size])

        Note
        ----
        The output has shape [B, 1, D] instead of [B, D] to maintain
        compatibility with the pooler interface. Use IdentityPooler
        to squeeze out the fake sequence dimension.
        """
        # Decode token IDs back to text
        # NV-Embed's encode() method expects raw text strings
        texts = self._tokenizer.batch_decode(
            batch_encoding['input_ids'],
            skip_special_tokens=True,
        )

        # Use NV-Embed's built-in encode method which includes
        # the trained latent attention pooling
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                instruction=self.config.instruction,
                max_length=self.config.max_length,
            )

        # Ensure embeddings are on the correct device and dtype
        if isinstance(embeddings, torch.Tensor):
            pooled = embeddings
        else:
            # Convert numpy array to tensor if needed
            pooled = torch.from_numpy(embeddings)

        # Move to model device if needed
        pooled = pooled.to(device=self.device, dtype=self.dtype)

        # Add fake sequence dimension for pooler compatibility: [B, D] -> [B, 1, D]
        return pooled.unsqueeze(1)
