"""Single sequence per line file dataset."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader

from distllm.embed.datasets.utils import DataCollator
from distllm.embed.datasets.utils import InMemoryDataset
from distllm.embed.encoders.base import Encoder
from distllm.utils import BaseConfig


class EmptyDatasetError(Exception):
    """Raised when all sequences are filtered out from a dataset."""

    pass


class JsonlDatasetConfig(BaseConfig):
    """Configuration for the JsonlDataset."""

    # The name of the dataset
    name: Literal['jsonl'] = 'jsonl'  # type: ignore[assignment]

    # The name of the text field in the jsonl file
    text_field: str = 'text'
    # Whether to preserve all other fields as metadata
    preserve_metadata: bool = True
    # Number of data workers for batching.
    num_data_workers: int = 4
    # Inference batch size.
    batch_size: int = 8
    # Whether to pin memory for the dataloader.
    pin_memory: bool = True

    # Minimum sequence length filter (None = no filtering)
    # Sequences shorter than this will be dropped
    min_sequence_length: int | None = None
    # Maximum sequence length filter (None = no filtering)
    # Sequences longer than this will be dropped (not truncated)
    max_sequence_length: int | None = None
    # Field name containing the ID to log when sequences are dropped
    # (e.g., 'primary_accession' for protein datasets)
    id_field: str | None = None
    # Filter out sequences containing non-standard amino acid characters
    # Standard amino acids are: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y
    # Non-standard codes (B, Z, X, U, J, O, etc.) cause tokenization errors in ESM models
    filter_non_standard_aa: bool = False


class JsonlDataset:
    """Jsonl file dataset."""

    def __init__(self, config: JsonlDatasetConfig):
        """Initialize the dataset."""
        self.config = config

    def get_dataloader(
        self,
        data_file: Path,
        encoder: Encoder,
    ) -> DataLoader:
        """Instantiate a dataloader for the dataset.

        Parameters
        ----------
        data_file : Path
            The file to read.
        encoder : Encoder
            The encoder instance.

        Returns
        -------
        DataLoader
            The dataloader instance.

        Raises
        ------
        EmptyDatasetError
            If all sequences are filtered out.
        """
        # Read the jsonl file
        lines = data_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        original_count = len(content)

        # Filter sequences by length
        if (
            self.config.min_sequence_length is not None
            or self.config.max_sequence_length is not None
        ):
            content = self._filter_by_length(content, data_file)

        # Filter sequences with non-standard amino acids
        if self.config.filter_non_standard_aa:
            content = self._filter_non_standard_aa(content, data_file)

        # Check if all sequences were filtered out
        if len(content) == 0:
            raise EmptyDatasetError(
                f'[{data_file.name}] All {original_count} sequences were '
                f'filtered out. No data to process.',
            )

        # Extract the text data
        data = [item[self.config.text_field] for item in content]

        # Extract metadata (all fields except the text field)
        metadata = None
        if self.config.preserve_metadata:
            metadata = [
                {k: v for k, v in item.items() if k != self.config.text_field}
                for item in content
            ]

        # Instantiate the dataloader
        return DataLoader(
            pin_memory=self.config.pin_memory,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_data_workers,
            dataset=InMemoryDataset(data, metadata=metadata),
            collate_fn=DataCollator(encoder.tokenizer),
        )

    def _filter_by_length(
        self,
        content: list[dict],
        data_file: Path,
    ) -> list[dict]:
        """Filter out sequences outside the min/max length bounds.

        Parameters
        ----------
        content : list[dict]
            The loaded JSONL content.
        data_file : Path
            The source file (for logging context).

        Returns
        -------
        list[dict]
            Filtered content.
        """
        min_len = self.config.min_sequence_length
        max_len = self.config.max_sequence_length
        text_field = self.config.text_field
        id_field = self.config.id_field

        filtered = []
        dropped_too_short: list[tuple[str, int]] = []
        dropped_too_long: list[tuple[str, int]] = []

        for item in content:
            seq = item[text_field]

            # Handle None or empty sequences
            if seq is None or seq == '':
                seq_id = item.get(id_field, '<unknown>') if id_field else '<unknown>'
                dropped_too_short.append((seq_id, 0))
                continue

            seq_len = len(seq)

            # Check min length
            if min_len is not None and seq_len < min_len:
                seq_id = item.get(id_field, f'<unknown>') if id_field else '<unknown>'
                dropped_too_short.append((seq_id, seq_len))
                continue

            # Check max length
            if max_len is not None and seq_len > max_len:
                seq_id = item.get(id_field, f'<unknown>') if id_field else '<unknown>'
                dropped_too_long.append((seq_id, seq_len))
                continue

            filtered.append(item)

        # Log dropped sequences
        if dropped_too_short:
            print(
                f'[{data_file.name}] Dropped {len(dropped_too_short)} sequences '
                f'shorter than {min_len} residues:',
                file=sys.stderr,
            )
            for seq_id, seq_len in dropped_too_short:
                print(f'  - {seq_id} (length={seq_len})', file=sys.stderr)

        if dropped_too_long:
            print(
                f'[{data_file.name}] Dropped {len(dropped_too_long)} sequences '
                f'longer than {max_len} residues:',
                file=sys.stderr,
            )
            for seq_id, seq_len in dropped_too_long:
                print(f'  - {seq_id} (length={seq_len})', file=sys.stderr)

        return filtered

    def _filter_non_standard_aa(
        self,
        content: list[dict],
        data_file: Path,
    ) -> list[dict]:
        """Filter out sequences containing non-standard amino acid characters.

        ESM models only support the 20 standard amino acids. Sequences containing
        non-standard codes (B, Z, X, U, J, O, etc.) will cause tokenization errors.

        Parameters
        ----------
        content : list[dict]
            The loaded JSONL content.
        data_file : Path
            The source file (for logging context).

        Returns
        -------
        list[dict]
            Filtered content containing only standard amino acid sequences.
        """
        # Standard 20 amino acids supported by ESM tokenizer
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        text_field = self.config.text_field
        id_field = self.config.id_field

        filtered = []
        dropped: list[tuple[str, set[str]]] = []

        for item in content:
            seq = item[text_field]
            if seq is None or seq == '':
                continue

            # Check for non-standard characters
            seq_chars = set(seq.upper())
            non_standard = seq_chars - standard_aa

            if non_standard:
                seq_id = item.get(id_field, '<unknown>') if id_field else '<unknown>'
                dropped.append((seq_id, non_standard))
                continue

            filtered.append(item)

        # Log dropped sequences
        if dropped:
            print(
                f'[{data_file.name}] Dropped {len(dropped)} sequences '
                f'with non-standard amino acids:',
                file=sys.stderr,
            )
            for seq_id, chars in dropped:
                print(f'  - {seq_id} (chars: {sorted(chars)})', file=sys.stderr)

        return filtered
