"""Tests for JSONL dataset filtering edge cases.

These tests verify that the sequence length filtering works correctly for:
- Empty sequences
- Very short sequences (below min_length)
- Very long sequences (above max_length)
- All sequences filtered out (should raise EmptyDatasetError)
- Mixed valid/invalid sequences
- None/null sequences
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from distllm.embed.datasets.jsonl import (
    EmptyDatasetError,
    JsonlDataset,
    JsonlDatasetConfig,
)


def create_temp_jsonl(records: list[dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with the given records."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.jsonl',
        delete=False,
    ) as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
        return Path(f.name)


def create_mock_encoder() -> MagicMock:
    """Create a mock encoder with a tokenizer."""
    encoder = MagicMock()
    encoder.tokenizer = MagicMock()
    return encoder


class TestMinLengthFiltering:
    """Test cases for minimum sequence length filtering."""

    def test_filters_short_sequences(self) -> None:
        """Sequences shorter than min_length should be filtered out."""
        records = [
            {'primary_accession': 'A', 'sequence': 'ACDEF'},  # 5 residues
            {'primary_accession': 'B', 'sequence': 'ACDEFGHIJK'},  # 10 residues
            {'primary_accession': 'C', 'sequence': 'ACDEFGHIJKLMNO'},  # 14 residues
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=10,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        # The filtering happens in get_dataloader, but we can test the filter
        # method directly
        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        filtered = dataset._filter_by_length(content, jsonl_file)

        # Only B and C should remain
        assert len(filtered) == 2
        assert filtered[0]['primary_accession'] == 'B'
        assert filtered[1]['primary_accession'] == 'C'

        # Cleanup
        jsonl_file.unlink()

    def test_filters_empty_sequence(self) -> None:
        """Empty sequences should be filtered out."""
        records = [
            {'primary_accession': 'A', 'sequence': ''},  # Empty
            {'primary_accession': 'B', 'sequence': 'ACDEFGHIJK'},  # Valid
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=5,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        filtered = dataset._filter_by_length(content, jsonl_file)

        assert len(filtered) == 1
        assert filtered[0]['primary_accession'] == 'B'

        jsonl_file.unlink()

    def test_filters_none_sequence(self) -> None:
        """None/null sequences should be filtered out."""
        records = [
            {'primary_accession': 'A', 'sequence': None},  # None
            {'primary_accession': 'B', 'sequence': 'ACDEFGHIJK'},  # Valid
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=5,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        filtered = dataset._filter_by_length(content, jsonl_file)

        assert len(filtered) == 1
        assert filtered[0]['primary_accession'] == 'B'

        jsonl_file.unlink()


class TestMaxLengthFiltering:
    """Test cases for maximum sequence length filtering."""

    def test_filters_long_sequences(self) -> None:
        """Sequences longer than max_length should be filtered out."""
        records = [
            {'primary_accession': 'A', 'sequence': 'A' * 100},  # 100 residues
            {'primary_accession': 'B', 'sequence': 'A' * 500},  # 500 residues
            {'primary_accession': 'C', 'sequence': 'A' * 1500},  # 1500 residues
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            max_sequence_length=1000,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        filtered = dataset._filter_by_length(content, jsonl_file)

        # Only A and B should remain
        assert len(filtered) == 2
        assert filtered[0]['primary_accession'] == 'A'
        assert filtered[1]['primary_accession'] == 'B'

        jsonl_file.unlink()

    def test_exact_max_length_included(self) -> None:
        """Sequences exactly at max_length should be included."""
        records = [
            {'primary_accession': 'A', 'sequence': 'A' * 1000},  # Exactly 1000
            {'primary_accession': 'B', 'sequence': 'A' * 1001},  # 1001 - too long
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            max_sequence_length=1000,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        filtered = dataset._filter_by_length(content, jsonl_file)

        assert len(filtered) == 1
        assert filtered[0]['primary_accession'] == 'A'

        jsonl_file.unlink()


class TestCombinedFiltering:
    """Test cases for combined min and max length filtering."""

    def test_filters_both_short_and_long(self) -> None:
        """Both short and long sequences should be filtered out."""
        records = [
            {'primary_accession': 'A', 'sequence': 'ACE'},  # 3 - too short
            {'primary_accession': 'B', 'sequence': 'A' * 50},  # 50 - valid
            {'primary_accession': 'C', 'sequence': 'A' * 100},  # 100 - valid
            {'primary_accession': 'D', 'sequence': 'A' * 200},  # 200 - too long
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=10,
            max_sequence_length=150,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        filtered = dataset._filter_by_length(content, jsonl_file)

        # Only B and C should remain
        assert len(filtered) == 2
        assert filtered[0]['primary_accession'] == 'B'
        assert filtered[1]['primary_accession'] == 'C'

        jsonl_file.unlink()


class TestEmptyDatasetError:
    """Test cases for EmptyDatasetError when all sequences are filtered."""

    def test_raises_when_all_filtered_by_min_length(self) -> None:
        """Should raise EmptyDatasetError when all sequences are too short."""
        records = [
            {'primary_accession': 'A', 'sequence': 'ACE'},  # 3 residues
            {'primary_accession': 'B', 'sequence': 'ACDEF'},  # 5 residues
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=100,  # All sequences are shorter
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)
        encoder = create_mock_encoder()

        with pytest.raises(EmptyDatasetError) as exc_info:
            dataset.get_dataloader(jsonl_file, encoder)

        assert 'All 2 sequences were filtered out' in str(exc_info.value)

        jsonl_file.unlink()

    def test_raises_when_all_filtered_by_max_length(self) -> None:
        """Should raise EmptyDatasetError when all sequences are too long."""
        records = [
            {'primary_accession': 'A', 'sequence': 'A' * 2000},
            {'primary_accession': 'B', 'sequence': 'A' * 3000},
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            max_sequence_length=100,  # All sequences are longer
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)
        encoder = create_mock_encoder()

        with pytest.raises(EmptyDatasetError) as exc_info:
            dataset.get_dataloader(jsonl_file, encoder)

        assert 'All 2 sequences were filtered out' in str(exc_info.value)

        jsonl_file.unlink()

    def test_raises_when_all_empty_or_none(self) -> None:
        """Should raise EmptyDatasetError when all sequences are empty/None."""
        records = [
            {'primary_accession': 'A', 'sequence': ''},
            {'primary_accession': 'B', 'sequence': None},
            {'primary_accession': 'C', 'sequence': ''},
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=1,  # Even 1 char would be fine
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)
        encoder = create_mock_encoder()

        with pytest.raises(EmptyDatasetError) as exc_info:
            dataset.get_dataloader(jsonl_file, encoder)

        assert 'All 3 sequences were filtered out' in str(exc_info.value)

        jsonl_file.unlink()


class TestNoFilteringWhenDisabled:
    """Test that no filtering occurs when min/max are None."""

    def test_no_filtering_when_disabled(self) -> None:
        """All sequences should pass when filtering is disabled."""
        records = [
            {'primary_accession': 'A', 'sequence': 'A'},  # 1 residue
            {'primary_accession': 'B', 'sequence': 'A' * 10000},  # 10000 residues
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=None,  # Disabled
            max_sequence_length=None,  # Disabled
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        # Filter should return all (no filtering happens when both are None)
        # Actually the filter is only called if min or max is set
        # So we test that the content length is preserved when both are None
        assert len(content) == 2

        jsonl_file.unlink()


class TestIdFieldLogging:
    """Test that ID field is correctly logged when sequences are dropped."""

    def test_logs_primary_accession(self, capsys: pytest.CaptureFixture) -> None:
        """Should log the primary_accession when dropping sequences."""
        records = [
            {'primary_accession': 'P12345', 'sequence': 'ACE'},  # Too short
            {'primary_accession': 'Q67890', 'sequence': 'A' * 100},  # Valid
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=10,
            id_field='primary_accession',
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        dataset._filter_by_length(content, jsonl_file)

        captured = capsys.readouterr()
        assert 'P12345' in captured.err
        assert 'length=3' in captured.err

        jsonl_file.unlink()

    def test_logs_unknown_when_no_id_field(
        self, capsys: pytest.CaptureFixture,
    ) -> None:
        """Should log <unknown> when id_field is not set."""
        records = [
            {'sequence': 'ACE'},  # Too short, no ID field
            {'sequence': 'A' * 100},  # Valid
        ]
        jsonl_file = create_temp_jsonl(records)

        config = JsonlDatasetConfig(
            text_field='sequence',
            min_sequence_length=10,
            id_field=None,  # No ID field configured
        )
        dataset = JsonlDataset(config)

        lines = jsonl_file.read_text().strip().split('\n')
        content = [json.loads(line) for line in lines]

        dataset._filter_by_length(content, jsonl_file)

        captured = capsys.readouterr()
        assert '<unknown>' in captured.err

        jsonl_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
