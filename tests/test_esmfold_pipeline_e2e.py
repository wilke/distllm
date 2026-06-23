"""End-to-end tests for ESMFold embedding pipeline.

These tests verify the ENTIRE pipeline from encoding to writing output,
ensuring no NaN values are produced at any stage.

Run with: pytest tests/test_esmfold_pipeline_e2e.py -v -s
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import load_from_disk

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA not available - ESMFold tests require GPU',
)


# Test sequences of varying lengths
TEST_SEQUENCES = [
    {
        'primary_accession': 'TEST_SHORT',
        'sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH',
        'pmids': '12345',
        'path': 'test.pdf',
    },
    {
        'primary_accession': 'TEST_MEDIUM', 
        'sequence': 'MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH'
                    'GSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLL'
                    'SHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR',
        'pmids': '67890',
        'path': 'test2.pdf',
    },
    {
        'primary_accession': 'TEST_LONG',
        'sequence': 'A' * 200,
        'pmids': None,
        'path': 'test3.pdf',
    },
]


@pytest.fixture(scope='module')
def esmfold_encoder():
    """Load ESMFold encoder once for all tests."""
    from distllm.embed.encoders.esmfold import EsmFoldEncoder, EsmFoldEncoderConfig
    
    config = EsmFoldEncoderConfig(
        pretrained_model_name_or_path='facebook/esmfold_v1',
        half_precision=True,  # Should use BF16 now
        max_length=512,
        num_recycles=2,
        multi_representation=True,  # Test multi-representation mode
    )
    encoder = EsmFoldEncoder(config)
    return encoder


@pytest.fixture
def mean_pooler():
    """Create mean pooler."""
    from distllm.embed.poolers.mean import MeanPooler, MeanPoolerConfig
    
    config = MeanPoolerConfig(exclude_special_tokens=False)
    return MeanPooler(config)


@pytest.fixture
def test_jsonl_file(tmp_path):
    """Create a temporary JSONL file with test sequences."""
    file_path = tmp_path / 'test_sequences.jsonl'
    with open(file_path, 'w') as f:
        for seq in TEST_SEQUENCES:
            item = {**seq, 'text': seq['sequence']}
            f.write(json.dumps(item) + '\n')
    return file_path


class TestEncoderDtype:
    """Test that encoder uses correct dtype (BF16, not FP16)."""
    
    def test_encoder_uses_bfloat16(self, esmfold_encoder):
        """Verify encoder model is in bfloat16, not float16."""
        assert esmfold_encoder.dtype == torch.bfloat16, (
            f'Expected bfloat16, got {esmfold_encoder.dtype}. '
            f'FP16 produces 100% NaN with ESMFold!'
        )
    
    def test_encoder_on_cuda(self, esmfold_encoder):
        """Verify encoder is on CUDA."""
        assert esmfold_encoder.device.type == 'cuda'


class TestEncoderOutput:
    """Test that encoder produces valid (non-NaN) embeddings."""
    
    def test_single_sequence_no_nan(self, esmfold_encoder):
        """Test single sequence produces no NaN."""
        seq = TEST_SEQUENCES[0]['sequence']
        
        tokenized = esmfold_encoder.tokenizer(
            [seq],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(esmfold_encoder.device) for k, v in tokenized.items()}
        
        # Encode
        result = esmfold_encoder.encode(tokenized)
        
        # Should return dict in multi_representation mode
        assert isinstance(result, dict), f'Expected dict, got {type(result)}'
        assert 'structure' in result
        assert 'pairwise' in result
        
        # Check for NaN
        for name, emb in result.items():
            nan_count = torch.isnan(emb).sum().item()
            total = emb.numel()
            assert nan_count == 0, (
                f'{name} embeddings have {nan_count}/{total} NaN values '
                f'({100*nan_count/total:.1f}%)'
            )
    
    def test_batch_sequences_no_nan(self, esmfold_encoder):
        """Test batch of sequences produces no NaN."""
        sequences = [s['sequence'] for s in TEST_SEQUENCES]
        
        tokenized = esmfold_encoder.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(esmfold_encoder.device) for k, v in tokenized.items()}
        
        result = esmfold_encoder.encode(tokenized)
        
        for name, emb in result.items():
            nan_count = torch.isnan(emb).sum().item()
            total = emb.numel()
            assert nan_count == 0, (
                f'Batch {name} embeddings have {nan_count}/{total} NaN values'
            )
    
    def test_embedding_shapes(self, esmfold_encoder):
        """Test embedding shapes are correct."""
        seq = TEST_SEQUENCES[0]['sequence']
        
        tokenized = esmfold_encoder.tokenizer(
            [seq],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(esmfold_encoder.device) for k, v in tokenized.items()}
        
        result = esmfold_encoder.encode(tokenized)
        
        # structure should be [batch, seq_len, 384]
        assert result['structure'].shape[0] == 1
        assert result['structure'].shape[2] == 384
        
        # pairwise should be [batch, seq_len, 128]
        assert result['pairwise'].shape[0] == 1
        assert result['pairwise'].shape[2] == 128


class TestPoolerOutput:
    """Test that pooler produces valid output."""
    
    def test_pooler_no_nan_with_bf16(self, esmfold_encoder, mean_pooler):
        """Test pooler doesn't introduce NaN with BF16 input."""
        seq = TEST_SEQUENCES[0]['sequence']
        
        tokenized = esmfold_encoder.tokenizer(
            [seq],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(esmfold_encoder.device) for k, v in tokenized.items()}
        
        result = esmfold_encoder.encode(tokenized)
        
        for name, emb in result.items():
            pooled = mean_pooler.pool(emb, tokenized['attention_mask'])
            
            nan_count = torch.isnan(pooled).sum().item()
            assert nan_count == 0, (
                f'Pooled {name} has {nan_count} NaN values'
            )
            
            # Check shape is [batch, hidden_dim]
            assert pooled.dim() == 2
            assert pooled.shape[0] == 1


class TestEmbedderOutput:
    """Test the full embedder (compute_multi_embeddings)."""
    
    def test_compute_multi_embeddings_no_nan(
        self, esmfold_encoder, mean_pooler, test_jsonl_file,
    ):
        """Test compute_multi_embeddings produces no NaN."""
        from distllm.embed.datasets.jsonl import JsonlDataset, JsonlDatasetConfig
        from distllm.embed.embedders.full_sequence import compute_multi_embeddings
        
        # Create dataset
        dataset_config = JsonlDatasetConfig(
            text_field='sequence',
            preserve_metadata=True,
            batch_size=2,
            num_data_workers=0,
            min_sequence_length=10,
        )
        dataset = JsonlDataset(dataset_config)
        dataloader = dataset.get_dataloader(test_jsonl_file, esmfold_encoder)
        
        # Compute embeddings
        result = compute_multi_embeddings(
            dataloader=dataloader,
            encoder=esmfold_encoder,
            pooler=mean_pooler,
            normalize=False,
        )
        
        # Check result is dict of numpy arrays
        assert isinstance(result, dict)
        assert 'structure' in result
        assert 'pairwise' in result
        
        for name, arr in result.items():
            assert isinstance(arr, np.ndarray), f'{name} should be numpy array'
            
            nan_count = np.isnan(arr).sum()
            total = arr.size
            assert nan_count == 0, (
                f'{name} embeddings have {nan_count}/{total} NaN values '
                f'({100*nan_count/total:.1f}%)'
            )
            
            # Check values are reasonable (not all zeros, not all same)
            assert not np.allclose(arr, 0), f'{name} embeddings are all zeros'
            assert arr.std() > 0.01, f'{name} embeddings have no variance'
    
    def test_embeddings_dtype_after_conversion(
        self, esmfold_encoder, mean_pooler, test_jsonl_file,
    ):
        """Test embeddings are float32 after BF16->numpy conversion."""
        from distllm.embed.datasets.jsonl import JsonlDataset, JsonlDatasetConfig
        from distllm.embed.embedders.full_sequence import compute_multi_embeddings
        
        dataset_config = JsonlDatasetConfig(
            text_field='sequence',
            preserve_metadata=True,
            batch_size=2,
            num_data_workers=0,
            min_sequence_length=10,
        )
        dataset = JsonlDataset(dataset_config)
        dataloader = dataset.get_dataloader(test_jsonl_file, esmfold_encoder)
        
        result = compute_multi_embeddings(
            dataloader=dataloader,
            encoder=esmfold_encoder,
            pooler=mean_pooler,
            normalize=False,
        )
        
        for name, arr in result.items():
            # BF16 should be converted to float32 for numpy
            assert arr.dtype == np.float32, (
                f'{name} dtype is {arr.dtype}, expected float32'
            )


class TestWriterOutput:
    """Test the HuggingFace writer produces valid output files."""
    
    def test_full_pipeline_write_and_load(
        self, esmfold_encoder, mean_pooler, test_jsonl_file, tmp_path,
    ):
        """Test full pipeline: encode -> pool -> write -> load -> verify."""
        from distllm.embed.datasets.jsonl import JsonlDataset, JsonlDatasetConfig
        from distllm.embed.embedders.full_sequence import FullSequenceEmbedder, FullSequenceEmbedderConfig
        from distllm.embed.writers.huggingface import HuggingFaceWriter, HuggingFaceWriterConfig
        
        # Setup
        dataset_config = JsonlDatasetConfig(
            text_field='sequence',
            preserve_metadata=True,
            batch_size=2,
            num_data_workers=0,
            min_sequence_length=10,
        )
        dataset = JsonlDataset(dataset_config)
        dataloader = dataset.get_dataloader(test_jsonl_file, esmfold_encoder)
        
        embedder_config = FullSequenceEmbedderConfig(normalize_embeddings=False)
        embedder = FullSequenceEmbedder(embedder_config)
        
        writer_config = HuggingFaceWriterConfig()
        writer = HuggingFaceWriter(writer_config)
        
        # Embed
        result = embedder.embed(dataloader, esmfold_encoder, mean_pooler)
        
        # Write
        output_dir = tmp_path / 'embeddings'
        writer.write(output_dir, result)
        
        # Load and verify
        loaded_ds = load_from_disk(str(output_dir))
        
        assert len(loaded_ds) == len(TEST_SEQUENCES)
        
        # Check each row
        for i in range(len(loaded_ds)):
            row = loaded_ds[i]
            
            # Check embeddings exist and are not NaN
            emb = np.array(row['embeddings'])
            nan_count = np.isnan(emb).sum()
            assert nan_count == 0, (
                f'Row {i} embeddings have {nan_count}/{len(emb)} NaN values'
            )
            
            # Check named embeddings
            if 'embeddings_structure' in row:
                emb_struct = np.array(row['embeddings_structure'])
                nan_count = np.isnan(emb_struct).sum()
                assert nan_count == 0, (
                    f'Row {i} structure embeddings have {nan_count} NaN values'
                )
            
            if 'embeddings_pairwise' in row:
                emb_pair = np.array(row['embeddings_pairwise'])
                nan_count = np.isnan(emb_pair).sum()
                assert nan_count == 0, (
                    f'Row {i} pairwise embeddings have {nan_count} NaN values'
                )
            
            # Check metadata preserved
            assert 'primary_accession' in row
            assert 'text' in row
    
    def test_saved_dtype_is_float32(
        self, esmfold_encoder, mean_pooler, test_jsonl_file, tmp_path,
    ):
        """Test that saved embeddings are float32, not float16 or bf16."""
        from distllm.embed.datasets.jsonl import JsonlDataset, JsonlDatasetConfig
        from distllm.embed.embedders.full_sequence import FullSequenceEmbedder, FullSequenceEmbedderConfig
        from distllm.embed.writers.huggingface import HuggingFaceWriter, HuggingFaceWriterConfig
        
        dataset_config = JsonlDatasetConfig(
            text_field='sequence',
            preserve_metadata=True,
            batch_size=2,
            num_data_workers=0,
            min_sequence_length=10,
        )
        dataset = JsonlDataset(dataset_config)
        dataloader = dataset.get_dataloader(test_jsonl_file, esmfold_encoder)
        
        embedder = FullSequenceEmbedder(FullSequenceEmbedderConfig())
        writer = HuggingFaceWriter(HuggingFaceWriterConfig())
        
        result = embedder.embed(dataloader, esmfold_encoder, mean_pooler)
        
        output_dir = tmp_path / 'embeddings_dtype_test'
        writer.write(output_dir, result)
        
        loaded_ds = load_from_disk(str(output_dir))
        
        # Check the feature types
        for col_name in ['embeddings', 'embeddings_structure', 'embeddings_pairwise']:
            if col_name in loaded_ds.features:
                feature = loaded_ds.features[col_name]
                # Feature should be Sequence of float32
                dtype_str = str(feature)
                assert 'float16' not in dtype_str.lower(), (
                    f'{col_name} saved as float16! This indicates BF16 conversion failed. '
                    f'Feature: {feature}'
                )


class TestEdgeCases:
    """Test edge cases that might cause NaN."""
    
    def test_poly_alanine_sequence(self, esmfold_encoder, mean_pooler):
        """Test poly-A sequence (simple, might trigger edge cases)."""
        seq = 'A' * 50
        
        tokenized = esmfold_encoder.tokenizer(
            [seq],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(esmfold_encoder.device) for k, v in tokenized.items()}
        
        result = esmfold_encoder.encode(tokenized)
        
        for name, emb in result.items():
            pooled = mean_pooler.pool(emb, tokenized['attention_mask'])
            nan_count = torch.isnan(pooled).sum().item()
            assert nan_count == 0, f'Poly-A {name} has NaN'
    
    def test_minimum_length_sequence(self, esmfold_encoder, mean_pooler):
        """Test sequence at minimum length threshold."""
        seq = 'MVLSPADKTN'  # 10 residues (minimum)
        
        tokenized = esmfold_encoder.tokenizer(
            [seq],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(esmfold_encoder.device) for k, v in tokenized.items()}
        
        result = esmfold_encoder.encode(tokenized)
        
        for name, emb in result.items():
            pooled = mean_pooler.pool(emb, tokenized['attention_mask'])
            nan_count = torch.isnan(pooled).sum().item()
            assert nan_count == 0, f'Min-length {name} has NaN'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
