"""
Multi-Reference Dataset Wrapper for Brain-to-Text Training

Wraps the existing BrainToTextDataset to add multi-reference phoneme
targets for each trial, enabling multi-reference CTC training.

Author: Generated for Utah Brain-to-Text project
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import numpy as np

from multi_pronunciation_lexicon import (
    MultiPronunciationLexicon,
    PHONE_TO_IDX
)


class MultiReferenceWrapper:
    """
    Wraps batch data to add multi-reference phoneme targets.
    
    This can be used with the existing BrainToTextDataset by processing
    batches after they are loaded.
    """
    
    def __init__(self,
                 merger_names: Optional[List[str]] = None,
                 max_variants: int = 4):
        """
        Args:
            merger_names: List of dialect mergers to use (default: ['COT_CAUGHT'])
            max_variants: Maximum variants per trial
        """
        self.lexicon = MultiPronunciationLexicon(
            merger_names=merger_names,
            max_variants_per_sentence=max_variants
        )
        self._cache = {}
    
    def _transcription_to_text(self, transcription_tensor: torch.Tensor) -> str:
        """Convert transcription tensor (char codes) to string."""
        chars = []
        for c in transcription_tensor:
            c_int = int(c.item()) if hasattr(c, 'item') else int(c)
            if c_int != 0:
                chars.append(chr(c_int))
        return ''.join(chars)
    
    def _get_variants(self, sentence: str) -> Dict:
        """Get pronunciation variants for a sentence (with caching)."""
        if sentence in self._cache:
            return self._cache[sentence]
        
        result = self.lexicon.get_sentence_variants(sentence)
        self._cache[sentence] = result
        return result
    
    def add_multiref_targets(self, batch: Dict) -> Dict:
        """
        Add multi-reference targets to a batch.
        
        Args:
            batch: Batch dictionary from BrainToTextDataset
            
        Returns:
            Modified batch with added fields:
                - 'seq_class_ids_variants': List of variant tensors per sample
                - 'variant_lengths': List of variant lengths per sample
        """
        batch_size = batch['transcriptions'].shape[0]
        
        seq_class_ids_variants = []
        variant_lengths = []
        
        for i in range(batch_size):
            # Get transcription text
            sentence = self._transcription_to_text(batch['transcriptions'][i])
            
            # Get pronunciation variants
            variant_data = self._get_variants(sentence)
            
            # Convert to tensors
            variant_tensors = [
                torch.tensor(ids, dtype=torch.long)
                for ids in variant_data['variant_ids']
            ]
            lengths = [len(v) for v in variant_data['variant_ids']]
            
            seq_class_ids_variants.append(variant_tensors)
            variant_lengths.append(lengths)
        
        batch['seq_class_ids_variants'] = seq_class_ids_variants
        batch['variant_lengths'] = variant_lengths
        
        return batch


class MultiReferenceDataset(Dataset):
    """
    Dataset that wraps BrainToTextDataset to add multi-reference targets.
    
    Usage:
        base_dataset = BrainToTextDataset(...)
        multiref_dataset = MultiReferenceDataset(base_dataset, merger_names=['COT_CAUGHT'])
    """
    
    def __init__(self,
                 base_dataset: Dataset,
                 merger_names: Optional[List[str]] = None,
                 max_variants: int = 4):
        """
        Args:
            base_dataset: The underlying BrainToTextDataset
            merger_names: List of dialect mergers to use
            max_variants: Maximum pronunciation variants per trial
        """
        self.base_dataset = base_dataset
        self.wrapper = MultiReferenceWrapper(
            merger_names=merger_names,
            max_variants=max_variants
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        batch = self.base_dataset[idx]
        return self.wrapper.add_multiref_targets(batch)
