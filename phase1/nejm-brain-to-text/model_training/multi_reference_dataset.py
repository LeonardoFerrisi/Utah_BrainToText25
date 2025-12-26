"""
Multi-Reference Dataset for Brain-to-Text Training

This module extends the original BrainToTextDataset to support multiple
valid phoneme sequences per trial, enabling multi-reference CTC training.

Author: Generated for Utah Brain-to-Text project
"""

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple
import os

# Import the multi-pronunciation lexicon
from multi_pronunciation_lexicon import (
    MultiPronunciationLexicon, 
    PHONE_TO_IDX,
    LOGIT_PHONE_DEF
)


class MultiReferenceDataset(Dataset):
    """
    Dataset that provides multiple valid phoneme sequences for each trial.
    
    This wraps or extends the original dataset to add pronunciation variants
    based on dialect-specific phoneme equivalences.
    """
    
    def __init__(self,
                 trial_indicies: Dict,
                 split: str,
                 days_per_batch: Optional[int],
                 n_batches: Optional[int],
                 batch_size: int,
                 must_include_days: Optional[List[int]] = None,
                 random_seed: int = 42,
                 feature_subset: Optional[List[int]] = None,
                 merger_names: Optional[List[str]] = None,
                 max_variants_per_trial: int = 8):
        """
        Args:
            trial_indicies: Dictionary mapping day indices to session paths and trial lists
            split: 'train' or 'test'
            days_per_batch: Number of days to include in each batch
            n_batches: Number of training batches
            batch_size: Number of trials per batch
            must_include_days: Days that must be included in each batch
            random_seed: Random seed for reproducibility
            feature_subset: Subset of neural features to use
            merger_names: List of dialect mergers to consider
            max_variants_per_trial: Maximum pronunciation variants per trial
        """
        self.trial_indicies = trial_indicies
        self.split = split
        self.days_per_batch = days_per_batch
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.must_include_days = must_include_days
        self.random_seed = random_seed
        self.feature_subset = feature_subset
        self.max_variants = max_variants_per_trial
        
        self.n_days = len(trial_indicies)
        
        # Initialize the multi-pronunciation lexicon
        self.lexicon = MultiPronunciationLexicon(
            merger_names=merger_names,
            max_variants_per_sentence=max_variants_per_trial
        )
        
        # Random state
        self.rng = np.random.default_rng(random_seed)
        
        # Create batch index
        if split == 'train':
            self.batch_index = self._create_batch_index_train()
        else:
            self.batch_index = self._create_batch_index_test()
            self.n_batches = len(self.batch_index)
        
        # Cache for transcription -> variants mapping
        self._variant_cache = {}
    
    def __len__(self):
        return self.n_batches
    
    def _get_phoneme_variants(self, transcription_ids: np.ndarray) -> Dict:
        """
        Convert transcription character IDs back to text and generate variants.
        
        Args:
            transcription_ids: Array of character codes
            
        Returns:
            Dict with 'variants' (list of phoneme ID lists) and 'lengths' (list of lengths)
        """
        # Convert character codes to string
        chars = [chr(int(c)) for c in transcription_ids if c != 0]
        sentence = ''.join(chars)
        
        # Check cache
        if sentence in self._variant_cache:
            return self._variant_cache[sentence]
        
        # Generate variants
        result = self.lexicon.get_sentence_variants(sentence)
        
        variant_data = {
            'variants': result['variant_ids'],
            'lengths': [len(v) for v in result['variant_ids']],
            'n_variants': result['n_variants']
        }
        
        # Cache for later use
        self._variant_cache[sentence] = variant_data
        
        return variant_data
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a batch with multi-reference targets.
        
        Returns a dict containing:
            - input_features: (B, T, F) neural features
            - seq_class_ids: (B, S) canonical phoneme IDs (for compatibility)
            - seq_class_ids_variants: List of B items, each a list of variant tensors
            - variant_lengths: List of B items, each a list of variant lengths
            - n_variants: (B,) number of variants per sample
            - n_time_steps, phone_seq_lens, day_indicies, etc.
        """
        batch = {
            'input_features': [],
            'seq_class_ids': [],           # Canonical (for backward compatibility)
            'seq_class_ids_variants': [],  # Multi-reference variants
            'variant_lengths': [],
            'n_variants': [],
            'n_time_steps': [],
            'phone_seq_lens': [],          # Length of canonical sequence
            'day_indicies': [],
            'transcriptions': [],
            'block_nums': [],
            'trial_nums': [],
        }
        
        index = self.batch_index[idx]
        
        for d in index.keys():
            session_path = self.trial_indicies[d]['session_path']
            
            with h5py.File(session_path, 'r') as f:
                for t in index[d]:
                    try:
                        g = f[f'trial_{t:04d}']
                        
                        # Load neural features
                        input_features = torch.from_numpy(g['input_features'][:])
                        if self.feature_subset:
                            input_features = input_features[:, self.feature_subset]
                        batch['input_features'].append(input_features)
                        
                        # Load canonical phoneme sequence
                        canonical_ids = torch.from_numpy(g['seq_class_ids'][:])
                        batch['seq_class_ids'].append(canonical_ids)
                        
                        # Load transcription and generate variants
                        transcription = g['transcription'][:]
                        batch['transcriptions'].append(torch.from_numpy(transcription))
                        
                        # Generate phoneme variants
                        variant_data = self._get_phoneme_variants(transcription)
                        variant_tensors = [
                            torch.tensor(v, dtype=torch.long) 
                            for v in variant_data['variants']
                        ]
                        batch['seq_class_ids_variants'].append(variant_tensors)
                        batch['variant_lengths'].append(variant_data['lengths'])
                        batch['n_variants'].append(variant_data['n_variants'])
                        
                        # Other metadata
                        batch['n_time_steps'].append(g.attrs['n_time_steps'])
                        batch['phone_seq_lens'].append(g.attrs['seq_len'])
                        batch['day_indicies'].append(int(d))
                        batch['block_nums'].append(g.attrs['block_num'])
                        batch['trial_nums'].append(g.attrs['trial_num'])
                        
                    except Exception as e:
                        print(f'Error loading trial {t} from session {session_path}: {e}')
                        continue
        
        # Pad features and canonical sequences
        batch['input_features'] = pad_sequence(
            batch['input_features'], batch_first=True, padding_value=0
        )
        batch['seq_class_ids'] = pad_sequence(
            batch['seq_class_ids'], batch_first=True, padding_value=0
        )
        
        # Convert lists to tensors where appropriate
        batch['n_time_steps'] = torch.tensor(batch['n_time_steps'])
        batch['phone_seq_lens'] = torch.tensor(batch['phone_seq_lens'])
        batch['day_indicies'] = torch.tensor(batch['day_indicies'])
        batch['n_variants'] = torch.tensor(batch['n_variants'])
        batch['transcriptions'] = torch.stack(batch['transcriptions'])
        batch['block_nums'] = torch.tensor(batch['block_nums'])
        batch['trial_nums'] = torch.tensor(batch['trial_nums'])
        
        return batch
    
    def _create_batch_index_train(self) -> Dict:
        """Create batch index for training (random sampling)."""
        batch_index = {}
        
        non_must_include_days = None
        if self.must_include_days is not None:
            non_must_include_days = [
                d for d in self.trial_indicies.keys() 
                if d not in self.must_include_days
            ]
        
        for batch_idx in range(self.n_batches):
            batch = {}
            
            # Select days for this batch
            if self.must_include_days is not None:
                days = list(self.must_include_days)
                remaining = self.days_per_batch - len(days)
                if remaining > 0 and non_must_include_days:
                    extra_days = self.rng.choice(
                        non_must_include_days, 
                        size=min(remaining, len(non_must_include_days)),
                        replace=False
                    )
                    days.extend(extra_days)
            else:
                days = self.rng.choice(
                    list(self.trial_indicies.keys()),
                    size=min(self.days_per_batch, self.n_days),
                    replace=False
                )
            
            # Allocate trials per day
            trials_per_day = self.batch_size // len(days)
            
            for d in days:
                available_trials = self.trial_indicies[d]['trial_indicies']
                selected = self.rng.choice(
                    available_trials,
                    size=min(trials_per_day, len(available_trials)),
                    replace=False
                )
                batch[d] = list(selected)
            
            batch_index[batch_idx] = batch
        
        return batch_index
    
    def _create_batch_index_test(self) -> Dict:
        """Create batch index for testing (sequential, all data)."""
        batch_index = {}
        batch_idx = 0
        
        for d in self.trial_indicies.keys():
            trials = self.trial_indicies[d]['trial_indicies']
            
            for i in range(0, len(trials), self.batch_size):
                batch_trials = trials[i:i + self.batch_size]
                batch_index[batch_idx] = {d: list(batch_trials)}
                batch_idx += 1
        
        return batch_index


def collate_multi_reference(batch_list: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    
    Note: If using batch_size=None in DataLoader (dataset returns full batches),
    this function is not needed.
    """
    # Assuming batch_list contains a single pre-batched item
    if len(batch_list) == 1:
        return batch_list[0]
    
    # Otherwise, merge multiple items
    raise NotImplementedError("Multiple batch merging not implemented")


# Testing
if __name__ == "__main__":
    print("MultiReferenceDataset module loaded successfully")
    print(f"Phoneme vocabulary size: {len(LOGIT_PHONE_DEF)}")
    print(f"Available phonemes: {LOGIT_PHONE_DEF}")
