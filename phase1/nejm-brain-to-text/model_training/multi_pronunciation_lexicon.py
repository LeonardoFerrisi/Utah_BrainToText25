"""
Multi-Pronunciation Lexicon for Brain-to-Text Training

Generates multiple valid phoneme sequences for a given sentence,
accounting for dialect variations like the cot-caught merger.

Author: Generated for Utah Brain-to-Text project
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from itertools import product
from g2p_en import G2p

# Phoneme definitions (matching the model's output vocabulary)
LOGIT_PHONE_DEF = [
    'BLANK', 'SIL',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

# Create phoneme to index mapping
PHONE_TO_IDX = {phone: idx for idx, phone in enumerate(LOGIT_PHONE_DEF)}


class DialectVariants:
    """
    Defines phoneme equivalence classes for valid dialect variations.
    
    Only includes mergers that represent valid English pronunciations,
    not decoding errors.
    """
    
    MERGERS = {
        # Cot-caught merger (Western/Canadian American English)
        # "caught" and "cot" are pronounced the same
        # ~34 errors in dataset
        'COT_CAUGHT': [
            ('AA', 'AO'),
        ],
        
        # Additional valid mergers (commented out - enable if needed)
        # These have more limited validity or context-dependence
        
        # # Pin-pen merger (Southern US) - only valid before nasals
        # 'PIN_PEN': [
        #     ('IH', 'EH'),
        # ],
        
        # # Non-rhotic dialects (rare in American English)
        # 'RHOTIC': [
        #     ('ER', 'AH'),
        # ],
    }
    
    # Default: only linguistically valid mergers
    DEFAULT_MERGERS = ['COT_CAUGHT']
    
    @classmethod
    def get_equivalence_pairs(cls, merger_names: Optional[List[str]] = None) -> List[Tuple[str, str]]:
        """Get all phoneme equivalence pairs for specified mergers."""
        if merger_names is None:
            merger_names = cls.DEFAULT_MERGERS
        
        pairs = []
        for name in merger_names:
            if name in cls.MERGERS:
                pairs.extend(cls.MERGERS[name])
        return pairs
    
    @classmethod
    def build_equivalence_map(cls, merger_names: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Build a mapping from each phoneme to all its equivalent phonemes.
        
        Returns:
            Dict mapping phoneme -> list of equivalent phonemes (including itself)
        """
        pairs = cls.get_equivalence_pairs(merger_names)
        
        # Initialize with identity mapping
        equiv_map = {phone: [phone] for phone in LOGIT_PHONE_DEF}
        
        # Add equivalences
        for p1, p2 in pairs:
            if p1 in equiv_map and p2 not in equiv_map[p1]:
                equiv_map[p1].append(p2)
            if p2 in equiv_map and p1 not in equiv_map[p2]:
                equiv_map[p2].append(p1)
        
        return equiv_map


class MultiPronunciationLexicon:
    """
    Generates multiple valid phoneme sequences for sentences,
    accounting for dialect variations.
    """
    
    def __init__(self, 
                 merger_names: Optional[List[str]] = None,
                 max_variants_per_sentence: int = 4,
                 g2p_instance: Optional[G2p] = None):
        """
        Args:
            merger_names: List of dialect mergers to consider
            max_variants_per_sentence: Maximum number of pronunciation variants
            g2p_instance: Optional pre-initialized G2p instance
        """
        self.merger_names = merger_names or DialectVariants.DEFAULT_MERGERS
        self.max_variants = max_variants_per_sentence
        self.equiv_map = DialectVariants.build_equivalence_map(self.merger_names)
        self.g2p = g2p_instance or G2p()
    
    def remove_punctuation(self, sentence: str) -> str:
        """Remove punctuation from text."""
        sentence = re.sub(r'[^a-zA-Z\- \']', '', sentence)
        sentence = sentence.replace('--', '').lower()
        sentence = sentence.replace(" '", "'").lower()
        sentence = sentence.strip()
        sentence = ' '.join(sentence.split())
        return sentence
    
    def sentence_to_phonemes(self, sentence: str) -> Tuple[List[str], str]:
        """Convert sentence to canonical phoneme sequence."""
        sentence = self.remove_punctuation(sentence)
        
        phonemes = []
        if len(sentence) == 0:
            phonemes = ['SIL']
        else:
            for p in self.g2p(sentence):
                if p == ' ':
                    phonemes.append('SIL')
                else:
                    p = re.sub(r'[0-9]', '', p)  # Remove stress markers
                    if re.match(r'[A-Z]+', p):
                        phonemes.append(p)
            phonemes.append('SIL')  # End with silence
        
        return phonemes, sentence
    
    def generate_variants(self, phonemes: List[str]) -> List[List[str]]:
        """
        Generate all valid pronunciation variants for a phoneme sequence.
        """
        # Find positions with alternatives
        alternatives_per_position = []
        for phone in phonemes:
            alts = self.equiv_map.get(phone, [phone])
            alternatives_per_position.append(alts)
        
        # Calculate total combinations
        n_combinations = 1
        for alts in alternatives_per_position:
            n_combinations *= len(alts)
        
        # If too many combinations, use sampling
        if n_combinations > self.max_variants:
            return self._sample_variants(phonemes, alternatives_per_position)
        
        # Generate all combinations
        variants = []
        for combo in product(*alternatives_per_position):
            variants.append(list(combo))
        
        return variants
    
    def _sample_variants(self, 
                         phonemes: List[str], 
                         alternatives: List[List[str]]) -> List[List[str]]:
        """Sample variants when there are too many combinations."""
        variants = [phonemes.copy()]  # Always include original
        
        variable_positions = [i for i, alts in enumerate(alternatives) if len(alts) > 1]
        
        np.random.seed(42)
        for _ in range(self.max_variants - 1):
            variant = phonemes.copy()
            for pos in variable_positions:
                if np.random.random() < 0.5:
                    alts = alternatives[pos]
                    variant[pos] = alts[np.random.randint(len(alts))]
            
            if variant not in variants:
                variants.append(variant)
        
        return variants
    
    def get_sentence_variants(self, sentence: str) -> Dict:
        """
        Get all pronunciation variants for a sentence.
        
        Returns:
            Dict with 'canonical', 'variants', 'variant_ids', 'n_variants'
        """
        phonemes, cleaned = self.sentence_to_phonemes(sentence)
        variants = self.generate_variants(phonemes)
        
        # Convert to IDs
        variant_ids = []
        for var in variants:
            ids = [PHONE_TO_IDX.get(p, 0) for p in var]
            variant_ids.append(ids)
        
        return {
            'sentence': cleaned,
            'canonical': phonemes,
            'variants': variants,
            'variant_ids': variant_ids,
            'n_variants': len(variants)
        }


def phonemes_to_ids(phonemes: List[str]) -> List[int]:
    """Convert phoneme list to integer IDs."""
    return [PHONE_TO_IDX.get(p, 0) for p in phonemes]


def ids_to_phonemes(ids: List[int]) -> List[str]:
    """Convert integer IDs back to phonemes."""
    return [LOGIT_PHONE_DEF[i] for i in ids]
