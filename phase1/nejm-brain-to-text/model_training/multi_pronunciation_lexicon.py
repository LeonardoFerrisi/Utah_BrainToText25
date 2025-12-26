"""
Multi-Pronunciation Lexicon for Brain-to-Text Training

This module provides functionality to generate multiple valid phoneme sequences
for a given sentence, accounting for dialect variations like the cot-caught merger.

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
    Defines phoneme equivalence classes for common dialect variations.
    
    These are phoneme pairs that may be interchangeable for a given speaker,
    meaning either pronunciation should be considered correct.
    """
    
    # Common American English dialect mergers
    MERGERS = {
        # Cot-caught merger (Western/Canadian American English)
        'COT_CAUGHT': [('AA', 'AO')],
        
        # Pin-pen merger (Southern American English)  
        'PIN_PEN': [('IH', 'EH')],  # before nasals
        
        # Mary-merry-marry merger
        'MARY_MERRY_MARRY': [('EH', 'AE'), ('EH', 'EY')],
        
        # Weak vowel merger (unstressed vowels)
        'WEAK_VOWEL': [('AH', 'IH'), ('AH', 'AX')],
        
        # Common voicing confusions (often articulatory, not dialect)
        'VOICING': [
            ('T', 'D'),
            ('P', 'B'),
            ('K', 'G'),
            ('S', 'Z'),
            ('F', 'V'),
        ],
        
        # Rhotic variations
        'RHOTIC': [('ER', 'AH')],
    }
    
    # Default set of mergers to use
    DEFAULT_MERGERS = ['COT_CAUGHT', 'WEAK_VOWEL']
    
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
                 max_variants_per_sentence: int = 16,
                 g2p_instance: Optional[G2p] = None):
        """
        Args:
            merger_names: List of dialect mergers to consider (see DialectVariants.MERGERS)
            max_variants_per_sentence: Maximum number of pronunciation variants to generate
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
        """
        Convert sentence to canonical phoneme sequence.
        
        Returns:
            Tuple of (phoneme_list, cleaned_sentence)
        """
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
        
        Uses the equivalence map to find positions where phonemes can vary,
        then generates the Cartesian product of all possibilities.
        
        Args:
            phonemes: Canonical phoneme sequence
            
        Returns:
            List of valid phoneme sequences (including the original)
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
        
        # If too many combinations, use sampling strategy
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
        """
        Sample a subset of variants when there are too many combinations.
        
        Strategy: Always include the canonical pronunciation, then randomly
        sample from the space of variants.
        """
        variants = [phonemes.copy()]  # Always include original
        
        # Find positions with multiple options
        variable_positions = [i for i, alts in enumerate(alternatives) if len(alts) > 1]
        
        # Random sampling
        np.random.seed(42)  # For reproducibility
        for _ in range(self.max_variants - 1):
            variant = phonemes.copy()
            # Randomly flip some variable positions
            for pos in variable_positions:
                if np.random.random() < 0.5:  # 50% chance to use alternative
                    alts = alternatives[pos]
                    variant[pos] = alts[np.random.randint(len(alts))]
            
            if variant not in variants:
                variants.append(variant)
        
        return variants
    
    def get_sentence_variants(self, sentence: str) -> Dict:
        """
        Get all pronunciation variants for a sentence.
        
        Args:
            sentence: Input sentence
            
        Returns:
            Dict containing:
                - 'sentence': cleaned sentence
                - 'canonical': canonical phoneme sequence
                - 'variants': list of all valid phoneme sequences
                - 'variant_ids': list of phoneme sequences as integer IDs
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


# Example usage and testing
if __name__ == "__main__":
    lexicon = MultiPronunciationLexicon(
        merger_names=['COT_CAUGHT', 'WEAK_VOWEL', 'VOICING']
    )
    
    test_sentences = [
        "caught the ball",
        "cot in the corner", 
        "the cat sat on the mat",
        "water bottle"
    ]
    
    for sentence in test_sentences:
        result = lexicon.get_sentence_variants(sentence)
        print(f"\nSentence: '{result['sentence']}'")
        print(f"Canonical: {' '.join(result['canonical'])}")
        print(f"Number of variants: {result['n_variants']}")
        if result['n_variants'] <= 4:
            for i, var in enumerate(result['variants']):
                print(f"  Variant {i+1}: {' '.join(var)}")
