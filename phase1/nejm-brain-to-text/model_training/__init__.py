"""
Multi-Reference CTC Training Module for Brain-to-Text

This module provides multi-reference CTC loss training, which accepts
multiple valid pronunciations for each trial (e.g., cot-caught merger).
"""

from .multi_pronunciation_lexicon import (
    MultiPronunciationLexicon,
    DialectVariants,
    LOGIT_PHONE_DEF,
    PHONE_TO_IDX
)
from .multi_reference_ctc_loss import MultiReferenceCTCLoss
from .multi_reference_dataset import MultiReferenceWrapper, MultiReferenceDataset

__all__ = [
    'MultiPronunciationLexicon',
    'DialectVariants', 
    'MultiReferenceCTCLoss',
    'MultiReferenceWrapper',
    'MultiReferenceDataset',
    'LOGIT_PHONE_DEF',
    'PHONE_TO_IDX'
]
