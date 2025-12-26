"""
Example: Training with Multi-Reference CTC Loss

This script demonstrates how to use the multi-reference CTC training pipeline
for brain-to-text decoding. The pipeline accepts multiple valid pronunciations
for each trial, reducing errors from dialect variations.

Usage:
    python train_multiref_example.py --config rnn_args.yaml

Author: Generated for Utah Brain-to-Text project
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

# Import the multi-reference training components
from multi_pronunciation_lexicon import (
    MultiPronunciationLexicon,
    DialectVariants,
    LOGIT_PHONE_DEF
)
from multi_reference_ctc_loss import MultiReferenceCTCLoss
from multi_reference_dataset import MultiReferenceDataset
from multi_reference_trainer import (
    MultiReferenceBrainToTextTrainer,
    create_multiref_args_from_existing
)


def demonstrate_pronunciation_variants():
    """Show how the pronunciation variant system works."""
    
    print("=" * 60)
    print("DEMONSTRATION: Pronunciation Variant Generation")
    print("=" * 60)
    
    # Create lexicon with different dialect settings
    lexicon_cot_caught = MultiPronunciationLexicon(
        merger_names=['COT_CAUGHT']
    )
    
    lexicon_all_mergers = MultiPronunciationLexicon(
        merger_names=['COT_CAUGHT', 'WEAK_VOWEL', 'VOICING']
    )
    
    test_sentences = [
        "caught the ball",
        "the water bottle",
        "I thought about it",
        "the cat sat on the mat"
    ]
    
    print("\n1. With COT_CAUGHT merger only:")
    print("-" * 40)
    for sentence in test_sentences:
        result = lexicon_cot_caught.get_sentence_variants(sentence)
        print(f"\n'{sentence}'")
        print(f"  Canonical: {' '.join(result['canonical'])}")
        print(f"  Variants: {result['n_variants']}")
        
    print("\n\n2. With multiple mergers (COT_CAUGHT + WEAK_VOWEL + VOICING):")
    print("-" * 40)
    for sentence in test_sentences:
        result = lexicon_all_mergers.get_sentence_variants(sentence)
        print(f"\n'{sentence}'")
        print(f"  Canonical: {' '.join(result['canonical'])}")
        print(f"  Variants: {result['n_variants']}")
        if result['n_variants'] <= 4:
            for i, var in enumerate(result['variants'][:4]):
                marker = " (canonical)" if i == 0 else ""
                print(f"    {i+1}. {' '.join(var)}{marker}")


def demonstrate_loss_computation():
    """Show how the multi-reference loss works."""
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION: Multi-Reference CTC Loss")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    # Simulated model output: (T=50, N=2, C=40)
    T, N, C = 50, 2, 40
    logits = torch.randn(T, N, C)
    log_probs = logits.log_softmax(dim=-1)
    
    input_lengths = torch.tensor([50, 45])
    
    # Standard CTC: single target per sample
    standard_targets = torch.tensor([
        [1, 2, 3, 4, 5, 0, 0],  # Sample 1: "AA AE AH AO AW"
        [2, 3, 4, 0, 0, 0, 0],  # Sample 2: "AE AH AO"
    ])
    standard_lengths = torch.tensor([5, 3])
    
    standard_loss = torch.nn.CTCLoss(blank=0, reduction='mean')
    loss_standard = standard_loss(
        log_probs.permute(1, 0, 2).log_softmax(dim=-1),
        standard_targets,
        input_lengths,
        standard_lengths
    )
    
    print(f"\n1. Standard CTC Loss (single reference):")
    print(f"   Loss = {loss_standard.item():.4f}")
    
    # Multi-reference CTC: multiple valid targets per sample
    multiref_loss = MultiReferenceCTCLoss(blank=0, reduction='none')
    
    # Sample 1: Two valid pronunciations
    # Sample 2: Three valid pronunciations
    targets = [
        [torch.tensor([1, 2, 3, 4, 5]), torch.tensor([1, 3, 3, 4, 5])],
        [torch.tensor([2, 3, 4]), torch.tensor([2, 4, 4]), torch.tensor([2, 3, 5])]
    ]
    target_lengths = [
        [5, 5],
        [3, 3, 3]
    ]
    
    loss_multiref = multiref_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths
    )
    
    print(f"\n2. Multi-Reference CTC Loss:")
    print(f"   Sample 1 (2 variants): loss = {loss_multiref[0].item():.4f}")
    print(f"   Sample 2 (3 variants): loss = {loss_multiref[1].item():.4f}")
    print(f"   Mean loss = {loss_multiref.mean().item():.4f}")
    
    print("\n   The multi-reference loss is ≤ standard loss because it")
    print("   uses the minimum loss across all valid pronunciations.")


def show_integration_example():
    """Show how to integrate with existing training code."""
    
    print("\n" + "=" * 60)
    print("INTEGRATION: Modifying Existing Training Code")
    print("=" * 60)
    
    print("""
To integrate multi-reference training into your existing pipeline:

1. MODIFY YOUR CONFIG (rnn_args.yaml):
   ```yaml
   # Add these settings
   use_multi_reference: true
   merger_names:
     - COT_CAUGHT
     - WEAK_VOWEL
   max_variants_per_trial: 8
   ```

2. MODIFY rnn_trainer.py - Replace the loss computation:

   # BEFORE (single reference):
   loss = self.ctc_loss(
       log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
       targets=labels,
       input_lengths=adjusted_lens,
       target_lengths=phone_seq_lens
   )
   
   # AFTER (multi-reference):
   from multi_reference_ctc_loss import MultiReferenceCTCLoss
   
   # In __init__:
   self.ctc_loss_multiref = MultiReferenceCTCLoss(blank=0, reduction='none')
   
   # In train():
   loss = self.ctc_loss_multiref(
       log_probs=torch.permute(logits.log_softmax(2), [1, 0, 2]),
       targets=batch['seq_class_ids_variants'],
       input_lengths=adjusted_lens,
       target_lengths=batch['variant_lengths']
   )
   loss = loss.mean()

3. MODIFY dataset.py - Use MultiReferenceDataset:
   
   from multi_reference_dataset import MultiReferenceDataset
   
   # Replace BrainToTextDataset with MultiReferenceDataset
   # It returns additional fields:
   #   - batch['seq_class_ids_variants']: List of variant tensors per sample
   #   - batch['variant_lengths']: List of lengths per sample

4. KEEP VALIDATION UNCHANGED:
   For fair comparison, continue using standard CTC loss during validation.
   This ensures metrics are comparable across experiments.
""")


def main():
    """Main demonstration."""
    
    print("\n" + "=" * 60)
    print("MULTI-REFERENCE CTC TRAINING PIPELINE")
    print("For Brain-to-Text with Dialect Variation Support")
    print("=" * 60)
    
    # Show pronunciation variants
    demonstrate_pronunciation_variants()
    
    # Show loss computation
    demonstrate_loss_computation()
    
    # Show integration
    show_integration_example()
    
    print("\n" + "=" * 60)
    print("AVAILABLE DIALECT MERGERS")
    print("=" * 60)
    print("\nYou can enable any combination of these mergers:")
    for name, pairs in DialectVariants.MERGERS.items():
        print(f"\n  {name}:")
        for p1, p2 in pairs:
            print(f"    {p1} ↔ {p2}")
    
    print("\n" + "=" * 60)
    print("FILES CREATED")
    print("=" * 60)
    print("""
    1. multi_pronunciation_lexicon.py
       - Generates pronunciation variants from text
       - Handles dialect mergers (cot-caught, etc.)
    
    2. multi_reference_ctc_loss.py
       - MultiReferenceCTCLoss: Takes minimum loss across variants
       - Compatible with standard PyTorch CTC interface
    
    3. multi_reference_dataset.py
       - Extends dataset to provide multiple targets per trial
       - Caches variant generation for efficiency
    
    4. multi_reference_trainer.py
       - Complete trainer with multi-reference support
       - Drop-in replacement for existing trainer
    
    5. train_multiref_example.py (this file)
       - Demonstration and usage examples
""")


if __name__ == "__main__":
    main()
