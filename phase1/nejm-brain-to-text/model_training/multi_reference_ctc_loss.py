"""
Multi-Reference CTC Loss for Brain-to-Text Training

Computes CTC loss against multiple valid target sequences and
returns the minimum (best matching) loss for backpropagation.

Author: Generated for Utah Brain-to-Text project
"""

import torch
import torch.nn as nn
from typing import List


class MultiReferenceCTCLoss(nn.Module):
    """
    CTC Loss that supports multiple reference sequences per sample.
    
    For each sample, computes CTC loss against all provided reference sequences
    and returns the minimum loss. This allows training to accept any valid
    pronunciation as correct (e.g., both "caught" with AA and AO).
    """
    
    def __init__(self, 
                 blank: int = 0, 
                 reduction: str = 'none',
                 zero_infinity: bool = False):
        """
        Args:
            blank: Index of the blank label (default: 0)
            reduction: 'none', 'mean', or 'sum'
            zero_infinity: Whether to zero infinite losses
        """
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        
        self.ctc_loss = nn.CTCLoss(
            blank=blank, 
            reduction='none', 
            zero_infinity=zero_infinity
        )
    
    def forward(self,
                log_probs: torch.Tensor,
                targets: List[List[torch.Tensor]],
                input_lengths: torch.Tensor,
                target_lengths: List[List[int]]) -> torch.Tensor:
        """
        Compute multi-reference CTC loss.
        
        Args:
            log_probs: (T, N, C) - Log probabilities from the model
                T = time steps, N = batch size, C = num classes
            targets: List of length N, where each element is a list of 
                valid target tensors for that sample
            input_lengths: (N,) - Length of each input sequence
            target_lengths: List of length N, where each element is a list
                of lengths for each target variant
                
        Returns:
            Loss tensor - shape depends on reduction setting
        """
        batch_size = log_probs.size(1)
        device = log_probs.device
        
        min_losses = []
        
        for batch_idx in range(batch_size):
            target_variants = targets[batch_idx]
            length_variants = target_lengths[batch_idx]
            
            # Get log_probs for this sample: (T, 1, C)
            sample_log_probs = log_probs[:, batch_idx:batch_idx+1, :]
            sample_input_length = input_lengths[batch_idx:batch_idx+1]
            
            # Compute CTC loss for each variant
            variant_losses = []
            for target, target_len in zip(target_variants, length_variants):
                target = target.to(device)
                target_len_tensor = torch.tensor([target_len], device=device)
                
                loss = self.ctc_loss(
                    sample_log_probs,
                    target.unsqueeze(0),
                    sample_input_length,
                    target_len_tensor
                )
                variant_losses.append(loss)
            
            # Take minimum loss across variants
            variant_losses = torch.stack(variant_losses)
            min_loss = torch.min(variant_losses)
            min_losses.append(min_loss)
        
        result = torch.stack(min_losses)
        
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result


class StandardOrMultiRefCTCLoss(nn.Module):
    """
    Wrapper that can operate in either standard or multi-reference mode.
    
    This allows easy switching between the two modes based on configuration.
    """
    
    def __init__(self,
                 blank: int = 0,
                 reduction: str = 'none',
                 zero_infinity: bool = False,
                 use_multi_reference: bool = True):
        super().__init__()
        self.use_multi_reference = use_multi_reference
        self.blank = blank
        self.reduction = reduction
        
        self.standard_ctc = nn.CTCLoss(
            blank=blank,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
        
        if use_multi_reference:
            self.multiref_ctc = MultiReferenceCTCLoss(
                blank=blank,
                reduction=reduction,
                zero_infinity=zero_infinity
            )
    
    def forward(self, log_probs, targets, input_lengths, target_lengths,
                targets_multiref=None, target_lengths_multiref=None):
        """
        Compute loss using either standard or multi-reference mode.
        
        Args:
            log_probs: (T, N, C) log probabilities
            targets: Standard targets (N, S) for standard mode
            input_lengths: (N,) input lengths
            target_lengths: (N,) target lengths for standard mode
            targets_multiref: Multi-ref targets (list of lists) - optional
            target_lengths_multiref: Multi-ref lengths (list of lists) - optional
        """
        if self.use_multi_reference and targets_multiref is not None:
            return self.multiref_ctc(
                log_probs, targets_multiref, input_lengths, target_lengths_multiref
            )
        else:
            return self.standard_ctc(
                log_probs, targets, input_lengths, target_lengths
            )
