"""
Multi-Reference CTC Loss for Brain-to-Text Training

This module implements a CTC loss that accepts multiple valid target sequences
and backpropagates only through the minimum loss (best matching variant).

Author: Generated for Utah Brain-to-Text project
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class MultiReferenceCTCLoss(nn.Module):
    """
    CTC Loss that supports multiple reference sequences per sample.
    
    For each sample, computes CTC loss against all provided reference sequences
    and returns the minimum loss. This allows training to accept any valid
    pronunciation as correct.
    
    The gradient flows only through the minimum-loss path, so the model learns
    to match whichever pronunciation variant is closest to its current prediction.
    """
    
    def __init__(self, 
                 blank: int = 0, 
                 reduction: str = 'none',
                 zero_infinity: bool = False):
        """
        Args:
            blank: Index of the blank label
            reduction: 'none', 'mean', or 'sum'
            zero_infinity: Whether to zero infinite losses
        """
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        
        # Base CTC loss (always use 'none' reduction internally)
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
                T = max time steps, N = batch size, C = num classes
            targets: List of length N, where each element is a list of 
                valid target tensors for that sample
            input_lengths: (N,) - Length of each input sequence
            target_lengths: List of length N, where each element is a list
                of lengths corresponding to each target variant
                
        Returns:
            Tensor of shape (N,) if reduction='none', else scalar
        """
        batch_size = log_probs.size(1)
        device = log_probs.device
        
        min_losses = []
        
        for batch_idx in range(batch_size):
            # Get all target variants for this sample
            target_variants = targets[batch_idx]
            length_variants = target_lengths[batch_idx]
            
            # Get the log_probs for this sample: (T, 1, C)
            sample_log_probs = log_probs[:, batch_idx:batch_idx+1, :]
            sample_input_length = input_lengths[batch_idx:batch_idx+1]
            
            # Compute CTC loss for each variant
            variant_losses = []
            for target, target_len in zip(target_variants, length_variants):
                # Ensure target is on correct device
                target = target.to(device)
                target_len_tensor = torch.tensor([target_len], device=device)
                
                loss = self.ctc_loss(
                    sample_log_probs,
                    target.unsqueeze(0),  # (1, S)
                    sample_input_length,
                    target_len_tensor
                )
                variant_losses.append(loss)
            
            # Stack and take minimum
            variant_losses = torch.stack(variant_losses)
            min_loss = torch.min(variant_losses)
            min_losses.append(min_loss)
        
        # Stack all minimum losses
        result = torch.stack(min_losses)
        
        # Apply reduction
        if self.reduction == 'mean':
            return result.mean()
        elif self.reduction == 'sum':
            return result.sum()
        else:
            return result


class MultiReferenceCTCLossOptimized(nn.Module):
    """
    Optimized version of MultiReferenceCTCLoss that batches computations
    where possible for better GPU utilization.
    
    This version groups samples by number of variants and processes them
    together, which is more efficient when most samples have similar
    numbers of pronunciation variants.
    """
    
    def __init__(self, 
                 blank: int = 0, 
                 reduction: str = 'none',
                 zero_infinity: bool = False,
                 max_variants: int = 16):
        """
        Args:
            blank: Index of the blank label
            reduction: 'none', 'mean', or 'sum'
            zero_infinity: Whether to zero infinite losses
            max_variants: Maximum number of variants to consider per sample
        """
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.max_variants = max_variants
        
        self.ctc_loss = nn.CTCLoss(
            blank=blank, 
            reduction='none', 
            zero_infinity=zero_infinity
        )
    
    def forward(self,
                log_probs: torch.Tensor,
                targets_padded: torch.Tensor,
                targets_mask: torch.Tensor,
                input_lengths: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-reference CTC loss with batched computation.
        
        Args:
            log_probs: (T, N, C) - Log probabilities from the model
            targets_padded: (N, V, S) - Padded targets where V is max variants,
                S is max target sequence length
            targets_mask: (N, V) - Boolean mask indicating valid variants
            input_lengths: (N,) - Length of each input sequence  
            target_lengths: (N, V) - Length of each target variant
                
        Returns:
            Tensor of shape (N,) if reduction='none', else scalar
        """
        batch_size = log_probs.size(1)
        n_variants = targets_padded.size(1)
        device = log_probs.device
        
        # Expand log_probs to compute all variants at once
        # (T, N, C) -> (T, N*V, C)
        T, N, C = log_probs.shape
        log_probs_expanded = log_probs.unsqueeze(2).expand(T, N, n_variants, C)
        log_probs_expanded = log_probs_expanded.reshape(T, N * n_variants, C)
        
        # Expand input lengths: (N,) -> (N*V,)
        input_lengths_expanded = input_lengths.unsqueeze(1).expand(N, n_variants)
        input_lengths_expanded = input_lengths_expanded.reshape(N * n_variants)
        
        # Flatten targets: (N, V, S) -> (N*V, S)
        targets_flat = targets_padded.reshape(N * n_variants, -1)
        
        # Flatten target lengths: (N, V) -> (N*V,)
        target_lengths_flat = target_lengths.reshape(N * n_variants)
        
        # Compute CTC loss for all variants at once
        # This is more efficient than looping
        all_losses = self.ctc_loss(
            log_probs_expanded,
            targets_flat,
            input_lengths_expanded,
            target_lengths_flat
        )
        
        # Reshape back: (N*V,) -> (N, V)
        all_losses = all_losses.reshape(N, n_variants)
        
        # Mask invalid variants with large value
        all_losses = torch.where(
            targets_mask,
            all_losses,
            torch.tensor(float('inf'), device=device)
        )
        
        # Take minimum across variants
        min_losses, _ = torch.min(all_losses, dim=1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return min_losses.mean()
        elif self.reduction == 'sum':
            return min_losses.sum()
        else:
            return min_losses


def prepare_multi_reference_batch(
    targets_list: List[List[List[int]]],
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of multi-reference targets for the optimized loss.
    
    Args:
        targets_list: List of samples, each containing list of variant sequences
        device: Device to place tensors on
        
    Returns:
        targets_padded: (N, V, S) padded targets
        targets_mask: (N, V) boolean mask for valid variants
        target_lengths: (N, V) length of each variant
    """
    batch_size = len(targets_list)
    max_variants = max(len(variants) for variants in targets_list)
    max_seq_len = max(
        len(seq) 
        for variants in targets_list 
        for seq in variants
    )
    
    # Initialize tensors
    targets_padded = torch.zeros(batch_size, max_variants, max_seq_len, 
                                  dtype=torch.long, device=device)
    targets_mask = torch.zeros(batch_size, max_variants, 
                               dtype=torch.bool, device=device)
    target_lengths = torch.zeros(batch_size, max_variants, 
                                  dtype=torch.long, device=device)
    
    # Fill in values
    for i, variants in enumerate(targets_list):
        for j, seq in enumerate(variants):
            seq_len = len(seq)
            targets_padded[i, j, :seq_len] = torch.tensor(seq, dtype=torch.long)
            targets_mask[i, j] = True
            target_lengths[i, j] = seq_len
    
    return targets_padded, targets_mask, target_lengths


# Testing
if __name__ == "__main__":
    # Test the multi-reference CTC loss
    torch.manual_seed(42)
    
    # Simulated model output
    T, N, C = 50, 2, 40  # time, batch, classes
    log_probs = torch.randn(T, N, C).log_softmax(dim=-1)
    input_lengths = torch.tensor([50, 45])
    
    # Two samples, each with multiple valid targets
    targets_list = [
        [[1, 2, 3, 4, 5], [1, 3, 3, 4, 5]],  # Sample 1: 2 variants
        [[2, 3, 4], [2, 4, 4], [2, 3, 5]],    # Sample 2: 3 variants
    ]
    
    # Test simple version
    print("Testing MultiReferenceCTCLoss...")
    loss_fn = MultiReferenceCTCLoss(blank=0, reduction='none')
    
    targets = [[torch.tensor(v) for v in variants] for variants in targets_list]
    target_lengths = [[len(v) for v in variants] for variants in targets_list]
    
    loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
    print(f"Simple version losses: {loss}")
    
    # Test optimized version
    print("\nTesting MultiReferenceCTCLossOptimized...")
    loss_fn_opt = MultiReferenceCTCLossOptimized(blank=0, reduction='none')
    
    targets_padded, targets_mask, target_lens = prepare_multi_reference_batch(targets_list)
    
    loss_opt = loss_fn_opt(log_probs, targets_padded, targets_mask, 
                           input_lengths, target_lens)
    print(f"Optimized version losses: {loss_opt}")
    
    print("\nLosses should be similar (minor numerical differences expected)")
