"""
Multi-Reference Brain-to-Text Trainer

This trainer extends the original BrainToTextDecoder_Trainer to support
multi-reference CTC loss, allowing multiple valid pronunciations per trial.

Author: Generated for Utah Brain-to-Text project
"""

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import time
import os
import numpy as np
import math
import logging
import json

import torchaudio.functional as F
from omegaconf import OmegaConf

# Import multi-reference components
from multi_reference_ctc_loss import (
    MultiReferenceCTCLoss,
    MultiReferenceCTCLossOptimized,
    prepare_multi_reference_batch
)
from multi_reference_dataset import MultiReferenceDataset
from multi_pronunciation_lexicon import (
    MultiPronunciationLexicon,
    DialectVariants,
    PHONE_TO_IDX
)

# These would be imported from your existing codebase
# from rnn_model import GRUDecoder
# from data_augmentations import gauss_smooth

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True


class MultiReferenceBrainToTextTrainer:
    """
    Trainer for brain-to-text models using multi-reference CTC loss.
    
    This allows training where multiple pronunciations are considered correct,
    accounting for dialect variations like the cot-caught merger.
    """
    
    def __init__(self, args: dict, model_class=None):
        """
        Args:
            args: Configuration dictionary
            model_class: The model class to instantiate (e.g., GRUDecoder)
        """
        self.args = args
        self.model_class = model_class
        
        # Trainer state
        self.logger = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        
        # Loss functions
        self.ctc_loss_standard = None  # Standard CTC for backward compatibility
        self.ctc_loss_multiref = None  # Multi-reference CTC
        
        self.best_val_PER = float('inf')
        self.best_val_loss = float('inf')
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        # Multi-reference settings
        self.use_multi_reference = args.get('use_multi_reference', True)
        self.merger_names = args.get('merger_names', ['COT_CAUGHT', 'WEAK_VOWEL'])
        self.max_variants = args.get('max_variants_per_trial', 8)
        
        # Initialize
        self._setup_logging()
        self._setup_device()
        self._setup_directories()
        
    def _setup_logging(self):
        """Setup logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_device(self):
        """Setup compute device."""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
            
    def _setup_directories(self):
        """Create output directories."""
        if self.args.get('mode') == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=True)
            if self.args.get('save_best_checkpoint'):
                os.makedirs(self.args['checkpoint_dir'], exist_ok=True)
    
    def initialize(self, train_trials: dict, val_trials: dict, feature_subset=None):
        """
        Initialize model, datasets, optimizer, and loss functions.
        
        Args:
            train_trials: Dictionary of training trial indices
            val_trials: Dictionary of validation trial indices
            feature_subset: Optional subset of features to use
        """
        # Initialize model
        if self.model_class is not None:
            self.model = self.model_class(**self.args['model'])
            self.logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Initialize datasets with multi-reference support
        self.train_dataset = MultiReferenceDataset(
            trial_indicies=train_trials,
            split='train',
            days_per_batch=self.args['dataset']['days_per_batch'],
            n_batches=self.args['num_training_batches'],
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=self.args['dataset'].get('must_include_days'),
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset,
            merger_names=self.merger_names if self.use_multi_reference else None,
            max_variants_per_trial=self.max_variants
        )
        
        self.val_dataset = MultiReferenceDataset(
            trial_indicies=val_trials,
            split='test',
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args['dataset']['batch_size'],
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset,
            merger_names=self.merger_names if self.use_multi_reference else None,
            max_variants_per_trial=self.max_variants
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,  # Dataset returns full batches
            shuffle=False,
            num_workers=self.args['dataset'].get('num_dataloader_workers', 0),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args['lr_max'],
            weight_decay=self.args.get('weight_decay', 0.01)
        )
        
        # Initialize learning rate scheduler
        self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=self.args['lr_min'] / self.args['lr_max'],
            total_iters=self.args['lr_decay_steps']
        )
        
        # Initialize loss functions
        self.ctc_loss_standard = torch.nn.CTCLoss(
            blank=0, reduction='none', zero_infinity=False
        )
        
        if self.use_multi_reference:
            self.ctc_loss_multiref = MultiReferenceCTCLoss(
                blank=0, reduction='none', zero_infinity=False
            )
            self.logger.info(f"Using multi-reference CTC loss with mergers: {self.merger_names}")
        
        # Move model to device
        self.model.to(self.device)
        
        self.logger.info("Initialization complete")
    
    def compute_loss_multi_reference(self, 
                                     logits: torch.Tensor,
                                     batch: dict,
                                     adjusted_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-reference CTC loss.
        
        For each sample, computes loss against all valid pronunciation variants
        and returns the minimum.
        
        Args:
            logits: (B, T, C) model output logits
            batch: Batch dictionary with 'seq_class_ids_variants' and 'variant_lengths'
            adjusted_lens: (B,) adjusted input lengths
            
        Returns:
            Scalar loss value
        """
        log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2])  # (T, B, C)
        
        # Get variants from batch
        targets = batch['seq_class_ids_variants']  # List of lists of tensors
        target_lengths = batch['variant_lengths']   # List of lists of ints
        
        # Compute multi-reference loss
        loss = self.ctc_loss_multiref(
            log_probs=log_probs,
            targets=targets,
            input_lengths=adjusted_lens,
            target_lengths=target_lengths
        )
        
        return loss.mean()
    
    def compute_loss_standard(self,
                              logits: torch.Tensor,
                              labels: torch.Tensor,
                              adjusted_lens: torch.Tensor,
                              phone_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Compute standard (single-reference) CTC loss.
        
        This is the original loss function for backward compatibility.
        """
        log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2])
        
        loss = self.ctc_loss_standard(
            log_probs=log_probs,
            targets=labels,
            input_lengths=adjusted_lens,
            target_lengths=phone_seq_lens
        )
        
        return loss.mean()
    
    def train_step(self, batch: dict) -> dict:
        """
        Execute a single training step.
        
        Returns:
            Dict with 'loss' and optionally other metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        features = batch['input_features'].to(self.device)
        labels = batch['seq_class_ids'].to(self.device)
        n_time_steps = batch['n_time_steps'].to(self.device)
        phone_seq_lens = batch['phone_seq_lens'].to(self.device)
        day_indicies = batch['day_indicies'].to(self.device)
        
        with torch.autocast(device_type="cuda", 
                           enabled=self.args.get('use_amp', True), 
                           dtype=torch.bfloat16):
            
            # Compute adjusted lengths based on model architecture
            patch_size = self.args['model'].get('patch_size', 1)
            patch_stride = self.args['model'].get('patch_stride', 1)
            adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
            
            # Forward pass
            logits = self.model(features, day_indicies)
            
            # Compute loss
            if self.use_multi_reference:
                loss = self.compute_loss_multi_reference(logits, batch, adjusted_lens)
            else:
                loss = self.compute_loss_standard(logits, labels, adjusted_lens, phone_seq_lens)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        grad_norm = 0.0
        if self.args.get('grad_norm_clip_value', 0) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.args['grad_norm_clip_value']
            )
        
        self.optimizer.step()
        self.learning_rate_scheduler.step()
        
        return {
            'loss': loss.detach().item(),
            'grad_norm': float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
        }
    
    def validate(self, loader=None) -> dict:
        """
        Run validation and compute metrics.
        
        Returns:
            Dict with 'avg_loss', 'avg_PER', and per-day metrics
        """
        if loader is None:
            loader = self.val_loader
            
        self.model.eval()
        
        total_loss = 0.0
        total_edit_distance = 0
        total_seq_length = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                features = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)
                
                with torch.autocast(device_type="cuda",
                                   enabled=self.args.get('use_amp', True),
                                   dtype=torch.bfloat16):
                    
                    patch_size = self.args['model'].get('patch_size', 1)
                    patch_stride = self.args['model'].get('patch_stride', 1)
                    adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
                    
                    logits = self.model(features, day_indicies)
                    
                    # Always use standard loss for validation (canonical sequence)
                    # This gives comparable metrics across experiments
                    loss = self.compute_loss_standard(
                        logits, labels, adjusted_lens, phone_seq_lens
                    )
                
                total_loss += loss.item()
                n_batches += 1
                
                # Compute PER
                for i in range(logits.shape[0]):
                    decoded = torch.argmax(
                        logits[i, :adjusted_lens[i], :], dim=-1
                    )
                    decoded = torch.unique_consecutive(decoded)
                    decoded = decoded[decoded != 0].cpu().numpy()
                    
                    true_seq = labels[i, :phone_seq_lens[i]].cpu().numpy()
                    
                    total_edit_distance += F.edit_distance(
                        list(decoded), list(true_seq)
                    )
                    total_seq_length += len(true_seq)
        
        avg_loss = total_loss / max(n_batches, 1)
        avg_per = total_edit_distance / max(total_seq_length, 1)
        
        return {
            'avg_loss': avg_loss,
            'avg_PER': avg_per,
            'total_edit_distance': total_edit_distance,
            'total_seq_length': total_seq_length
        }
    
    def train(self) -> dict:
        """
        Main training loop.
        
        Returns:
            Dict with training history
        """
        self.logger.info("Starting training...")
        
        train_losses = []
        val_losses = []
        val_pers = []
        
        best_per = float('inf')
        steps_without_improvement = 0
        
        start_time = time.time()
        
        for i, batch in enumerate(self.train_loader):
            # Training step
            step_metrics = self.train_step(batch)
            train_losses.append(step_metrics['loss'])
            
            # Logging
            if i % self.args.get('batches_per_train_log', 100) == 0:
                self.logger.info(
                    f"Batch {i}: loss={step_metrics['loss']:.4f}, "
                    f"grad_norm={step_metrics['grad_norm']:.2f}"
                )
            
            # Validation
            if i % self.args.get('batches_per_val_step', 500) == 0:
                val_metrics = self.validate()
                val_losses.append(val_metrics['avg_loss'])
                val_pers.append(val_metrics['avg_PER'])
                
                self.logger.info(
                    f"Validation at batch {i}: "
                    f"loss={val_metrics['avg_loss']:.4f}, "
                    f"PER={val_metrics['avg_PER']:.4f}"
                )
                
                # Check for improvement
                if val_metrics['avg_PER'] < best_per:
                    best_per = val_metrics['avg_PER']
                    steps_without_improvement = 0
                    
                    if self.args.get('save_best_checkpoint'):
                        self._save_checkpoint(
                            f"{self.args['checkpoint_dir']}/best_checkpoint",
                            val_metrics['avg_PER'],
                            val_metrics['avg_loss']
                        )
                else:
                    steps_without_improvement += 1
                
                # Early stopping
                if (self.args.get('early_stopping') and 
                    steps_without_improvement >= self.args.get('early_stopping_val_steps', 10)):
                    self.logger.info(f"Early stopping at batch {i}")
                    break
            
            # Check if done
            if i >= self.args.get('num_training_batches', float('inf')) - 1:
                break
        
        duration = time.time() - start_time
        self.logger.info(f"Training complete in {duration/60:.2f} minutes")
        self.logger.info(f"Best validation PER: {best_per:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_PERs': val_pers,
            'best_PER': best_per,
            'training_time': duration
        }
    
    def _save_checkpoint(self, path: str, per: float, loss: float):
        """Save a model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_PER': per,
            'val_loss': loss,
            'args': self.args
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")


def create_multiref_args_from_existing(existing_args: dict) -> dict:
    """
    Create multi-reference training args from existing args.
    
    This adds the necessary multi-reference settings to an existing config.
    """
    args = dict(existing_args)
    
    # Add multi-reference settings
    args['use_multi_reference'] = True
    args['merger_names'] = ['COT_CAUGHT', 'WEAK_VOWEL']
    args['max_variants_per_trial'] = 8
    
    return args


# Example usage
if __name__ == "__main__":
    print("Multi-Reference Trainer module loaded successfully")
    print(f"Available dialect mergers: {list(DialectVariants.MERGERS.keys())}")
