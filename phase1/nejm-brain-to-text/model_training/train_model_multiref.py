"""
Train Brain-to-Text Model with Multi-Reference CTC Loss

This script trains the RNN decoder using multi-reference CTC loss,
which accepts multiple valid pronunciations per trial (e.g., cot-caught merger).

Usage:
    python train_model_multiref.py --config rnn_args_multiref.yaml

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
import argparse
import sys

import torchaudio.functional as F
from omegaconf import OmegaConf

# Import existing modules
from dataset import BrainToTextDataset, train_test_split_indicies
from rnn_model import GRUDecoder
from data_augmentations import gauss_smooth

# Import multi-reference modules
from multi_reference_ctc_loss import MultiReferenceCTCLoss
from multi_reference_dataset import MultiReferenceWrapper

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.deterministic = True


class MultiRefBrainToTextTrainer:
    """
    Trainer for brain-to-text using multi-reference CTC loss.
    
    Based on BrainToTextDecoder_Trainer but with multi-reference support.
    """
    
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        
        # Loss functions
        self.ctc_loss_standard = None
        self.ctc_loss_multiref = None
        
        # Multi-reference wrapper
        self.multiref_wrapper = None
        
        self.best_val_PER = float('inf')
        self.best_val_loss = float('inf')
        
        self.train_dataset = None
        self.val_dataset = None
        self.train_loader = None
        self.val_loader = None
        
        self.transform_args = args['dataset']['data_transforms']
        
        # Setup
        self._setup_logging()
        self._setup_directories()
        self._setup_device()
    
    def _setup_logging(self):
        log_file = os.path.join(self.args['output_dir'], 'training.log')
        os.makedirs(self.args['output_dir'], exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _setup_directories(self):
        os.makedirs(self.args['output_dir'], exist_ok=True)
        if self.args.get('save_best_checkpoint') or self.args.get('save_final_model'):
            os.makedirs(self.args['checkpoint_dir'], exist_ok=True)
    
    def _setup_device(self):
        gpu_num = self.args.get('gpu_number', '0')
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_num}')
            self.logger.info(f"Using CUDA device {gpu_num}: {torch.cuda.get_device_name(int(gpu_num))}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
    
    def initialize(self):
        """Initialize model, datasets, optimizer, and loss functions."""
    
        # Build file paths from data directory and sessions (matching rnn_trainer.py)
        train_file_paths = [os.path.join(self.args['dataset']['dataset_dir'], s, 'data_train.hdf5') 
                            for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args['dataset']['dataset_dir'], s, 'data_val.hdf5') 
                          for s in self.args['dataset']['sessions']]
    
        # Get train/val splits
        train_trials, _ = train_test_split_indicies(
            file_paths=train_file_paths,
            test_percentage=0,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=self.args['dataset'].get('bad_trials_dict')
        )
    
        _, val_trials = train_test_split_indicies(
            file_paths=val_file_paths,
            test_percentage=1,
            seed=self.args['dataset']['seed'],
            bad_trials_dict=self.args['dataset'].get('bad_trials_dict')
        )
        # Feature subset
        feature_subset = self.args['dataset'].get('feature_subset')
        
        # Initialize model
        self.model = GRUDecoder(
            neural_dim=self.args['model']['n_input_features'],
            n_units=self.args['model']['n_units'],
            n_days=len(self.args['dataset']['sessions']),
            n_classes=self.args['dataset']['n_classes'],
            rnn_dropout=self.args['model']['rnn_dropout'],
            input_dropout=self.args['model']['input_network']['input_layer_dropout'],
            n_layers=self.args['model']['n_layers'],
            patch_size=self.args['model']['patch_size'],
            patch_stride=self.args['model']['patch_stride'],
        )
        
        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model initialized with {n_params:,} parameters")
        
        # Initialize datasets
        self.train_dataset = BrainToTextDataset(
            trial_indicies=train_trials,
            split='train',
            days_per_batch=self.args['dataset']['days_per_batch'],
            n_batches=self.args['num_training_batches'],
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=self.args['dataset'].get('must_include_days'),
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        
        self.val_dataset = BrainToTextDataset(
            trial_indicies=val_trials,
            split='test',
            days_per_batch=None,
            n_batches=None,
            batch_size=self.args['dataset']['batch_size'],
            must_include_days=None,
            random_seed=self.args['dataset']['seed'],
            feature_subset=feature_subset
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.args['dataset'].get('num_dataloader_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        # Initialize multi-reference wrapper
        self.multiref_wrapper = MultiReferenceWrapper(
            merger_names=self.args.get('merger_names', ['COT_CAUGHT']),
            max_variants=self.args.get('max_variants_per_trial', 4)
        )
        self.logger.info(f"Multi-reference enabled with mergers: {self.args.get('merger_names', ['COT_CAUGHT'])}")
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.learning_rate_scheduler = self._create_lr_scheduler()
        
        # Loss functions
        self.ctc_loss_standard = torch.nn.CTCLoss(blank=0, reduction='none', zero_infinity=False)
        self.ctc_loss_multiref = MultiReferenceCTCLoss(blank=0, reduction='none', zero_infinity=False)
        
        # Load checkpoint if specified
        if self.args.get('init_from_checkpoint'):
            self._load_checkpoint(self.args['init_checkpoint_path'])
        
        # Move model to device
        self.model.to(self.device)
        
        self.logger.info("Initialization complete")
    
    def _create_optimizer(self):
        """Create optimizer with parameter groups."""
        # Separate parameters for different learning rates
        day_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'day' in name:
                day_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': other_params, 'lr': self.args['lr_max']},
            {'params': day_params, 'lr': self.args.get('lr_max_day', self.args['lr_max'])}
        ]
        
        return torch.optim.AdamW(
            param_groups,
            betas=(self.args.get('beta0', 0.9), self.args.get('beta1', 0.999)),
            eps=self.args.get('epsilon', 0.1),
            weight_decay=self.args.get('weight_decay', 0.001)
        )
    
    def _create_lr_scheduler(self):
        """Create cosine learning rate scheduler."""
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']
        lr_warmup_steps = self.args.get('lr_warmup_steps', 1000)
        
        def lr_lambda(step, min_ratio, decay_steps, warmup_steps):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            if step < decay_steps:
                progress = float(step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_ratio, min_ratio + (1 - min_ratio) * cosine_decay)
            return min_ratio
        
        lr_lambdas = [
            lambda step: lr_lambda(step, lr_min/lr_max, lr_decay_steps, lr_warmup_steps),
            lambda step: lr_lambda(step, lr_min/lr_max, lr_decay_steps, lr_warmup_steps),
        ]
        
        return LambdaLR(self.optimizer, lr_lambdas)
    
    def transform_data(self, features, n_time_steps, mode='train'):
        """Apply augmentations to data."""
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]
        
        if mode == 'train':
            # White noise
            if self.transform_args.get('white_noise_std', 0) > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']
            
            # Constant offset
            if self.transform_args.get('constant_offset_std', 0) > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']
            
            # Random cut
            if self.transform_args.get('random_cut', 0) > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut
        
        # Gaussian smoothing
        if self.transform_args.get('smooth_data', False):
            features = gauss_smooth(
                inputs=features,
                device=self.device,
                smooth_kernel_std=self.transform_args['smooth_kernel_std'],
                smooth_kernel_size=self.transform_args['smooth_kernel_size']
            )
        
        return features, n_time_steps
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        train_losses = []
        val_losses = []
        val_PERs = []
        
        val_steps_since_improvement = 0
        start_time = time.time()
        
        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()
            
            # Move data to device
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)
            
            # Add multi-reference targets
            batch = self.multiref_wrapper.add_multiref_targets(batch)
            
            with torch.autocast(device_type="cuda", enabled=self.args.get('use_amp', True), dtype=torch.bfloat16):
                # Transform data
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')
                
                # Compute adjusted lengths
                patch_size = self.args['model']['patch_size']
                patch_stride = self.args['model']['patch_stride']
                adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
                
                # Forward pass
                logits = self.model(features, day_indicies)
                log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2])
                
                # Multi-reference CTC loss
                loss = self.ctc_loss_multiref(
                    log_probs,
                    batch['seq_class_ids_variants'],
                    adjusted_lens,
                    batch['variant_lengths']
                )
                loss = loss.mean()
            
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
            
            train_losses.append(loss.detach().item())
            
            # Logging
            if i % self.args.get('batches_per_train_log', 200) == 0:
                self.logger.info(
                    f"Batch {i}: loss={loss.item():.4f}, grad_norm={float(grad_norm):.2f}"
                )
            
            # Validation
            if i % self.args.get('batches_per_val_step', 2000) == 0 or i == self.args['num_training_batches'] - 1:
                val_metrics = self.validate()
                val_losses.append(val_metrics['avg_loss'])
                val_PERs.append(val_metrics['avg_PER'])
                
                self.logger.info(
                    f"Validation at batch {i}: loss={val_metrics['avg_loss']:.4f}, PER={val_metrics['avg_PER']:.4f}"
                )
                
                # Save best model
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    val_steps_since_improvement = 0
                    
                    if self.args.get('save_best_checkpoint'):
                        self._save_checkpoint(
                            os.path.join(self.args['checkpoint_dir'], 'best_checkpoint'),
                            val_metrics['avg_PER'],
                            val_metrics['avg_loss']
                        )
                else:
                    val_steps_since_improvement += 1
                
                # Early stopping
                if self.args.get('early_stopping') and val_steps_since_improvement >= self.args.get('early_stopping_val_steps', 20):
                    self.logger.info(f"Early stopping at batch {i}")
                    break
            
            if i >= self.args['num_training_batches'] - 1:
                break
        
        duration = time.time() - start_time
        self.logger.info(f"Training complete in {duration/60:.2f} minutes")
        self.logger.info(f"Best validation PER: {self.best_val_PER:.4f}")
        
        # Save final model
        if self.args.get('save_final_model'):
            self._save_checkpoint(
                os.path.join(self.args['checkpoint_dir'], f'final_checkpoint_batch_{i}'),
                val_PERs[-1] if val_PERs else 0,
                val_losses[-1] if val_losses else 0
            )
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_PERs': val_PERs,
            'best_PER': self.best_val_PER
        }
    
    def validate(self):
        """Run validation."""
        self.model.eval()
        
        total_loss = 0
        total_edit_distance = 0
        total_seq_length = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['input_features'].to(self.device)
                labels = batch['seq_class_ids'].to(self.device)
                n_time_steps = batch['n_time_steps'].to(self.device)
                phone_seq_lens = batch['phone_seq_lens'].to(self.device)
                day_indicies = batch['day_indicies'].to(self.device)
                
                # Skip days not in validation set
                day = day_indicies[0].item()
                if self.args['dataset']['dataset_probability_val'][day] == 0:
                    continue
                
                with torch.autocast(device_type="cuda", enabled=self.args.get('use_amp', True), dtype=torch.bfloat16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')
                    
                    patch_size = self.args['model']['patch_size']
                    patch_stride = self.args['model']['patch_stride']
                    adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
                    
                    logits = self.model(features, day_indicies)
                    
                    # Use standard CTC for validation (canonical targets)
                    loss = self.ctc_loss_standard(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        adjusted_lens,
                        phone_seq_lens
                    )
                    loss = loss.mean()
                
                total_loss += loss.item()
                n_batches += 1
                
                # Compute PER
                for idx in range(logits.shape[0]):
                    decoded = torch.argmax(logits[idx, :adjusted_lens[idx], :], dim=-1)
                    decoded = torch.unique_consecutive(decoded)
                    decoded = decoded[decoded != 0].cpu().numpy()
                    
                    true_seq = labels[idx, :phone_seq_lens[idx]].cpu().numpy()
                    
                    total_edit_distance += F.edit_distance(list(decoded), list(true_seq))
                    total_seq_length += len(true_seq)
        
        return {
            'avg_loss': total_loss / max(n_batches, 1),
            'avg_PER': total_edit_distance / max(total_seq_length, 1)
        }
    
    def _save_checkpoint(self, path, per, loss):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.learning_rate_scheduler.state_dict(),
            'val_PER': per,
            'val_loss': loss
        }
        torch.save(checkpoint, path)
        
        # Save config
        config_path = os.path.join(self.args['checkpoint_dir'], 'args.yaml')
        OmegaConf.save(self.args, config_path)
        
        self.logger.info(f"Saved checkpoint to {path}")
    
    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint.get('val_PER', float('inf'))
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        self.logger.info(f"Loaded checkpoint from {path}")


def main():
    parser = argparse.ArgumentParser(description='Train with Multi-Reference CTC Loss')
    parser.add_argument('--config', type=str, default='rnn_args_multiref.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    print(f"Starting multi-reference training with config: {args.config}")
    print(f"Mergers: {config.get('merger_names', ['COT_CAUGHT'])}")
    
    # Create trainer and run
    trainer = MultiRefBrainToTextTrainer(config)
    trainer.initialize()
    results = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Best validation PER: {results['best_PER']:.4f}")


if __name__ == '__main__':
    main()
