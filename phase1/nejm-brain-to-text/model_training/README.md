# Multi-Reference CTC Training for Brain-to-Text

This module enables training with **multi-reference CTC loss**, which accepts multiple valid pronunciations per trial. Currently configured with the **COT_CAUGHT** merger (AA ↔ AO), which is a valid dialect feature in Western and Canadian American English.

## File Placement

Place all files in your `model_training` folder:

```
Utah_BrainToText25/
└── phase1/
    └── nejm-brain-to-text/
        └── model_training/
            ├── dataset.py                      # existing
            ├── rnn_model.py                    # existing  
            ├── rnn_trainer.py                  # existing
            ├── data_augmentations.py           # existing
            ├── train_model.py                  # existing
            ├── rnn_args.yaml                   # existing
            │
            ├── multi_pronunciation_lexicon.py  # ← NEW
            ├── multi_reference_ctc_loss.py     # ← NEW
            ├── multi_reference_dataset.py      # ← NEW
            ├── train_model_multiref.py         # ← NEW
            ├── rnn_args_multiref.yaml          # ← NEW
            ├── run_multiref.slurm              # ← NEW
            └── __init__.py                     # ← NEW (optional)
```

## Installation Requirements

Make sure you have these packages installed:
```bash
pip install g2p_en torch torchaudio omegaconf h5py numpy
```

## Running Training

### Option 1: SLURM (recommended for cluster)

1. Edit `run_multiref.slurm`:
   - Update `--mail-user=id@utah.edu` with your email
   - Adjust time/memory if needed
   - Uncomment module loads for your cluster

2. Submit the job:
```bash
cd Utah_BrainToText25/phase1/nejm-brain-to-text/
sbatch model_training/run_multiref.slurm
```

### Option 2: Direct execution

```bash
cd Utah_BrainToText25/phase1/nejm-brain-to-text/model_training
python train_model_multiref.py --config rnn_args_multiref.yaml
```

## Configuration

Edit `rnn_args_multiref.yaml` to modify:

```yaml
# Multi-reference settings
use_multi_reference: true
merger_names:
  - COT_CAUGHT      # AA ↔ AO (valid American English merger)
max_variants_per_trial: 4

# Output location
output_dir: trained_models/multiref_cot_caught
checkpoint_dir: trained_models/multiref_cot_caught/checkpoint
```

## What It Does

With COT_CAUGHT merger enabled, words like:
- "caught" → accepts both [K **AA** T] and [K **AO** T]
- "thought" → accepts both [TH **AA** T] and [TH **AO** T]  
- "water" → accepts both [W **AA** T ER] and [W **AO** T ER]

The model learns to match whichever pronunciation is closer to its prediction, reducing false errors from this dialect variation.

## Expected Impact

Based on error analysis, the COT_CAUGHT merger addresses ~34 substitution errors (AA↔AO confusions). This is a conservative approach using only linguistically valid pronunciation variants.

## Comparing Results

After training, compare validation PER:
- Baseline model (standard CTC): Check your existing results
- Multi-ref model (this training): Check `trained_models/multiref_cot_caught/`

The multi-reference model should show improvement on words containing AA/AO sounds.
