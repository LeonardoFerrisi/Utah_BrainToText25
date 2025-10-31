# figure_2 (script)

This folder contains `figure_2.py`, a plain-Python conversion of
`figure_2.ipynb` (from the `analyses` notebook set).

Requirements
- Python 3.8+ (tested syntax with 3.10)
- numpy, matplotlib, g2p_en
- the package `nejm_b2txt_utils` must be importable (relative imports in this repo assume you run from the project root)

How to run
From the `analyses` folder run:

```powershell
python figure_2.py
```

Options:
- `--pkl PATH` : path to the pickled data (default `../data/t15_copyTask.pkl`)
- `--trial N` : trial index to show for the example (default 500)
- `--no-show` : do not display plots (useful for CI/headless)

Notes
- The script preserves the logic from the notebook, including a dependency on
  `LOGIT_PHONE_DEF` and `calculate_aggregate_error_rate` from
  `nejm_b2txt_utils.general_utils`.
- If plots do not render, try running the script in an environment where a display is available or use `--no-show` to skip showing plots.
