# Utah_BrainToText25
Algorithms and Software for submission to Brain-to-text '25

We propose ALAN-D, *A Language Algorithm for Neural Decoding*.

ALAN-D is based off of the baseline RNN speech decoding algorithm designed by UCD.

## Required Software

<!-- - Docker [https://www.docker.com/products/docker-desktop/]() -->
- VSCode [https://code.visualstudio.com/]()
- Git [https://git-scm.com/downloads]()
- Docker Desktop [https://www.docker.com/products/docker-desktop/]()

### If on Windows:
- [WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

## Initial Setup
 
1. Make sure you have Docker, VSCode, and Git installed.
2. In VSCode, install the **Dev Containers** extension
    + See [this guide](https://code.visualstudio.com/docs/getstarted/extensions) for details.
3. Build a dev container using `CTRL+SHIFT+P` and search for the command `Dev Containers: Rebuild without Cache`
4. A new window will open up, the dev container will begin installing. This may take up to **20 minutes** of build time.
5. Once the dev container has installed and opened, install the `nejm-brain-to-text` baseline as an editable python packge.

```bash
pip install -e phase1/nejm-brain-to-text
```

6. Install the data that is intended to be trained on or processed by `nejm-brain-to-text`.
    + The easiest way to do this is to just download it from this google drive link (its roughly ~23 GB)
    + Otherwise follow the instructions in `phase1/nejm-brain-to-text/README.md`
        + The file `phase1/nejm-brain-to-text/download_data.py` also downloads it


## Roadmap

### Phase 1: Foundational Work

See [Phase 1 README](./phase1/README.md)

-   [ ] **Replicate Baseline Results** (https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text/tree/main)
    -   [x] **Environment Setup**:
        -   [x] Set up `docker` environemnt for model training. (only useful for Windows, CHPC is on Fedora linux)
    -   [x] **Data Preparation**:
        -   [x] Download the complete dataset from the provided Dryad link.
        -   [x] Unzip and organize the data into the correct directory structure.
    -   [ ] **Secure CHPC Access**:
        -   [x] Acquired access through _1_ lab thus far
        -   [x] Demonstrate successful env setup on CHPC for model training
    -   [ ] **Model Training & Evaluation**:
        -   [ ] Run Model training (phase1/nejm-brain-to-text/model_training/README.md)
        -   [ ] Train the baseline RNN model to achieve a phoneme error rate of approximately 10.1%.
        -   [ ] Run the full evaluation pipeline with the pretrained model to verify results.

### Phase 2: Advancing the State-of-the-Art

-   [ ] **Implementation of Winning 2024 Algorithm**
    -   [ ] **Analyze the 2024 winning algorithm:**
        -   [ ] Review the "Brain-to-Text Benchmark '24: Lessons Learned" paper to understand the key components of the winning entries.
        -   [ ] Pay close attention to the use of Transformer architectures, which were a key component of the top-performing entries.
    -   [ ] **Implement the Transformer-based model:**
        -   [ ] Create a new model file (e.g., `transformer_model.py`) that implements a Transformer-based decoder.
        -   [ ] Adapt the existing `GRUDecoder` class in `rnn_model.py` as a starting point.
    -   [ ] **Train and evaluate the Transformer model:**
        -   [ ] Modify the `train_model.py` script to use your new Transformer model.
        -   [ ] Adjust hyperparameters in `rnn_args.yaml` for the Transformer architecture.
        -   [ ] Evaluate the performance of your Transformer model using `evaluate_model.py` and compare it to the baseline.

### Phase 3: Experimentation

-   [ ] **Sentiment Analysis Integration:**
    -   [ ] Train a sentiment analysis model on the text output of the baseline decoder.
    -   [ ] Use the predicted sentiment as an additional feature to be fed into a second-stage predictor.
    -   [ ] The second-stage predictor would then try to make a more accurate prediction by leveraging the predicted intention. (may not be most accurate though)
-   [ ] **Expert System with Large Language Models (LLMs):**
    -   [ ] Investigate the use of LLMs that have been "grokked" or fine-tuned on specific domains of knowledge.
    -   [ ] Extract features from the neural data that might be relevant to different expert domains.
    -   [ ] Use these features to gate the outputs of different expert LLMs, effectively creating a mixture of experts model.

## Reference

- [Brain To Text 2024 Lessons Learned](./reference/BraintotextBenchmark24_LessonsLearned.pdf)
- [Recent UC Davis paper](./reference/Wairagkar_et_al-2025-Nature.pdf)
- Winning Brain To Text 24' algorithm: https://github.com/fwillett/speechBCI
