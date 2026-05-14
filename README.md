# SMILES-2026 Hallucination Detection — Solution

Detecting hallucinations in Qwen2.5-0.5B responses using hidden state analysis and logit-based confidence features.

**Author:** Yehuda (Igor) Itkin

## Results

| Metric | Baseline | Solution |
|--------|----------|----------|
| Test Accuracy | 70.10% | **73.00%** |
| Test AUROC | — | **73.89%** |
| Test F1 | 82.42% | **83.12%** |

## Predictions

`predictions.csv` is included in the repository.

Direct link: [predictions.csv](https://raw.githubusercontent.com/YehudaItkin/Smiles2026/main/predictions.csv)

## Approach

1. **Feature extraction** (1046 dim): raw last-layer hidden states (896) + per-layer geometric features (norms, cosine similarities, statistics — 142) + logit-based confidence features (entropy, max probability, top-1 vs top-2 gap — 8)
2. **Classifier**: 5-seed CatBoost ensemble with `rsm=0.3` (stochastic feature selection)
3. **Evaluation**: 5-fold stratified cross-validation

Key insight: loading the LM head weight separately and computing `logits = hidden_states @ W.T` gives direct access to the model's prediction confidence — a fundamentally different signal from hidden-state geometry.

See [SOLUTION.md](SOLUTION.md) for the full report with 11 experiments.

## Reproducing

### Setup

```bash
git clone https://github.com/YehudaItkin/Smiles2026.git
cd Smiles2026
conda create -n smiles2026 python=3.12 -y
conda activate smiles2026
pip install -r requirements.txt
```

### Run

```bash
python solution.py
```

Produces `results.json` and `predictions.csv`. Takes ~5-6 minutes on MPS (Apple Silicon) or T4 GPU.

## Repository Structure

```
Smiles2026/
├── aggregation.py      # Feature extraction (modified)
├── probe.py            # CatBoost classifier (modified)
├── splitting.py        # 5-fold stratified CV (modified)
├── solution.py         # Main script (fixed, do not edit)
├── evaluate.py         # Evaluation loop (fixed, do not edit)
├── model.py            # Loads Qwen2.5-0.5B (fixed, do not edit)
├── data/
│   ├── dataset.csv     # 689 labelled samples
│   └── test.csv        # 100 unlabelled test samples
├── predictions.csv     # Test predictions (100 samples)
├── results.json        # Evaluation metrics
├── SOLUTION.md         # Full report with experiments
└── requirements.txt    # Dependencies (includes catboost)
```
