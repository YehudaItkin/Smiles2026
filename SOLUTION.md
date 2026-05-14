# Solution Report — SMILES-2026 Hallucination Detection

## Reproducibility

### Environment

```bash
conda create -n smiles2026 python=3.12 -y
conda activate smiles2026
pip install -r requirements.txt
```

### Running

```bash
python solution.py
```

This produces `results.json` and `predictions.csv`. Execution takes ~3–4 minutes on MPS (Apple Silicon) or a T4 GPU.

---

## Final Solution Description

### Modified Files

1. **`aggregation.py`** — multi-layer geometric feature extraction
2. **`probe.py`** — CatBoost classifier (replaces baseline MLP)
3. **`splitting.py`** — 5-fold stratified cross-validation

### Approach

#### Feature Engineering (`aggregation.py`)

The baseline uses only the last token of the final transformer layer (896 dimensions), discarding ~96% of available information. Our approach extracts 1038 features capturing the model's internal "confidence trajectory" across all 25 layers:

- **Last-token representation** from the final layer (896 dim) — preserves the baseline signal
- **Per-layer geometric features** (100 dim): L2 norms, mean activations, standard deviations, and max absolute values of the last token at each of the 25 layers. These track how activation magnitudes evolve through the network — hallucinated responses show different norm trajectories.
- **Inter-layer cosine similarities** (24 dim): cosine similarity between consecutive layers' last-token representations, measuring "representation drift." Larger drift between layers may indicate the model is less certain.
- **Token-level mean-pooling features** (6 dim): for layers 0, 12, 24 — L2 norm of the mean-pooled vector and its cosine similarity with the last token. Captures how the last token diverges from the overall sequence representation.
- **Positional contrast** (2 dim): cosine similarity and norm ratio between the first and last real tokens of the final layer.
- **Cross-layer summary statistics** (10 dim): mean, std, max, min of layer norms; argmax layer; norm gradient and curvature; mean, std, min of cosine similarities.

#### Classifier (`probe.py`)

Replaced the baseline single-hidden-layer MLP with **CatBoost** (gradient boosting with ordered boosting):

- **CatBoost** with `rsm=0.3` (Random Subspace Method): on each boosting step, only 30% of features are used — acts as stochastic feature selection, critical for preventing overfitting with 1038 features and only 447 training samples per fold.
- **Ordered boosting**: CatBoost computes gradients using a permutation-based scheme where each sample's gradient is computed only on data it "hasn't seen," reducing target leakage.
- **`auto_class_weights="Balanced"`** handles the 70/30 class imbalance.
- **Internal threshold tuning** via 3-fold CV inside `fit()`, since the final probe (for `predictions.csv`) does not call `fit_hyperparameters()`.

#### Cross-Validation (`splitting.py`)

5-fold stratified CV with a validation split carved from each fold's training data. Each fold trains on ~447 samples (vs 483 in the baseline single split). The final probe trains on all 689 samples.

### What Contributed Most

The **geometric feature engineering** was the single most impactful change — particularly the layer-wise norms and inter-layer cosine similarities. These capture the "trajectory" of representations through the network, which differs between hallucinated and truthful responses.

### Results

| Metric | Baseline | Our Solution |
|--------|----------|-------------|
| Test Accuracy | 70.10% | **72.28%** |
| Test AUROC | N/A | **73.15%** |
| Test F1 | 82.42% | **82.61%** |

---

## Experiments and Failed Attempts

### Experiment 1: PCA + Ensemble (LogReg + CatBoost)
- **Setup**: 1038 features → PCA(150) → VotingClassifier(LogReg + CatBoost)
- **Result**: Test AUROC 72.25%, Accuracy 71.55%
- **Issue**: PCA may remove discriminative directions that don't align with highest-variance components. The ensemble added complexity without proportional benefit.

### Experiment 2: Geometric features only (no raw hidden states)
- **Setup**: 142 geometric features only → PCA(80) → ensemble
- **Result**: Test AUROC 70.80%, Accuracy 70.82%
- **Issue**: Geometric features alone were too lossy — barely beat the baseline. The raw 896-dim representation carries signal that summary statistics cannot fully capture.

### Experiment 3: Multi-layer raw features (2830 dim)
- **Setup**: Last-token from layers 12 and 24 + mean-pooled final layer + geometric = 2830 features → PCA(40)
- **Result**: Test AUROC 68.65%, Accuracy 70.83%
- **Issue**: PCA(40) was too aggressive for 2830 features — important signal lost. More raw features ≠ better when PCA is the bottleneck.

### Experiment 4: PCA(100) with moderate regularization
- **Setup**: 1038 features → PCA(100) → VotingClassifier(LogReg C=0.3 + CatBoost)
- **Result**: Test AUROC 71.73%, Accuracy 70.10%
- **Issue**: Middle ground didn't outperform either extreme. LogReg regularization too strong (C=0.3) suppressed useful signal.

### Experiment 5: Extended features (entropy + eigenvalues + trajectory + Mahalanobis)
- **Setup**: 1085 features (added activation entropy per layer, prompt/response contrast, trajectory curvature, eigenvalue features from token covariance matrices) + Mahalanobis distance from class distributions as probe features
- **Result**: Test AUROC 71.96%, Accuracy 72.71%
- **Issue**: Additional features added noise. Eigenvalues from small token counts were unstable. Activation entropy (softmax of raw hidden states) was a crude proxy. Prompt/response boundary heuristic (60% split) was imprecise without token IDs. More features = more spurious correlations with 689 samples.

### Experiment 6: Extended aggregation features without Mahalanobis
- **Setup**: 1085 features (same as Exp 5) + simple CatBoost (no Mahalanobis)
- **Result**: Test AUROC 72.58%, Accuracy 72.28%
- **Issue**: Removing Mahalanobis helped AUROC slightly, but still worse than the simpler 1038-feature set. Confirmed that the additional features were noise, not signal.

### Key Takeaway
CatBoost directly on scaled features (no PCA) with `rsm=0.3` outperformed all PCA-based and extended-feature approaches. CatBoost's built-in feature selection via random subspace method is more effective than PCA for this task, because PCA optimizes for variance rather than discriminative power. With only 689 training samples, **feature quality matters more than quantity** — adding noisy features hurts even with built-in feature selection.
