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

This produces `results.json` and `predictions.csv`. Execution takes ~5–6 minutes on MPS (Apple Silicon) or a T4 GPU. The LM head weight is loaded once for logit-based features (adds ~30s on first batch).

---

## Final Solution Description

### Modified Files

1. **`aggregation.py`** — multi-layer geometric + logit-based feature extraction (1046 dim)
2. **`probe.py`** — multi-seed CatBoost ensemble (replaces baseline MLP)
3. **`splitting.py`** — 5-fold stratified cross-validation

### Approach

#### Feature Engineering (`aggregation.py`)

The baseline uses only the last token of the final transformer layer (896 dimensions), discarding ~96% of available information. Our approach extracts **1046 features** across two categories: hidden-state geometry and model confidence signals.

**Hidden-state geometry (1038 dim):**

- **Last-token representation** from the final layer (896 dim) — preserves the baseline signal.
- **Per-layer geometric features** (100 dim): L2 norms, mean activations, standard deviations, and max absolute values of the last token at each of the 25 layers. These track how activation magnitudes evolve through the network — hallucinated responses show different norm trajectories.
- **Inter-layer cosine similarities** (24 dim): cosine similarity between consecutive layers' last-token representations, measuring "representation drift."
- **Token-level mean-pooling features** (6 dim): for layers 0, 12, 24 — L2 norm of the mean-pooled vector and its cosine similarity with the last token.
- **Positional contrast** (2 dim): cosine similarity and norm ratio between the first and last real tokens of the final layer.
- **Cross-layer summary statistics** (10 dim): mean, std, max, min of layer norms; argmax layer; norm gradient and curvature; mean, std, min of cosine similarities.

**Logit-based confidence features (8 dim):**

The key insight: hidden states encode the model's internal representation, but **logits encode its predictions**. Hallucination is fundamentally about incorrect predictions — directly measuring prediction confidence should carry stronger signal.

We load the LM head weight matrix (`lm_head.weight`, shape `vocab_size × 896`) separately and compute `logits = hidden_states[-1] @ W.T` for the last 10 real tokens. From these logits we extract:
- **Max log-probability**: mean, min, std across tokens (3 features) — how confident is the model?
- **Token entropy**: mean, max, std across tokens (3 features) — how uncertain is the model?
- **Top-1 vs top-2 gap**: mean, min (2 features) — how decisive is the model between its top two predictions?

#### Classifier (`probe.py`)

Replaced the baseline single-hidden-layer MLP with a **multi-seed CatBoost ensemble**:

- **5 CatBoost models** with different random seeds (42, 123, 456, 789, 2024), predictions averaged. This reduces variance — individual CatBoost runs on 1046 features with 447 training samples have high variance.
- **CatBoost** with `rsm=0.3` (Random Subspace Method): on each boosting step, only 30% of features are used — acts as stochastic feature selection.
- **Ordered boosting**: CatBoost computes gradients using a permutation-based scheme where each sample's gradient is computed only on data it "hasn't seen," reducing target leakage. Critical with only 689 samples.
- **`auto_class_weights="Balanced"`** handles the 70/30 class imbalance.
- **Internal threshold tuning** via 3-fold CV inside `fit()`, since the final probe (for `predictions.csv`) does not call `fit_hyperparameters()`.

#### Cross-Validation (`splitting.py`)

5-fold stratified CV with a validation split carved from each fold's training data. Each fold trains on ~447 samples. The final probe trains on all 689 samples (since `idx_non_test` covers all indices in k-fold).

### What Contributed Most

1. **Logit-based features** — the single biggest improvement. Directly measuring model confidence (entropy, max probability, top gap) from logits provided a fundamentally new signal that hidden-state geometry alone doesn't capture. (+0.7% AUROC over geometric-only CatBoost)
2. **Multi-seed ensemble** — averaging 5 CatBoost models with different seeds reduced prediction variance.
3. **CatBoost replacing MLP** — with 689 samples, gradient boosting with `rsm=0.3` generalizes much better than neural networks. Ordered boosting further reduces overfitting.
4. **Layer-wise geometric features** — norms and cosine similarities across all 25 layers capture the "confidence trajectory" through the network.

### Final Results

| Metric | Baseline | Our Solution | Improvement |
|--------|----------|-------------|-------------|
| Test Accuracy | 70.10% | **73.00%** | +2.90% |
| Test AUROC | N/A | **73.89%** | — |
| Test F1 | 82.42% | **83.12%** | +0.70% |

---

## Experiments and Failed Attempts

### Experiment 1: PCA + Ensemble (LogReg + CatBoost)
- **Setup**: 1038 features → PCA(150) → VotingClassifier(LogReg + CatBoost)
- **Result**: Test AUROC 72.25%, Accuracy 71.55%
- **Issue**: PCA removes discriminative directions that don't align with highest-variance components.

### Experiment 2: Geometric features only (no raw hidden states)
- **Setup**: 142 geometric features only → PCA(80) → ensemble
- **Result**: Test AUROC 70.80%, Accuracy 70.82%
- **Issue**: Geometric features alone too lossy — barely beat baseline. Raw 896-dim representation carries signal that summary statistics cannot fully capture.

### Experiment 3: Multi-layer raw features (2830 dim)
- **Setup**: Last-token from layers 12 and 24 + mean-pooled final layer + geometric = 2830 → PCA(40)
- **Result**: Test AUROC 68.65%, Accuracy 70.83%
- **Issue**: PCA(40) too aggressive for 2830 features — important signal lost.

### Experiment 4: PCA(100) with moderate regularization
- **Setup**: 1038 features → PCA(100) → VotingClassifier(LogReg C=0.3 + CatBoost)
- **Result**: Test AUROC 71.73%, Accuracy 70.10%
- **Issue**: LogReg regularization too strong suppressed useful signal.

### Experiment 5: Extended features (entropy + eigenvalues + trajectory + Mahalanobis)
- **Setup**: 1085 features (activation entropy per layer, prompt/response contrast, trajectory curvature, eigenvalue features from token covariance matrices) + Mahalanobis distance
- **Result**: Test AUROC 71.96%, Accuracy 72.71%
- **Issue**: Additional features added noise. Eigenvalues unstable from small token counts. Prompt/response boundary heuristic (60% split) imprecise without token IDs.

### Experiment 6: CatBoost direct (no PCA, geometric + raw only)
- **Setup**: 1038 features → StandardScaler → CatBoost(rsm=0.3)
- **Result**: Test AUROC 73.15%, Accuracy 72.28%
- **Outcome**: Best result at this point. CatBoost's built-in feature selection via rsm outperformed all PCA-based approaches.

### Experiment 7: Logit-based features + multi-seed ensemble (FINAL)
- **Setup**: 1046 features (1038 geometric + 8 logit-based) → StandardScaler → 5× CatBoost ensemble
- **Result**: Test AUROC 73.89%, Accuracy 73.00%
- **Outcome**: Best overall. Logit features added a fundamentally new signal (model confidence), multi-seed averaging reduced variance.

### Experiment 8: RDE (KernelPCA + MinCovDet Mahalanobis, inspired by lm-polygraph)
- **Setup**: 1046 features + KernelPCA(rbf, 50 components) + MCD per class → Mahalanobis distance as extra features
- **Result**: Test AUROC 70.76%, Accuracy 73.73%
- **Issue**: KernelPCA with RBF on 1046 features with ~200 samples per class was numerically unstable. Accuracy improved slightly but AUROC degraded significantly — probability calibration worse despite better binary predictions.

### Experiment 9: Mean-pooled middle layer (896 raw dim)
- **Setup**: 1942 features (added mean-pooled layer 12 representation) + RDE
- **Result**: Test AUROC 68.91%, Accuracy 70.39%
- **Issue**: 896 extra raw dimensions overwhelmed the useful features with noise.

### Experiment 10: Stacking ensemble (LDA + SVM + CatBoost)
- **Setup**: 1050 features → PCA(80) → LDA + SVM(RBF) + CatBoost trained separately → OOF meta-features → LogReg meta-classifier
- **Result**: Test AUROC 73.47%, Accuracy 71.69%
- **Issue**: PCA(80) bottleneck lost discriminative information before base models. Much less overfitting (train 88.5% vs 98.8%) but worse test generalization. Stacking overhead didn't justify marginal gains.

### Experiment 11: Extended logit features (trend + response-focused)
- **Setup**: 1050 features (added entropy/probability trend first-half vs second-half, response-focused stats) → 5× CatBoost
- **Result**: Test AUROC 72.54%, Accuracy 71.84%
- **Issue**: Trend and response-focused features were noise — the heuristic split (first/second half of tokens) doesn't reliably separate prompt from response without token IDs.

### Summary Table

| # | Method | Features | AUROC | Accuracy |
|---|--------|----------|-------|----------|
| 1 | PCA(150) + LR+CatBoost | 1038 | 72.25% | 71.55% |
| 2 | Geometric only + PCA(80) | 142 | 70.80% | 70.82% |
| 3 | Multi-layer raw + PCA(40) | 2830 | 68.65% | 70.83% |
| 4 | PCA(100) + LR+CatBoost | 1038 | 71.73% | 70.10% |
| 5 | Extended (eigen+traj) + Mahalanobis | 1085 | 71.96% | 72.71% |
| 6 | CatBoost direct, rsm=0.3 | 1038 | 73.15% | 72.28% |
| **7** | **Logits + 5-seed CatBoost** | **1046** | **73.89%** | **73.00%** |
| 8 | RDE (KernelPCA+MCD) | 1046 | 70.76% | 73.73% |
| 9 | Mean-pooled mid layer + RDE | 1942 | 68.91% | 70.39% |
| 10 | Stacking (LDA+SVM+CB) | 1050 | 73.47% | 71.69% |
| 11 | Extended logits (trend) | 1050 | 72.54% | 71.84% |

### Key Takeaways

1. **Feature quality > quantity** with 689 samples. Adding noisy features hurts even with built-in feature selection.
2. **Logit-based features** (entropy, max probability, top gap) provide a fundamentally different signal from hidden-state geometry — the model's **prediction confidence** vs its **internal representation structure**.
3. **CatBoost with rsm=0.3** is the right classifier for high-dimensional small-sample problems — better than PCA+LogReg, PCA+ensemble, stacking, or MLP.
4. **Multi-seed ensemble** is a cheap way to reduce variance with no downside.
5. **Density-based methods** (Mahalanobis, RDE) from lm-polygraph were theoretically promising but numerically unstable with only ~200 samples per class.
6. **Stacking (LDA+SVM+CatBoost)** reduced overfitting but the PCA bottleneck before base models lost too much signal. Different classifier families didn't add enough diversity to overcome information loss.
7. **Heuristic token splitting** (first/second half, last 30%) fails without actual token IDs to separate prompt from response.
