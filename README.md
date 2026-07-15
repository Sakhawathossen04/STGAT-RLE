# STGAT-RLE: A Spatio-Temporal Graph Attention Network with Cross-Modal Gating for Proactive Radio Link Failure Prediction in MANETs.

**Official implementation** of the paper Under Review at **ACML 2026**.

> **STGAT-RLE** models dynamic spatial neighborhood dependencies via Graph Attention Networks (GAT) and temporal RF dynamics via Transformer encoders, fused through a novel **feature-wise cross-modal gating** mechanism. This enables selective modulation between spatial contagion and temporal degradation signals for highly calibrated proactive link failure prediction in Mobile Ad Hoc Networks (MANETs).

[![arXiv](https://img.shields.io/badge/arXiv-Preprint-b31b1b.svg)](https://arxiv.org/abs/...) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/stgat-rle-a-spatio-temporal-graph-attention/manet-link-failure-prediction)](https://paperswithcode.com/sota/manet-link-failure-prediction?p=stgat-rle-a-spatio-temporal-graph-attention)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C)
![PyG](https://img.shields.io/badge/PyTorch_Geometric-2.x-FF6B6B)

---

##  Key Contributions 

1. **Feature-wise Cross-Modal Gating**: Bidirectional mutual modulation between GAT spatial representations and Transformer temporal representations — outperforms simple concatenation in calibration (ECE 0.0218 vs 0.0713).
2. **Comprehensive Spatio-Temporal Benchmarking**: Strong performance against STGCN, DCRNN, Graph WaveNet, and strong temporal baselines.
3. **Imbalance-Aware Training**: Focal Loss + Weighted Random Sampling.
4. **Explainability**: Gradient-based feature attribution + GAT attention analysis. Neighborhood failure rate and topological features are dominant transferable predictors.
5. **Cross-Simulator Generalization**: Zero-shot evaluation on independent NS-3 dataset.
6. **Proactive Rerouting Impact**: 94.9% reduction in packet-equivalent overhead.

**Paper PDF**: [paper_STGAT_RLE_for_ACML_final.pdf](paper_STGAT_RLE_for_ACML_final.pdf) (included in repo)

---

##  Project Structure

```
STGAT-RLE/
├── data/                          # Preprocessed OMNeT++ + NS-3 datasets
├── notebooks/
│   └── stgat_rle_full_pipeline.ipynb   # End-to-end reproduction
├── src/
│   ├── models/                    # STGAT-RLE, baselines (STGCN, DCRNN, Graph WaveNet, etc.)
│   ├── data/                      # Dynamic graph construction, feature engineering
│   ├── utils/                     # Training, evaluation, calibration, explainability
│   └── loss/                      # Focal Loss implementation
├── configs/                       # Hyperparameters (CFG)
├── results/                       # Tables, plots, ablation results
├── scripts/                       # Training scripts
├── requirements.txt
├── README.md
└── paper_STGAT_RLE_for_ACML_final.pdf
```

---

##  Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/STGAT-RLE.git
cd STGAT-RLE
pip install -r requirements.txt
```

**Key Dependencies**:
- PyTorch + Torch-Geometric
- PyTorch Lightning (optional)
- scikit-learn, pandas, networkx, matplotlib, seaborn
- captum (for gradient attribution)

### 2. Data Preparation

Download/preprocess OMNeT++ and NS-3 datasets (scripts provided). Expected structure includes RF KPIs, positions, and link failure labels.

### 3. Training

```bash
python scripts/train.py --config configs/stgat_rle.yaml
```

**Main Configuration Highlights** :
- Lookback `T=32`
- Neighborhood `k=5`
- GAT: 2 layers, multi-head attention
- Transformer: 3 layers, `d_model=128`
- Cross-modal gating + residual fusion
- Focal Loss (`α=0.75`, `γ=2.0`)
- AdamW + Cosine Annealing

---

##  Results (OMNeT++ Test Set)

| Model              | F1     | Precision | Recall | AUC    | ECE↓   |
|--------------------|--------|-----------|--------|--------|--------|
| Graph WaveNet     | 0.9647 | 0.9641    | 0.9654 | 0.9838 | 0.0846 |
| STGCN             | 0.9621 | 0.9557    | **0.9686** | 0.9855 | 0.0526 |
| **STGAT-RLE (Ours)** | **0.9596** | 0.9674 | 0.9519 | **0.9842** | **0.0262** |
| BiLSTM (best temporal) | 0.9513 | 0.9626 | 0.9403 | 0.9763 | **0.0239** |

- **Best calibration** among spatio-temporal models.
- Strong cross-simulator performance on NS-3 (zero-shot).

**Full tables** and ablation studies available in `results/`.

---

##  Ablation Studies

**Cross-Modal Gating** significantly improves calibration over concatenation.

**Spatial Encoders**: GAT shows highest recall.

Detailed results in the paper (Section 6) and notebook.

---

##  Explainability

- Gradient-based feature importance (Captum).
- Neighborhood failure rate (`ρ_fail`) and node degree are top operationally transferable features.
- **Time** feature (simulator leakage) explicitly analyzed and discounted.

See `notebooks/stgat_rle_full_pipeline.ipynb` for SHAP/Gradient visualizations.

---

##  Proactive Rerouting Simulation

Static cost-impact evaluation shows **94.9%** reduction in overhead vs. reactive AODV-style recovery.

---

##  Cross-Simulator Validation

Trained only on OMNeT++, evaluated zero-shot on NS-3. Competitive generalization (Graph WaveNet slightly better on NS-3).

---


##  Contact & Contributing

- Open issues for questions/bugs.
- Pull requests welcome for extensions (real-world datasets, federated learning, adaptive k, etc.).
- Full reproducibility materials (datasets, trained weights) available upon reasonable request.

---
