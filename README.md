# STGAT-RLE: Spatio-Temporal Graph Attention Transformer for Radio Link Failure Estimation in MANETs

> Predicting radio link failures in Mobile Ad hoc Networks using a novel fusion of Graph Attention Networks and Transformer-based temporal encoding.....

---

##  File

| File | Description |
|------|-------------|
| `radio-link-estimation-model.ipynb` | Complete end-to-end pipeline: data loading, EDA, feature engineering, model training, evaluation, ablation, explainability, and rerouting simulation |

---

##  Overview

Radio link failures in **Mobile Ad hoc Networks (MANETs)** cause packet loss and costly reactive rerouting. This project proposes **STGAT** — a Spatio-Temporal Graph Attention Transformer — that fuses two complementary views of the network:

- **Spatial branch (GAT):** captures inter-node dependencies by building a k-nearest-neighbor graph of transmitters and applying multi-head Graph Attention Convolution.
- **Temporal branch (Transformer):** encodes the historical RF signal sequence for each node using a Transformer Encoder with sinusoidal positional encoding.

The fused representation is passed through an MLP classifier trained with **Focal Loss** to handle the severe class imbalance between active and failed links.

---

##  Pipeline

```
Phase 1  →  Setup, dependencies, dataset loading
Phase 1D →  Exploratory Data Analysis (SNR, PowerReception, class distribution)
Phase 2  →  MANET graph construction & visualisation (NetworkX)
Phase 3  →  Feature engineering + outlier removal + train/val/test split
Phase 4  →  Class imbalance handling (SMOTETomek)
Phase 5  →  Sequential DataLoaders with WeightedRandomSampler
Phase 6A →  Baseline model definitions (LSTM, BiLSTM, Transformer)
Phase 6C →  Graph dataset construction + STGAT model definition
Phase 7  →  Training & evaluation utilities
Phase 7B →  Training all models (XGBoost, RF, LSTM, BiLSTM, Transformer, STGAT)
Phase 7C →  Results table + comparison plots + confusion matrix + training curves
Phase 8  →  Ablation study (w/o GAT / w/o Transformer / w/o Focal Loss)
Phase 9  →  SHAP explainability (XGBoost TreeExplainer)
Phase 10 →  Proactive rerouting cost simulation
Phase 11 →  Model checkpointing
```

---

##  Models Compared

| Model | Type |
|-------|------|
| XGBoost | Gradient Boosted Trees |
| Random Forest | Ensemble |
| LSTM | Recurrent |
| BiLSTM | Bidirectional Recurrent |
| Temporal Transformer | Attention-based sequence model |
| **STGAT (Proposed)** | **Spatio-Temporal Graph Attention Transformer** |

All models are evaluated on **F1, Precision, Recall, AUC-ROC, Average Precision, and Error Rate**.

---

##  Key Features

- **Graph-based spatial modeling** — k-NN transmitter graph with 2-layer GATConv (`add_self_loops=False` for compatibility)
- **Focal Loss** (α = 0.75, γ = 2.0) to address class imbalance
- **SMOTETomek** oversampling on training data
- **WeightedRandomSampler** for balanced mini-batches
- **CosineAnnealingLR** scheduler with early stopping
- **SHAP** feature importance via XGBoost TreeExplainer
- **Proactive rerouting simulation** quantifying packet-cost savings

---

##  Requirements

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install imbalanced-learn shap networkx xgboost scikit-learn
```

> Tested with **PyTorch 2.1.0**. Ensure `torch-geometric` version is compatible with your PyTorch + CUDA version. See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

---

##  Dataset

The notebook expects CSV files at:

```
/kaggle/input/radio-link-estimation-data-preprocessing/*.csv
```

Each CSV should contain columns including:

- `IdTransmitter`, `DatasetID`
- `PositionTransmission0/1/2`, `PositionReception0/1/2`
- `SNR`, `PowerReception`
- `Link` (binary label: 0 = active, 1 = failed)

To run locally, update `CFG['data_path']` to point to your local data directory.

---

##  Configuration

All hyperparameters are controlled through the `CFG` dictionary at the top of the notebook:

```python
CFG = {
    'train_ratio'   : 0.8,
    'val_ratio'     : 0.1,
    'k_neighbors'   : 5,       # k-NN graph for GAT
    'time_steps'    : 32,      # sequence length
    'batch_size'    : 256,
    'epochs'        : 150,
    'lr'            : 3e-4,
    'patience'      : 15,
    'gat_hidden'    : 64,
    'gat_heads'     : 4,
    'trans_d_model' : 128,
    'trans_layers'  : 3,
    ...
}
```

---

##  Outputs

The notebook generates the following files upon completion:

| File | Description |
|------|-------------|
| `eda_distributions.pdf` | SNR & PowerReception class distributions |
| `eda_correlation.pdf` | Feature correlation with link failure |
| `graph_snapshot.pdf` | MANET topology visualisation |
| `results_comparison.pdf` | Bar chart comparing all models |
| `results_comparison.csv` | Numerical results table |
| `stgat_cm.pdf` | STGAT confusion matrix |
| `training_curves.pdf` | Loss & F1 curves per epoch |
| `ablation_study.pdf` | Ablation F1 comparison |
| `ablation_results.csv` | Ablation numerical results |
| `shap_summary.pdf` | SHAP feature importance plot |
| `proactive_rerouting.pdf` | Cost comparison: reactive vs. proactive |
| `stgat_best.pth` | STGAT model weights |
| `lstm_best.pth` | LSTM model weights |
| `bilstm_best.pth` | BiLSTM model weights |
| `transformer_best.pth` | Temporal Transformer weights |
| `xgboost_best.json` | XGBoost model |
| `rf_best.pkl` | Random Forest model |
| `scaler.pkl` | Fitted StandardScaler |

---

##  Proactive Rerouting

The trained STGAT is used to simulate a **proactive rerouting policy**:

- **Reactive cost** — every actual failure incurs 50 packet-equivalent units.
- **Proactive cost** — missed detections (FN) cost 50 units; false alarms (FP) cost only 5 units.
- The model's early predictions are shown to significantly reduce total network cost vs. purely reactive rerouting.

---
  
## 📬 Contact

For questions or collaboration, please open an issue in this repository.
