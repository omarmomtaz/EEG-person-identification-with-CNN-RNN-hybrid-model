# EEG Person Identification using CNN + GRU

## Project Overview

This project implements a **CNN + GRU hybrid deep learning model** for person identification using EEG (electroencephalogram) signals from the PhysioNet Motor Movement/Imagery Dataset.

**Goal**: Classify which subject (1-109) a given EEG segment belongs to based on unique brainwave patterns.

---

## 📁 Project Structure

```
EEG_Person_Identification/
├── Notebook_1_Preprocessing.ipynb      # Data download, filtering, segmentation
├── Notebook_2_CNN_GRU_Model.ipynb      # Model training and evaluation
├── Notebook_3_Performance_Report.ipynb  # Metrics, confusion matrix, analysis
├── Notebook_4_Visualizations.ipynb      # (Optional) Spectrograms, t-SNE
└── README.md                            # This file
```

---

## 🔧 Technical Specifications

### Dataset
- **Source**: PhysioNet EEG Motor Movement/Imagery Dataset
- **Subjects**: 109 healthy individuals
- **Task**: Motor imagery (imagining hand/feet movements)
- **Runs Used**: Motor imagery runs only (4, 6, 8, 10, 12, 14)

### Preprocessing
| Parameter | Value |
|-----------|-------|
| Sampling Rate | 160 Hz |
| Channels | 17 (motor cortex: Fc3, Fc1, Fcz, Fc2, Fc4, C5, C3, C1, Cz, C2, C4, C6, Cp3, Cp1, Cpz, Cp2, Cp4) |
| Band-pass Filter | 0.5 - 45 Hz |
| Segment Duration | 2 seconds (320 samples) |
| Overlap | 50% |
| Normalization | Z-score per segment |

### Model Architecture
```
Input: (17 channels, 320 samples)
    │
    ▼
┌─────────────────────────────────┐
│  CNN Feature Extractor          │
│  ├─ Conv1D(32) + BN + ReLU + Pool│
│  ├─ Conv1D(64) + BN + ReLU + Pool│
│  └─ Conv1D(128)+ BN + ReLU + Pool│
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  GRU Temporal Encoder           │
│  ├─ 2 Bidirectional GRU Layers  │
│  └─ Hidden Size: 128            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Classification Head            │
│  ├─ Dense(256) + BN + ReLU      │
│  └─ Dense(109) - Output         │
└─────────────────────────────────┘
```

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam (weight decay: 1e-4) |
| Loss Function | Cross-Entropy |
| Early Stopping | Patience: 10 epochs |
| LR Scheduler | ReduceLROnPlateau |

---

## 🚀 How to Run

### Step 1: Open Google Colab
Go to [Google Colab](https://colab.research.google.com/)

### Step 2: Run Notebooks in Order

#### Notebook 1: Preprocessing (~20-30 minutes)
1. Create a new notebook and paste the code from `Notebook_1_Preprocessing`
2. Run all cells
3. This will:
   - Download the PhysioNet dataset (~500 MB)
   - Process and segment EEG data
   - Save preprocessed data to `/content/processed_data/`

#### Notebook 2: Model Training (~30-60 minutes with GPU)
1. **IMPORTANT**: Enable GPU in Colab: `Runtime > Change runtime type > GPU`
2. Create a new notebook and paste the code from `Notebook_2_CNN_GRU_Model`
3. Run all cells
4. This will:
   - Load preprocessed data
   - Train the CNN + GRU model
   - Save trained model to `/content/models/`

#### Notebook 3: Performance Report (~5 minutes)
1. Create a new notebook and paste the code from `Notebook_3_Performance_Report`
2. Run all cells
3. This generates:
   - Confusion matrices
   - Accuracy, F1, precision, recall metrics
   - Per-class analysis
   - Performance discussion

#### Notebook 4: Visualizations (Optional, ~10 minutes)
1. Create a new notebook and paste the code from `Notebook_4_Visualizations`
2. Run all cells
3. This generates:
   - EEG signal plots
   - Spectrograms
   - t-SNE feature embeddings
   - Power spectrum analysis

---

## 📊 Expected Results

Based on the architecture and dataset, you can expect:

| Metric | Expected Range |
|--------|----------------|
| Top-1 Accuracy | 85-95% |
| Top-5 Accuracy | 95-99% |
| F1 Score (Weighted) | 0.85-0.95 |
| Training Time | 30-60 min (GPU) |

---

## 📂 Output Files

### Preprocessed Data (`/content/processed_data/`)
- `X_train.npy`, `X_test.npy` - EEG segments
- `y_train.npy`, `y_test.npy` - Subject labels
- `metadata.pkl` - Configuration info

### Models (`/content/models/`)
- `best_model.pt` - Best weights
- `complete_model.pt` - Full model checkpoint
- `results.pkl` - Evaluation results
- `training_history.png` - Training curves

### Reports (`/content/report/`)
- `final_report.txt` - Summary report
- `confusion_matrix.png` - Confusion matrices
- `f1_analysis.png` - F1 score analysis
- `classification_report.txt` - Per-class metrics
- `discussion.txt` - Performance discussion

### Visualizations (`/content/visualizations/`)
- `eeg_signals.png` - Raw EEG plots
- `spectrograms.png` - Time-frequency analysis
- `tsne_embeddings.png` - Feature embeddings
- `power_spectrum.png` - Frequency analysis

---

## 💾 Saving to Google Drive (Recommended)

To persist data between Colab sessions, add this code after preprocessing:

```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copytree('/content/processed_data', '/content/drive/MyDrive/EEG_Project/data')
shutil.copytree('/content/models', '/content/drive/MyDrive/EEG_Project/models')
shutil.copytree('/content/report', '/content/drive/MyDrive/EEG_Project/report')
```

---

## 🔬 Key Findings

1. **EEG as Biometric**: EEG signals contain unique patterns that can identify individuals with high accuracy.

2. **Architecture Choice**: The CNN extracts spatial-frequency features while GRU captures temporal dynamics—both essential for EEG analysis.

3. **Motor Cortex Focus**: Using only motor cortex channels (17) instead of all 64 channels reduces noise while maintaining discriminative power.

4. **Subject Variability**: Some subjects are easier to identify due to more distinctive neural patterns.

---

## 📚 References

1. **Dataset**: Schalk, G., et al. "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System." IEEE TBME, 2004.
   - PhysioNet URL: https://physionet.org/content/eegmmidb/1.0.0/

2. **Related Work**: 
   - "EEG-based biometric system using deep learning" (various papers)
   - "Motor imagery classification with CNN-RNN" (various papers)

---

## ⚠️ Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size to 32
   - Use fewer subjects for testing first

2. **Download Errors**
   - PhysioNet might be slow; retry downloads
   - Some subjects may fail; code handles this gracefully

3. **Low Accuracy**
   - Ensure GPU is enabled
   - Check that data loaded correctly
   - Try longer training (more epochs)

4. **Colab Disconnection**
   - Save to Google Drive frequently
   - Use Colab Pro for longer sessions

---

## 👥 Team Submission Checklist

- [ ] Notebook 1: Preprocessing (loading, filtering, segmenting EEG)
- [ ] Notebook 2: CNN + RNN model (training and evaluation)  
- [ ] Notebook 3: Performance report including:
  - [ ] Confusion matrix
  - [ ] Accuracy and F1-score
  - [ ] Discussion of model performance
- [ ] Notebook 4: (Optional) Visualization of spectrograms or t-SNE

---

## 📝 License

This project uses the PhysioNet EEG Motor Movement/Imagery Dataset, which is available under the PhysioNet Credentialed Health Data License.
