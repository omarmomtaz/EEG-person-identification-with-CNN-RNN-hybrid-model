# EEG Person Identification 🧠

[![Python](https://img.shields.io/badge/Python-3.10-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Dataset: PhysioNet](https://img.shields.io/badge/Dataset-PhysioNet-blue?style=for-the-badge)](https://physionet.org/content/eegmmidb/1.0.0/)

**99.87% Accuracy | CNN-GRU Hybrid Model for Biometric EEG Identification**

Built a state-of-the-art deep learning system to identify 109 individuals from EEG signals during motor imagery tasks. **This has huge potential for secure biometrics, personalized BCIs, helping paralyzed people control devices with thoughts, neurological research, and more.**

[🚀 Live Inference Demo](https://your-flask-app-link-here) | [📄 Full Project Report PDF](reports/eeg-person-identification-report.pdf) | [🎥 60s Demo Video](https://youtu.be/your-video-link) | [GitHub Repo](https://github.com/omarmomtaz/eeg-person-identification)

---

## 🧩 The Challenge

Traditional biometrics (fingerprints, faces) fail in high-security or hands-free scenarios. **EEG signals offer unique "brain fingerprints"**—but raw data is noisy, high-dimensional, and subject-specific.

**My Solution:** A hybrid **CNN-GRU model** that extracts spatial-frequency features (via CNN) and temporal dynamics (via bidirectional GRU) from 10 motor cortex channels. Achieves near-perfect identification on the PhysioNet dataset.

---

## ✨ Key Features

- **Robust Preprocessing Pipeline**: Band-pass filtering (0.5–45 Hz), 2s epochs with 50% overlap, Z-score normalization.
- **Advanced Architecture**: 3-layer 1D CNN (32→64→128 filters) + 2-layer BiGRU (128 units) + dense head for 109-class softmax.
- **Training Optimizations**: Adam optimizer, early stopping, LR scheduler, dropout (0.3–0.5) to prevent overfitting.
- **Top-Tier Performance**: 99.87% Top-1 accuracy, 99.96% Top-3, weighted F1=0.9987.
- **Rich Visualizations**: t-SNE clusters, spectrograms, CNN feature maps, PSD analysis—proving discriminative power.
- **Deployable**: CPU-trained, ready for real-time inference (Flask/Streamlit integration).

---

## 🛠️ Tech Stack

| Component       | Tools                          |
|-----------------|--------------------------------|
| **Language**    | Python 3.10                   |
| **Framework**   | PyTorch 2.0                   |
| **Data**        | NumPy, Pandas, MNE-Python     |
| **Viz**         | Matplotlib, Seaborn           |
| **Utils**       | Scikit-learn, TorchMetrics    |

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/omarmomtaz/eeg-person-identification.git
cd eeg-person-identification
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python src/preprocess.py --data-dir data/raw
```

### 3. Train the Model
```bash
python src/train.py --epochs 50 --batch-size 64
```

### 4. Run Inference
```bash
python src/infer.py --model-path models/best_model.pth --eeg-file sample_eeg.npy
```
**Output:** Predicted subject ID with confidence score.

**Full Dataset:** [Download from PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) (109 subjects, 160 Hz).

---

## 📊 Results

### Overall Performance
| Metric              | Value    |
|---------------------|----------|
| **Top-1 Accuracy**  | **99.87%** |
| **Top-3 Accuracy**  | 99.96%  |
| **Top-5 Accuracy**  | 99.97%  |
| **Top-10 Accuracy** | 99.99%  |
| **Weighted F1**     | 0.9987  |
| **Macro F1**        | 0.9988  |
| **Precision (W)**   | 0.9988  |
| **Recall (W)**      | 0.9987  |

**Test Set:** 15,926 samples (20% split, stratified).

### Per-Subject Insights
- **Mean Per-Class Accuracy**: 99.88%
- **Best Subject**: 100.00% (e.g., Subject 1)
- **Worst Subject**: 97.96% (Subject 75)
- **Error Rate**: 0.13% (20 misclassifications)

![t-SNE Feature Clusters](images/tsne_clusters.png)
*Distinct subject clusters from GRU embeddings—clear separation for biometrics.*

![Average Spectrograms](images/avg_spectrograms.png)
*Unique frequency "fingerprints" per subject.*

---

## 🌍 Real-World Impact

**This has huge potential for:**
- **Secure Biometrics**: Tamper-proof identity verification (beyond passwords/fingerprints) for high-stakes access.
- **Personalized BCIs**: Tailored brain-computer interfaces for gaming, productivity, and adaptive tech.
- **Helping Paralyzed People Control Devices with Thoughts**: Enabling thought-controlled prosthetics, wheelchairs, and communication aids for motor impairments.
- **Neurological Research**: Unlocking insights into unique brain signatures for studying disorders, cognition, and personalized medicine.

---

## 📈 Visualizations & Insights

- **Raw EEG Signals**: Subject-specific patterns emerge.
- **CNN Feature Maps**: Shows how conv layers capture motor cortex dynamics.
- **Power Spectral Density (PSD)**: Delta/Theta/Alpha/Beta/Gamma differences across individuals.

![CNN Feature Maps](images/cnn_feature_maps.png)
*Intermediate activations visualizing spatial-temporal learning.*

**See all in the full report!**

---

## 📘 Deep Dive: Full Project Report

Want the complete methodology, ablation studies, error analysis, and future work (e.g., attention mechanisms, ICA preprocessing)?

[📥 Download PDF (12 pages)](reports/eeg-person-identification-report.pdf)

Includes:
- Data pipeline details
- Architecture diagram
- Training curves
- Interpretability analysis

---

## 🔮 Future Work

- **Real-World BCI**: Cross-session transfer learning.
- **Augmentation**: Time-warping + GANs for robustness.
- **Interpretability**: SHAP/LIME for EEG features.
- **Deployment**: Edge inference on Raspberry Pi for wearables.

---

## 🤝 Contributing

Open to collaborations on BCI/neurotech! Fork, PR, or reach out.

**Built by Omar Momtaz**  
*Undergraduate AI Engineer | Robotics & Generative AI*  
[LinkedIn](https://www.linkedin.com/in/omar-momtaz-/) | [GitHub](https://github.com/omarmomtaz) | [Email](omarmomtaz310@gmail.com)

---

*Last updated: February 2026 | Stars & forks welcome! ⭐*
```
