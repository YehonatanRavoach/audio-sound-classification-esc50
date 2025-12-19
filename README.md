# Audio Sound Classification on ESC-50 (PyTorch)

End-to-end deep learning project for **environmental sound classification** using the ESC-50 dataset.  
The project implements a customized **ResNet-18 CNN** trained on **log-mel spectrograms**, achieving **99.33% test accuracy** through extensive data augmentation and careful model optimization.

---

## Project Overview

Environmental sound classification is a challenging task due to high intra-class variability and noise in real-world audio signals.  
This project focuses on building a **robust and scalable deep learning pipeline** for classifying **50 environmental sound classes**, grouped into 5 high-level categories.

The pipeline covers the full machine learning lifecycle:
- Exploratory Data Analysis (EDA)
- Audio preprocessing and feature extraction
- Data augmentation
- Model training and evaluation
- Performance analysis and validation

---

## Dataset

The project uses the **ESC-50 dataset**, a labeled benchmark dataset containing:
- 2,000 audio recordings
- 50 sound classes (e.g., dog bark, rain, siren)
- 5-second WAV clips
- Balanced class distribution

Dataset source:  
https://github.com/karolpiczak/ESC-50

---

## Pipeline & Methodology

### 1. Exploratory Data Analysis (EDA)
- Verified class balance across all 50 categories
- Validated audio duration consistency (~5 seconds)
- Visualized raw waveforms and mel-spectrograms
- Analyzed spectral characteristics (energy, spectral centroid)

### 2. Preprocessing & Feature Engineering
- Converted raw audio signals into **log-mel spectrograms**
- Applied padding, normalization, and resizing to **128×128**
- Cached preprocessed features for efficient reuse
- Implemented extensive **data augmentation**:
  - Gaussian noise injection
  - Time stretching
  - Pitch shifting

### 3. Modeling & Training
- **Architecture**:
  - Base model: ResNet-18 (CNN)
  - Adapted first convolution layer for single-channel input
  - Replaced final fully connected layer for 50-class classification
- **Training setup**:
  - Optimizer: Adam
  - Loss function: CrossEntropyLoss
  - Batch size: 64
  - Early stopping and loss monitoring
- **Dataset split**:
  - Training / validation / test split
  - Test set kept fully isolated to prevent data leakage

---

## Results

| Dataset | Total Samples | Train | Validation | Test | Train Acc | Val Acc | Test Acc |
|-------|---------------|-------|------------|------|-----------|---------|----------|
| Original | 2,000 | 1,190 | 510 | 300 | 97.56% | 66.27% | — |
| Original + Augmented | 8,143 | 6,921 | 1,222 | 300 | 97.20% | 98.28% | **99.33%** |

Final model achieved **99.33% accuracy on the test set**, with strict separation between training and evaluation data to avoid leakage.

---

## Design Decisions: Why CNN and Not Transformers?

Although Vision Transformers (ViT) were considered, a CNN-based approach was selected due to:
- Limited dataset size (2,000 original samples)
- Short audio duration (~5 seconds)
- Lower computational cost
- Superior empirical performance using ResNet-18

The CNN architecture provided excellent generalization without the overhead of transformer-based models.

---

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/YehonatanRavoach/audio-sound-classification-esc50.git
cd ESC50-DeepLearning
```
2. Install required packages:
```bash
   torch>=2.0
   torchvision>=0.15
   numpy>=1.24
   pandas>=1.5
   librosa>=0.10
   matplotlib>=3.7
   seaborn>=0.12
   opencv-python>=4.7
   scikit-learn>=1.2
```
3. Download the ESC-50 dataset:
```bash
   python download_data.py
```
4. Run the notebooks in order:
  * 01_EDA.ipynb
  * 02_Preprocessing.ipynb
  * 03_Modeling.ipynb

## 📁 Project Structure
```bash
project-root/
│── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── config.py
│── download_data.py
│── README.md
│── .gitignore
```

## Technologies & Skills Demonstrated
* Python
* PyTorch
* Convolutional Neural Networks (CNN)
* Audio signal processing
* Log-mel spectrograms
* Data augmentation
* Model evaluation & validation
* Experimental analysis
* End-to-end ML pipeline design


