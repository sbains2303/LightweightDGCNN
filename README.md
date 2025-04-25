# 🚁 A Sensor-Based Model for Subtle UAV GPS Spoofing 

This repository contains the main codebase for a project focused on **subtle spoofing detection of UAV (Unmanned Aerial Vehicle) missions** using deep learning. The primary aim is to detect anomalies or patterns in sensor behaviors by modeling sensor entries as graph-structured data and evaluating various neural architectures.

> ⚠️ This repository **does not include the UAV sensor dataset**, as it is private. The purpose of this codebase is to demonstrate the main project structure and logic used in this research.

---

## 🧠 Project Overview

The approach involves converting UAV sensor data from Gazebo mission simulations into graph-based representations and applying a range of graph neural network (GNN) models and a convolutional neural network (CNN) model to perform classification. On top of this, four unique models (namely `lightweightDGCNN.py`, `attention.py`, `glu.py`, and `highway.py`) are developed to investigate whether they are able to outperform the others in terms of accuracy. The models were evaluated on their ability to generalize across different mission types and route complexities as well as latency outputs.

---

## 📂 Repository Structure
```
├── README.md                      # Project overview (this file)
├── routes/                        # Contains UAV mission plans
├── src/ 
│   ├── models/                    # Model definitions
│   │   ├── cnn.py
│   │   ├── gat.py
│   │   ├── gcn.py
│   │   ├── gnn.py
│   │   ├── lightweightDGCNN.py
│   │   ├── attention.py
│   │   ├── highway.py
│   │   └── glu.py
│   ├── prep/                      # Data preprocessing and augmentation
│   │   ├── clean.py               # Algorithm to find missing values 
│   │   ├── impute.py              # KNN imputation
│   │   └── prepare_augment.py     # Noise-based data augmentation
│   ├── compare.py                 # Code for comparing model performance
│   ├── configure.py               # Centralised configuration (e.g. batch size, learning rate)
│   ├── gradAttack.py              # Adversarial gradual attack script
│   ├── graphCons.py               # Graph construction with temporal and spatial edges
│   └── mission.py                 # MAVSDK mission flight control and sensor collection
```
---

## ⚙️ Core Functional Components

### 🔧 Configuration
All experimental settings are managed centrally via `configure.py`, including:
- Batch size  
- Learning rate  
- Weight decay  
- Number of epochs  
- Device selection (CPU/GPU)
- Dropout etc

---

### 🧹 Data Processing & Preparation
Located in `src/prep`, this stage ensures that the dataset is clean, complete, and augmented for robust model training:

- `clean.py` – Scans the dataset to identify and standardize missing values (sets them as NaN).
- `impute.py` – Applies K-Nearest Neighbours (KNN) imputation to fill missing values based on feature similarity.
- `prepare_augment.py` – Augments training data by injecting controlled noise to improve model generalization and resilience.

---

### 🧠 Model Architectures
Located in `src/models`, this project includes implementations of:

- **CNN** – Convolutional network  
- **GCN / GAT / GNN** – Classical graph neural networks  
- **DGCNN Variants** – Dynamic Graph CNNs with lightweight, attention, highway, and GLU variants  

---

### 📊 Evaluation & Comparison
`compare.py` handles logging and comparison of evaluation metrics (e.g., accuracy, F1-score).
- A custom early stopping mechanism is applied based on validation loss trends.
- The latencies of each model is measured in this file.

---

### ⚔️ Adversarial Attack Implementation
- `gradAttack.py` provides a sinusoidal drift attack which is deployed to simulate subtle adversaries.

---

## 📊 Performance Evaluation

The following summarizes the performance metrics for each model in terms of **test accuracy** and **latency**:

| Model                | Test Accuracy (%) | Latency (ms) |
|----------------------|-------------------|--------------|
| **CNN**              | 56                | 7            |
| **GAT**              | 88                | 52           |
| **GCN**              | 80                | 19           |
| **GNN**              | 69                | 8            |
| **Lightweight DGCNN**| 93                | 72           |
| **Attention DGCNN**  | 81                | 105          |
| **Highway DGCNN**    | 75                | 60           |
| **GLU DGCNN**        | 87                | 151          |

- **Test Accuracy**: Measures how well the model generalizes to unseen data.
- **Latency**: Time taken for the model to process one inference (in milliseconds).

This table provides an overview of each model's **classification performance** as well as its computational efficiency, which is crucial for real-time deployment in UAV systems.

---

### 🚀 Future Work
*Ongoing work is focused on transitioning from graph-level classification to node-level classification, where individual sensor entries are classified to detect spoofing. This shift aims to extend the approach towards real-world deployment by improving granularity in anomaly detection.*
