# Comparative Analysis of CNNs for Automated Brain Tumor Detection in MRI Images

[![Project Status: Completed](https://img.shields.io/badge/Status-Completed-success.svg)](#performance)
[![Model: YOLOv8-Nano](https://img.shields.io/badge/Model-YOLOv8--Nano-blue.svg)](#architecture)
[![Accuracy: 98.5%](https://img.shields.io/badge/Accuracy-98.5%25-orange.svg)](#performance)

This repository contains the official implementation of a high-efficiency brain tumor classification framework. By utilizing **YOLOv8-Nano**, this project achieves state-of-the-art diagnostic accuracy while maintaining a lightweight profile suitable for real-time clinical applications.

---

## 🚀 Performance Summary
Our model was evaluated against four distinct tumor classes (Glioma, Meningioma, Pituitary, and No Tumor) using the Masoud Nickparvar dataset.

* **Peak Validation Accuracy:** `98.5%`
* **Parameters:** `1.4 Million` (Nano Variant)
* **Inference Speed:** Real-time capable on standard hardware
* **Evaluation Metric:** Cross-Entropy Loss optimization

---

## 🏗️ Repository Structure
```text
📁 assets/             # Architecture diagrams, Confusion Matrix, and Accuracy plots
📁 data_preprocessing/ # Automated 80/20 data splitting script (split_data.py)
📁 models/             # Trained weights (best.pt)
📁 training_logic/     # Core training script (train.py)
📄 data.yaml           # YOLOv8 dataset configuration
📄 predict.py          # Script for single-image inference
📄 requirements.txt    # Project dependencies
