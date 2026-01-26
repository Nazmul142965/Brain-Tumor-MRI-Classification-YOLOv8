# Brain Tumor MRI Classification using YOLOv8-Nano

This project implements a high-efficiency framework for classifying brain tumors from MRI scans into four categories: Glioma, Meningioma, Pituitary, and No Tumor.

## 🚀 Performance
- **Validation Accuracy:** 98.5%
- **Architecture:** YOLOv8-Nano (yolov8n-cls)
- **Dataset:** Masoud Nickparvar Brain Tumor MRI Dataset (7,023 images)
- **Hardware:** NVIDIA T4 GPU

## 📂 Dataset
The dataset is available on Kaggle. [Link to Masoud Nickparvar Dataset]
Please follow the `data_preprocessing` script to organize the images into the required YOLOv8 structure.

## 🛠️ Usage
1. Install requirements: `pip install ultralytics`
2. Run preprocessing: `python data_preprocessing/split_data.py`
3. Train model: `python training/train.py`
