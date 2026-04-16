# Advancing Cataract Detection and Screening Through Multi-Modal and Explainable Artificial Intelligence Systems

This project implements an AI-based cataract detection system using fundus images and patient metadata based on the ODIR-5K dataset.

## Features
- Multi-modal learning (image + metadata)
- MobileNetV2 (transfer learning)
- Stratified K-Fold Cross Validation
- Grad-CAM visualization for explainability
- Binary classification (Cataract vs Normal)

## Methodology
The system extracts deep features from fundus images using MobileNetV2 and combines them with structured metadata. The fused features are passed through fully connected layers for classification.

## Results
- Accuracy: ~96–97%
- ROC-AUC: ~0.94

## Dataset
ODIR-5K dataset (public dataset for ocular disease recognition)

## Technologies
- Python
- TensorFlow / Keras
- OpenCV
- Scikit-learn

## Future Work
- Deployment as web application
- Real-time clinical screening
- Enhanced multi-modal fusion
- ## Author
- Aamna Girnari
