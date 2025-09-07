Diabetic Retinopathy Detection with Explainability
Overview

This project focuses on detecting and classifying Diabetic Retinopathy (DR) severity from retinal fundus images using deep learning. DR is a complication of diabetes that affects the eyes, and early detection is crucial to prevent vision loss.

The model leverages Convolutional Neural Networks (CNNs) such as ResNet, EfficientNet, or Vision Transformers to classify the severity of the disease into multiple stages. Additionally, explainability techniques (e.g., Grad-CAM, LIME, SHAP) are integrated to highlight the regions of the retina that influenced the model's predictions, ensuring transparency and trust in the results.

Dataset

Source: Kaggle Diabetic Retinopathy Dataset
 (Raw & Processed Data)

Description:

Thousands of high-resolution retina fundus images.

Labeled by disease severity levels:

0: No DR

1: Mild

2: Moderate

3: Severe

4: Proliferative DR

Objectives

Build a deep learning model for automatic classification of DR severity.

Compare performance of different architectures (ResNet, EfficientNet, Vision Transformer).

Integrate explainability methods to visualize important regions influencing the decision.

Provide insights for medical practitioners to assist in diagnosis.

Methodology

Preprocessing

Image resizing and normalization.

Data augmentation to handle imbalance and improve generalization.

Model Training

Train CNN-based architectures (ResNet/EfficientNet/ViT).

Evaluate with metrics: Accuracy, F1-score, AUC, and confusion matrix.

Explainability

Use Grad-CAM, SHAP, or LIME to generate heatmaps.

Provide interpretability for clinical adoption.

Tools and Technologies

Languages: Python

Frameworks: TensorFlow / PyTorch

Libraries: NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn

Explainability: Captum (PyTorch), SHAP, LIME, Grad-CAM implementations

Expected Outcomes

A trained model capable of classifying diabetic retinopathy severity.

Visual explanations highlighting critical regions of retina images.

A reproducible workflow for medical AI research.

Future Work

Extend the model to detect other retinal diseases.

Deploy as a web application for real-world usage.

Optimize model for low-resource settings (mobile or edge devices).