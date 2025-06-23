# 🧠 MNIST Handwritten Digit Recognition Web App

A machine learning project that recognizes handwritten digits using a trained Random Forest Classifier and provides a real-time prediction interface with Gradio.

## ✨ Features

- 🖼️ Draw digits (0–9) using an interactive sketchpad
- 🧠 Predicts the digit using a trained ML model
- 📊 Model trained on MNIST dataset from OpenML
- 🔍 Evaluated using confusion matrix and classification report
- 🌐 Gradio-powered web app for easy deployment

---

## 🧪 Machine Learning Details

- **Dataset**: [MNIST 784](https://www.openml.org/d/554)
- **Models used**:
  - SGDClassifier (hinge loss)
  - RandomForestClassifier (final)
- **Best Accuracy**: ≥ 95%
- **Error Analysis**: Common misclassifications (like 9 → 4) were explored
- **Model Saved As**: `rf_mnist_model.joblib`

