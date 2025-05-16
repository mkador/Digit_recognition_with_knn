# 🔢 Handwritten Digits Recognition using K-Nearest Neighbors (KNN)

This project presents a machine learning solution for recognizing handwritten digits using the **K-Nearest Neighbors (KNN)** algorithm. It leverages the `digits` dataset from `scikit-learn`, which contains grayscale images of digits (0–9), and demonstrates a complete pipeline from data preprocessing to model evaluation and visualization.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Highlights](#project-highlights)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Author](#author)

---

## 📌 Overview

The goal of this notebook is to build an efficient and interpretable digit classification model using KNN. The notebook walks through:

- Loading and visualizing the digit images
- Reshaping and splitting data
- Hyperparameter tuning using `GridSearchCV`
- Evaluating the model with accuracy and confusion matrix
- Displaying prediction results alongside actual labels

---

## 🧾 Dataset

- **Source:** [`sklearn.datasets.load_digits()`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- **Description:** 8x8 grayscale images representing handwritten digits (0–9)
- **Total samples:** 1,797
- **Features:** 64 (flattened pixel values)
- **Target classes:** 10

---

## 🌟 Project Highlights

- ✅ End-to-end digit recognition pipeline
- 🔍 Optimized K-value selection using `GridSearchCV`
- 📈 Achieved accuracy of **~98.9%**
- 🖼️ Visual comparison of true vs. predicted labels
- 📊 Detailed performance report with confusion matrix

---

## 💻 Installation

To run this project locally, ensure you have Python 3 and the following libraries installed:

pip install numpy matplotlib scikit-learn


## 🚀 Usage
### 1. Clone the repository:
`git clone https://github.com/mkador/Digits_recognition_knn.git
cd Digits_recognition_knn
`
### 2. Launch the notebook:
jupyter notebook Digits_recognition_knn.ipynb
### 3. Run all cells to:
- Load the dataset
- Train and evaluate the KNN model
- Visualize prediction outcomes

## 📊 Results
- Best k value: Found via Grid Search
- Test Accuracy: ~98.9%
- Confusion Matrix: Included for performance diagnostics
- Prediction Samples: Displayed alongside actual labels

## 🛠 Technologies Used
- Python 3
- Jupyter Notebook
- scikit-learn
- Matplotlib
- NumPy




