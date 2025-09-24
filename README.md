# Heart-Disease-Prediction-System - Machine Learning
Built an end-to-end machine learning pipeline in Python (pandas, scikit-learn, matplotlib, seaborn) to predict the likelihood of heart disease from medical data with different learning models 
The dataset used is the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (Cleveland subset, 303 samples).

---

## 📂 Project Overview

The pipeline includes:
- **Data cleaning** (duplicates, outlier capping)
- **Categorical encoding** (one-hot for medical categorical attributes)
- **Feature engineering** (age grouping, heart stress ratio, risk score)
- **Feature selection** (RandomForest importance + SelectKBest)
- **Model training and evaluation** with:
  - K-Nearest Neighbors (KNN, tuned with GridSearchCV)
  - Decision Tree (tuned with GridSearchCV)
  - Logistic Regression
  - Gaussian Naive Bayes
- **Evaluation metrics**: Accuracy, Precision, Recall, F1 Score

---

## ⚙️ Installation

Clone repository and install dependencies:

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
pip install pandas numpy matplotlib seaborn scikit-learn




