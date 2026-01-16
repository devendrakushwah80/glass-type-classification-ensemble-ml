# 🧪 Glass Type Classification using Machine Learning & Ensemble Methods

## 📌 Project Overview
This project implements an **end-to-end machine learning pipeline** to classify different **types of glass** based on their chemical composition.  
The workflow includes **EDA, preprocessing, feature scaling, class imbalance handling, model training, hyperparameter tuning, ensemble learning**, and **model comparison** using multiple evaluation metrics.

---

## 📊 Dataset Description

### 🔹 Features
- RI – Refractive Index  
- Na – Sodium  
-  
- Mg – Magnesium  
- Al – Aluminum  
- Si – Silicon  
- K – Potassium  
- Ca – Calcium  
- Ba – Barium  
- Fe – Iron  

### 🎯 Target
- **Type** → Multiclass label representing different glass categories

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Feature distribution analysis
- Boxplots of numerical features vs target classes
- Outlier detection

### 2️⃣ Train–Test Split
- Stratified split to preserve class distribution
```python
train_test_split(..., stratify=y)
```
3️⃣ Data Preprocessing

ColumnTransformer

StandardScaler

set_output(transform="pandas")

make_pipeline() for clean workflow

🧠 Models Implemented

Each model was trained using GridSearchCV and evaluated using weighted metrics.

Model	Description
Logistic Regression	Linear classifier
Support Vector Machine (SVM)	Linear & RBF kernels
K-Nearest Neighbors (KNN)	Distance-based
Decision Tree	Tree-based
Random Forest	Ensemble (Bagging)
⚖️ Class Imbalance Handling

class_weight='balanced'

Stratified train-test split

Weighted evaluation metrics

🔍 Hyperparameter Tuning

GridSearchCV

5-fold Cross-Validation

Scoring: f1_weighted

📈 Evaluation Metrics

All models were compared using:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

ROC-AUC (One-vs-Rest for multiclass)

Results were stored in a comparison table for easy analysis.

🧩 Ensemble Learning
🔹 Voting Classifier (Soft Voting)

A soft voting ensemble was created using:

Logistic Regression

Support Vector Machine

Random Forest

This approach improved overall performance by combining linear, non-linear, and tree-based models.

🏆 Final Outcome

Improved generalization using ensemble learning

Balanced performance across all classes

Better F1-score and ROC-AUC compared to individual models

🛠️ Tech Stack

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

🚀 How to Run
git clone https://github.com/your-username/glass-type-classification-ensemble-ml.git
cd glass-type-classification-ensemble-ml
jupyter notebook glass_classification_full_ml_pipeline.ipynb

📌 Key Learnings

Building clean ML pipelines

Handling multiclass imbalance

Hyperparameter tuning with GridSearchCV

Ensemble learning using Voting Classifier

Model comparison using multiple evaluation metrics

👤 Author

Devendra Kushwah


---
