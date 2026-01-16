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
