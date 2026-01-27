# ❤️ Heart Failure Prediction using Machine Learning

## 📌 Project Overview
This project predicts the **likelihood of heart failure events** using clinical data.  
The goal is to assist in **early risk detection** using Machine Learning models.

We evaluate models using:
- ROC Curve
- AUC Score
- Recall-based threshold tuning

---

## 📊 Dataset Information
**Dataset:** Heart Failure Clinical Records  
**Source:** UCI Machine Learning Repository  

| Feature | Description |
|---------|------------|
| age | Age of patient (years) |
| anaemia | Decrease of RBC/Hemoglobin (0/1) |
| creatinine_phosphokinase | CPK enzyme level |
| diabetes | Diabetes status (0/1) |
| ejection_fraction | % of blood pumped from heart |
| high_blood_pressure | Hypertension (0/1) |
| platelets | Platelet count |
| serum_creatinine | Creatinine level |
| serum_sodium | Sodium level |
| sex | Gender (1 = male, 0 = female) |
| smoking | Smoking status (0/1) |
| time | Follow-up period (days) |
| death_event | Target (1 = died, 0 = survived) |

---

## 🎯 Objective
Build ML models to predict **death events** and evaluate performance using medical-risk focused metrics.

---

## ⚙️ Project Workflow

### 1️⃣ Data Preprocessing
- Train-test split
- StandardScaler applied only to continuous features:
  - age
  - creatinine_phosphokinase
  - ejection_fraction
  - platelets
  - serum_creatinine
  - serum_sodium
  - time

(Binary features were not scaled.)

---

### 2️⃣ Models Used
- Gaussian Naive Bayes
- Support Vector Machine (SVM)
- XGBoost Classifier

---

### 3️⃣ Evaluation Metrics
- Classification Report
- ROC Curve
- AUC Score
- Threshold tuning (Recall = 85%)

---

## 📈 ROC & AUC
AUC measures overall model performance.  
Higher AUC = better ability to separate classes.

---

## ⚖️ Cost-Sensitive Decision Making
We selected a probability threshold achieving **85% recall** to minimize missed heart failure cases and checked the false positive rate.

---

## 🧠 Key Learnings
- Scaling improves SVM & GaussianNB
- XGBoost is less affected by scaling
- Accuracy alone is misleading in medical ML
- Recall is critical in healthcare
- Threshold tuning is important

---

## 🛠 Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  

---

## 🚀 How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib
