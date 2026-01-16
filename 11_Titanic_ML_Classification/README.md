# 11) Titanic Survival Prediction using Gaussian Naive Bayes 🚢

This project builds a **Binary Classification Machine Learning model** to predict whether a passenger survived the Titanic crash.

Target column:
- `survived = 1` → Passenger survived ✅
- `survived = 0` → Passenger did not survive ❌

---

## 📂 Dataset
- File name: `titanic.csv`

### Columns in Dataset
- `passenger_id`
- `name`
- `p_class`
- `sex`
- `age`
- `sib_sp`
- `parch`
- `ticket`
- `fare`
- `cabin`
- `embarked`
- `survived` (Target)

---

## 🎯 Objective
To train a model that predicts passenger survival based on important features such as:
- Passenger class
- Gender
- Age
- Fare

---

## 🧠 Algorithm Used
✅ **Gaussian Naive Bayes**

Gaussian Naive Bayes works well for classification problems where features are numeric and assumed to follow a **normal distribution**.

---

## ⚙️ Workflow
1. Import Titanic dataset
2. Data Exploration (EDA)
3. Drop unnecessary columns
4. Handle missing values (`age`, `fare`)
5. One-Hot Encoding (`sex`)
6. Feature scaling using **StandardScaler** (`fare`)
7. Train-Test Split
8. Train Gaussian Naive Bayes model
9. Predict test results
10. Model evaluation

---

## 📊 EDA Visualizations
- Bar chart for:
  - `survived`
  - `p_class`
- Pie chart for:
  - `sex`
- Histograms for:
  - `age`
  - `fare`

---

## ✅ Model Evaluation Metrics
The model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

---

## 🚀 How to Run
### 1) Install Dependencies
```bash
pip install pandas numpy matplotlib scikit-learn
