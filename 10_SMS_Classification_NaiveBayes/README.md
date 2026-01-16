# 10) SMS Classification using Naive Bayes 📩

This project builds a **Spam vs Ham (Not Spam)** SMS classifier using the **Naive Bayes algorithm**.  
The goal is to predict whether a message is **spam (1)** or **ham (0)** based on the text content.

---

## 📌 Problem Statement
Given SMS text messages, classify them into:
- **Spam**
- **Not Spam (Ham)**

This is a **Binary Classification** problem.

---

## 📂 Dataset
Typical dataset used: **SMS Spam Collection Dataset**
- Columns:
  - `label` → spam / ham
  - `message` → SMS text

---

## 🧠 ML Concepts Used
- Text preprocessing (cleaning)
- Tokenization
- Vectorization using **CountVectorizer / TF-IDF**
- Train-Test split
- Model training using **Multinomial Naive Bayes**
- Evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
  - Classification Report

---

## ⚙️ Workflow
1. Load dataset
2. Clean & preprocess text
3. Convert text → numeric features (vectorization)
4. Split data into train/test
5. Train Naive Bayes model
6. Predict test data
7. Evaluate model performance

---

## 🚀 How to Run
### 1) Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib
