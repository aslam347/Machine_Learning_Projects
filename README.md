# Machine Learning Projects (Fundamentals)

This repository contains my hands-on Machine Learning practice projects using real-world datasets.
Each folder includes notebooks/scripts with step-by-step implementation, preprocessing, training, and evaluation.

---

## 📌 Projects Included

### 1) Simple Linear Regression
- Predict house price using linear regression  
- Dataset: home_prices.csv  
- Key topics: regression basics, model fit, prediction  

### 2) Gradient Descent (From Scratch + Scikit-learn)
- Implemented gradient descent for linear regression  
- Compared results with scikit-learn LinearRegression  

### 3) Train-Test Split & Model Evaluation
- Car mileage prediction
- Key topics: train-test split, evaluation metrics

### 4) One-Hot Encoding
- Handling categorical variables using one-hot encoding  
- Key topics: encoding, preprocessing pipeline

### 5) Polynomial Regression
- Learned how polynomial features improve regression performance  
- Key topics: polynomial features, model comparison

### 6) Logistic Regression (Precision, Recall, Accuracy)
- Built a Logistic Regression model for classification
- Evaluated performance using Precision, Recall, Accuracy
- Dataset: `social_network_ads.csv`

### 7) Multiclass Classification using Logistic Regression (Dry Bean Dataset)
- Built a Logistic Regression model for multiclass classification  
- Predicted bean class based on shape features  
- Dataset: `dry_bean_dataset.xlsx`
- Key topics:
  - Multiclass classification
  - Feature scaling (StandardScaler)
  - Train-test split with stratification
  - Evaluation: Accuracy, Precision, Recall, F1-score
  - Confusion Matrix

### 8) SVM (Gamma, Regularization & Scaling) – Raisin Dataset
- Built a Support Vector Machine (SVM) model using Raisin dataset  
- Dataset: `Raisin_Dataset.xlsx`
- Key topics:
  - Support Vector Machine (SVM)
  - Feature scaling using StandardScaler
  - Understanding Gamma and C (Regularization)
  - Model evaluation (Accuracy, Classification Report, Confusion Matrix)
  - Pipeline implementation (Scaling + SVM)

### 9) Weather Type Classification using SVM (Kaggle Dataset)
- Predicted weather type (Rainy / Sunny / Cloudy / Snowy) using weather condition features  
- Dataset: `weather_classification_data.csv` (Kaggle)
- Key topics:
  - Data preprocessing (One-Hot Encoding + Scaling)
  - Exploratory Data Analysis (EDA): Pie chart, Histogram, Box plot
  - SVM with different kernels (Linear, RBF)
  - Hyperparameter tuning (C, gamma)
  - Pipeline usage (StandardScaler + SVM)
  - Evaluation: Accuracy, Classification Report, Confusion Matrix
- Dataset credits: Nikhil Narayan (Kaggle)    

### 10) SMS Classification using Naive Bayes
- Built an SMS Spam Detection model (Spam / Ham)
- Algorithm: Naive Bayes
- Key topics:
  - Text preprocessing
  - Feature extraction using Bag of Words / TF-IDF
  - Train-test split
  - Naive Bayes model training
  - Evaluation: Accuracy, Precision, Recall, F1-score
  - Confusion Matrix + Classification Report

### 11) Titanic Survival Prediction using Gaussian Naive Bayes
- Built a binary classification model to predict Titanic passenger survival
- Dataset: `titanic.csv`
- Algorithm: Gaussian Naive Bayes
- Key topics:
  - Data Cleaning & Preprocessing
  - Handling missing values (median imputation)
  - One-hot encoding (sex)
  - Feature scaling using StandardScaler (fare)
  - Train-test split
  - Evaluation: Accuracy, Precision, Recall, F1-score
  - Confusion Matrix + Classification Report

---

### 12) Decision Tree (Fundamentals)
- Built a Decision Tree model for classification
- Key topics:
  - Gini / Entropy
  - Information Gain
  - Overfitting control (max_depth, min_samples_split)
  - Confusion Matrix + Classification Report
- Folder: [View](./12_Decision_Tree)

---

### 13) Pet Adoption Prediction using Decision Tree
- Built a Decision Tree model to predict pet adoption outcome
- Dataset: Pet adoption dataset
- Key topics:
  - Feature preprocessing (encoding)
  - Train-test split with stratification
  - Decision Tree classifier
  - Evaluation: Accuracy, Precision, Recall, F1-score
  - Confusion Matrix
- Folder: [View](./13_Pet_adoption_Decision_Tree)

---

### 14) Customer Churn Prediction (Class Imbalance)
- Built churn prediction model on imbalanced dataset
- Dataset: Customer churn dataset
- Techniques:
  - Undersampling
  - Oversampling (SMOTE)
  - SMOTE-Tomek (Best hybrid)
- Key topics:
  - Precision / Recall / F1-score (focus on Recall)
  - Confusion Matrix
  - Threshold tuning
- Folder: [View](./14_ChurnPredection_ClassImbalance)

---

### 15) Breast Cancer Prediction
- Built ML model to predict breast cancer (Benign/Malignant)
- Key topics:
  - Data cleaning + preprocessing
  - Model training + evaluation
  - Confusion matrix + classification report
  - Model comparison
- Folder: [View](./15_Breast_Cancer)

---

### 16) Cancer Prediction using Random Forest (Ensemble Learning)
- Compared Decision Tree vs Random Forest for cancer prediction
- Dataset: `cancer_data.csv` (Kaggle)
- Key topics:
  - Bagging concept
  - Random Forest training (random rows + random features)
  - Feature importance
  - Hyperparameter tuning (`n_estimators`, `max_depth`, etc.)
  - Handling imbalance using undersampling (optional)
- Folder: [View](./16_Cancer_Prediction_Random_Forest)




---




## 📌 Portfolio Projects (Quick View)

|## 📌 Portfolio Projects (Quick View)

| Project | Type | Algorithm | Link |
|--------|------|-----------|------|
| Titanic Survival Prediction | Classification | Gaussian Naive Bayes | [View](./11_Titanic_ML_Classification) |
| SMS Spam Detection | NLP Classification | Multinomial Naive Bayes | [View](./10_SMS_Classification_NaiveBayes) |
| Decision Tree Fundamentals | Classification | Decision Tree Classifier | [View](./12_Decision_Tree) |
| Pet Adoption Prediction | Classification | Decision Tree Classifier | [View](./13_Pet_adoption_Decision_Tree) |
| Customer Churn Prediction | Imbalanced Classification | Random Forest Classifier | [View](./14_ChurnPredection_ClassImbalance) |
| Breast Cancer Prediction | Classification | Logistic Regression / Decision Tree | [View](./15_Breast_Cancer) |
| Cancer Prediction (Ensemble Learning) | Classification | Decision Tree + Random Forest | [View](./16_Cancer_Prediction_Random_Forest) |






## 🛠 Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

## 🚀 How to Run (Local Setup)

### 1) Clone the repository
```bash
git clone https://github.com/aslam347/Machine_Learning_Projects.git
cd Machine_Learning_Projects
