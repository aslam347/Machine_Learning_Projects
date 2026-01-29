# 🏨 Hotel Reservation Cancellation Prediction

## 📌 Problem Statement
Hotel cancellations lead to revenue loss and poor resource planning. This project builds a **Machine Learning classification model** to predict whether a hotel reservation will be **canceled or not** using booking and customer-related details.

The objective is to help hotels:
- Reduce cancellation losses  
- Improve revenue management  
- Optimize room allocation  

---

## 📂 Dataset Information

**Dataset Name:** `hotel_reservations.csv`  
**Source:** Kaggle – Ahsan Raza  

The dataset contains booking-related information such as customer details, stay duration, booking history, and pricing.

### 🎯 Target Variable
| Column | Meaning |
|--------|---------|
| **booking_status** | 0 → Not Canceled, 1 → Canceled |

---

## 📊 Features Description

| Feature | Description |
|---------|-------------|
| booking_id | Unique ID of each booking |
| no_of_adults | Number of adults |
| no_of_children | Number of children |
| no_of_weekend_nights | Weekend stay nights |
| no_of_week_nights | Weekday stay nights |
| type_of_meal_plan | Meal plan selected |
| required_car_parking_space | 0 = No, 1 = Yes |
| room_type_reserved | Type of room |
| lead_time | Days between booking and arrival |
| arrival_year | Year of arrival |
| arrival_month | Month of arrival |
| arrival_date | Date of arrival |
| market_segment_type | Market segment category |
| repeated_guest | 0 = No, 1 = Yes |
| no_of_previous_cancellations | Past canceled bookings |
| no_of_previous_bookings_not_canceled | Past successful bookings |
| avg_price_per_room | Average room price |
| no_of_special_requests | Number of customer requests |

---

## 🛠 Project Workflow

### 1️⃣ Data Preprocessing
- Removed irrelevant columns (`booking_id`, arrival date components)
- Handled categorical variables
- Feature scaling using **StandardScaler**
- Defined:
  - **X (features)**
  - **y (target)**

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Booking status distribution  
- Lead time vs cancellation analysis  
- Price trends  
- Special requests analysis  
- Class imbalance checking  

---

### 3️⃣ Models Implemented

| Model | Purpose |
|-------|---------|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Rule-based classifier |
| Random Forest | Ensemble model |
| Naive Bayes | Probabilistic classifier |

---

### 4️⃣ Model Evaluation Techniques

- Train-Test Split  
- **K-Fold Cross Validation**  
- **Stratified K-Fold Cross Validation**  
- Accuracy Score  
- Confusion Matrix  

---

### 5️⃣ Hyperparameter Tuning

| Method | Models Used |
|--------|-------------|
| Grid Search CV | Naive Bayes, Random Forest |
| Randomized Search | Tree-based models |

Optimized parameters include:
- `n_estimators`
- `max_depth`
- `var_smoothing`

---

## 🎯 Project Goal

To build a complete ML pipeline demonstrating:

- Data preprocessing  
- EDA & visualization  
- Feature scaling  
- Cross-validation  
- Hyperparameter tuning  
- Model comparison  

---

## 📈 Expected Outcome

The best-performing model helps hotels:

- Predict cancellation risk  
- Offer targeted incentives  
- Improve operational planning  

---

## 🚀 Technologies Used

- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  

---

## 📎 Dataset Credits

Ahsan Raza  
Kaggle Dataset: *Hotel Reservations Classification Dataset*

---

## 👨‍💻 Author

Mohamed Aslam  
AI/ML Engineer | Machine Learning Enthusiast
