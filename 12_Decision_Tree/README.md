# Salary Classification using Decision Tree 🌳💼

This project demonstrates how to use a **Decision Tree Classifier** to predict whether an employee salary is **more than 100K** based on:

- Company
- Job role
- Degree

The project also compares two splitting criteria:
✅ **Gini Impurity** vs ✅ **Entropy (Information Gain)**

---

## 📌 Dataset
**File:** `salaries.csv`

Sample columns:

| company | job | degree | salary_more_then_100k |
|--------|-----|--------|------------------------|
| google | sales executive | bachelors | 0 |
| google | business manager | masters | 1 |

Target column:
- `salary_more_then_100k`
  - `1` → Salary > 100K
  - `0` → Salary ≤ 100K

---

## ⚙️ Feature Engineering (Encoding)

### 1️⃣ Label Encoding for Degree
Degree values are converted into numbers:

- bachelors → `1`
- masters → `2`

```python
df['degree_number'] = df.degree.map({'bachelors':1, 'masters': 2})
df.drop('degree', axis="columns", inplace=True)
