# 🏦 Automated Loan Eligibility Evaluation

> **SK5016 — Data Mining | Institut Teknologi Bandung**  
> Kelompok 7 · May 2026

---

## 📌 Overview

Manual credit assessment is time-consuming, inconsistent, and prone to human bias — creating operational risk for lenders and missed revenue opportunities from misclassified applicants.

This project builds and evaluates a **machine learning framework** to automate loan eligibility decisions, enabling faster, more objective, and scalable credit screening.

**Key Result:** Random Forest achieved the best performance with **78% accuracy** and **AUC-ROC of 0.72**, with Credit History identified as the most influential predictor.

## 🧪 Dataset

| Field | Description |
|-------|-------------|
| `Gender` | Male / Female |
| `Married` | Yes / No |
| `Dependents` | 0 / 1 / 2 / 3+ |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Yes / No |
| `ApplicantIncome` | Monthly income of applicant |
| `CoapplicantIncome` | Monthly income of co-applicant |
| `LoanAmount` | Requested loan (in thousands USD) |
| `Loan_Amount_Term` | Repayment period (months) |
| `Credit_History` | 1 = Good / 0 = Bad |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | **Target** — Y (Approved) / N (Rejected) |

**Source:** [Kaggle — Loan Prediction Problem Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)

---

## ⚙️ Feature Engineering

Raw income and loan amount values don't fully capture creditworthiness. Five derived features were created:

| Feature | Formula | Rationale |
|---------|----------|-----------|
| `Total_Income` | `ApplicantIncome + CoapplicantIncome` | Combined household income |
| `Income_Per_Dependent` | `Total_Income / (Dependents + 1)` | Purchasing power per dependent |
| `Loan_to_Income` | `LoanAmount / (Total_Income/1000 + 1)` | Debt burden ratio |
| `Area_Income_Interaction` | `Property_Area × Log_Total_Income` | Location-income interaction |
| `Family_Size` | `Dependents + Married + 1` | Total household size |

Log-transformation (`log1p`) was applied to all skewed income/loan features.

---

## 🤖 Models Evaluated

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.66 | 0.52 |
| Decision Tree | 0.72 | 0.67 |
| **Random Forest** ⭐ | **0.78** | **0.72** |
| XGBoost | 0.73 | 0.68 |

**Winner: Random Forest** — best overall accuracy and discriminative power.

---

## 🔍 Key Insights

- **Credit History** is the single strongest predictor (χ²=176.11, p<0.0001)
- Engineered features consistently outperform raw inputs
- Married status and Property Area are statistically significant (Chi-square test)
- Income features show no significant T-test difference — raw income alone is insufficient

---

## 🚀 Running the App

### Prerequisites
```bash
pip install -r requirements.txt
```

### Launch Streamlit Dashboard
```bash
# Make sure train.csv and test.csv are in the working directory
streamlit run app/loan_approval.py
```

The app includes 4 pages:
1. **EDA** — Distributions, correlations, and statistical tests
2. **Feature Engineering** — New feature overview and selection results
3. **Model Performance** — Accuracy table, ROC curves, confusion matrices
4. **Live Prediction** — Input applicant data and get instant loan decision

---

## 👥 Team

| Name | NIM |
|------|-----|
| Ardiansah | 20225006 |
| Andika Mohammad Mahpudin | 20224301 |
| Keishi Tsabitah I | 10323016 |

---

## 📚 References

- Andriani, W., Gunawan, & Naja, N. N. P. W. (2025). *Analisis perbandingan machine learning untuk prediksi kelayakan kredit perbankan pada Bank BRI Tegal.* IT-Explore, 4(1), 82–92.
- Silmina, E. P., Sunardi, & Yudhana, A. (2025). *Predictive model for cooperative loan recipient eligibility using supervised machine learning.* JITK, 11(2), 496–507.
- Suftandar, H., Wasesa, M., & Putro, U. S. (2025). *Enhancing credit scoring prediction in Islamic banking with random forest machine learning model.* MEA, 9(2).
