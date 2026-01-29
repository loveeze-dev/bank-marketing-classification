# Bank Marketing Campaign: Predicting Term Deposit Subscriptions

## Overview
This project develops and compares supervised machine learning models to predict whether bank clients will subscribe to a term deposit following a direct marketing campaign. Using **Logistic Regression** and **Random Forest** classifiers, the analysis tackles a **severely imbalanced dataset (88.7% non-subscribers)** and identifies the economic, demographic, and campaign-related factors that most strongly influence subscription decisions.

The project focuses on **business impact**, emphasizing precision-driven targeting to reduce campaign costs and improve return on investment.

---

## Business Problem
Direct marketing campaigns are expensive and time-intensive. Contacting uninterested clients leads to wasted resources and customer fatigue.

This project aims to:
- **Improve targeting accuracy** by identifying high-probability subscribers
- **Reduce campaign costs** by minimizing unnecessary contacts
- **Optimize campaign strategy** through timing and frequency insights
- **Increase ROI** by prioritizing precision over volume

---

## Skills & Tools

### Programming & Libraries
- **Python**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

### Machine Learning
- Logistic Regression (L2 regularization)
- Random Forest (ensemble learning)
- Hyperparameter tuning with GridSearchCV
- Stratified train-test splitting for imbalanced data

### Data Processing
- One-hot encoding (10 categorical variables → 53 dummy features)
- Feature scaling (Z-score normalization)
- Data leakage prevention (removal of post-outcome features)

### Evaluation Metrics
Accuracy, Balanced Accuracy, Precision, Recall, F1-score, AUC-ROC, Matthews Correlation Coefficient (MCC), Confusion Matrix

---

## Dataset

**Source**: UCI Machine Learning Repository – Bank Marketing Dataset  
**Size**: 41,188 observations × 20 features  

### Target Variable
- `y`: Term deposit subscription (`yes` / `no`)

### Class Distribution
- No: 36,548 (88.7%)
- Yes: 4,640 (11.3%)
- **Imbalance ratio**: 7.9 : 1

### Key Feature Groups
- **Client demographics**: Age, job, marital status, education
- **Campaign features**: Contact type, month, day, number of contacts
- **Previous campaigns**: Past outcomes and contact history
- **Economic indicators**: Euribor rate, employment variation, CPI, consumer confidence

> **Important**: The `duration` feature was excluded to prevent data leakage, as it is only known after the campaign outcome.

---

## Key Findings

### Model Performance (Test Set)

| Metric | Logistic Regression | Random Forest |
|------|---------------------|---------------|
| Accuracy | 90.12% | **90.80%** |
| Balanced Accuracy | 60.91% | **62.28%** |
| Precision | 0.655 | **0.689** |
| Recall | 0.284 | **0.310** |
| F1-score | 0.396 | **0.428** |
| AUC-ROC | 0.798 | **0.805** |
| MCC | 0.372 | **0.402** |
| Training Time | **0.14s** | 18.7s |

**Result**: Random Forest delivers stronger predictive performance, while Logistic Regression offers faster training and simpler deployment.

---

## Insights & Interpretation

### 1. Economic Conditions Are Critical
- **Euribor 3-month rate** is the strongest predictor
  - Lower interest rates significantly increase subscription likelihood
- **Employment variation rate** shows a negative relationship
  - Economic uncertainty drives demand for safer investments
- **Consumer confidence** is inversely correlated with subscriptions

### 2. Campaign Strategy Matters More Than Volume
- Optimal number of contacts: **1–2 per client**
- Contacting clients 3+ times shows diminishing returns
- Clients who previously subscribed are **4× more likely** to subscribe again
- Highest conversion months: **May and October**

### 3. Demographic Patterns
- Higher subscription rates among clients aged **40–60**
- Students and retirees convert more frequently than blue-collar workers
- Education level has a mild positive effect

### 4. Handling Class Imbalance
- Recall is intentionally low (~30%) to prioritize **precision**
- High precision ensures fewer wasted calls
- False negatives are acceptable; false positives are costly

---

## Visualizations
All visualizations are rendered inline in the notebook:
1. Class distribution (target imbalance)
2. Age distribution by subscription outcome
3. Correlation heatmap (numeric features)
4. Campaign contact frequency analysis
5. ROC curves (model comparison)
6. Precision–Recall curves
7. Confusion matrix – Logistic Regression
8. Confusion matrix – Random Forest

---

## Model Architecture

### Logistic Regression
- L2 regularization (Ridge)
- Balanced class weights
- GridSearchCV (10-fold CV)
- Best C value selected via AUC-ROC

### Random Forest
- 200 trees
- Minimum samples per leaf: 5
- Bootstrap sampling
- Feature importance via Gini importance
- 5-fold cross-validation

---

## Project Structure
bank_marketing/

├── bank_marketing.ipynb

├── bank-additional-full.csv

└── README.md

---

## How to Run

git clone https://github.com/yourusername/bank-marketing-classification.git

cd bank-marketing-classification

pip install -r requirements.txt

jupyter notebook bank_marketing_analysis.ipynb

---

## Business Recommendations

### Immediate Actions
- Target clients with predicted probability >70%
- Prioritize previous subscribers
- Limit outreach to a maximum of two contacts

### Strategic Impact
- Reduce campaign costs by 40–60%
- Improve conversion rates by 15–25%
- Increase marketing ROI by 3–5×

---

## Limitations & Future Work
- Dataset reflects historical campaigns (2008–2013)
- Geographic specificity limits generalization

### Future improvements:
- Cost-sensitive learning
- Gradient boosting models
- SMOTE-based imbalance handling
- SHAP/LIME explainability
- Deployment via REST API


*This project demonstrates end-to-end supervised learning: data preprocessing, class imbalance handling, model comparison, and translating machine learning results into actionable business strategy.*
