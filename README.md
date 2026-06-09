# 🛡️ Credit Card Fraud Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detection-bqpxvewusrbthfca8hx3rg.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ROC--AUC%200.98-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning system to detect fraudulent credit card transactions. Built on 284,807 real-world transactions with a severe class imbalance of **577:1** (only 0.17% fraud), this project demonstrates how to build a production-ready fraud detector — from raw data to a live, explainable web app.

**[🚀 Live Demo](https://credit-card-fraud-detection-bqpxvewusrbthfca8hx3rg.streamlit.app/)**

---

## 📊 The Problem

Detecting fraud is not a standard classification problem. If we naively predict every transaction as legitimate:
- **Accuracy = 99.83%** — looks perfect
- **Fraud caught = 0** — completely useless

This is why this project focuses on **Recall, Precision, F1, and ROC-AUC** instead of accuracy, and uses dedicated techniques to handle the extreme class imbalance.

---

## 🔬 Approach

### Class Imbalance Handling
| Technique | Description |
|---|---|
| `class_weight='balanced'` | Penalizes misclassification of minority class proportionally |
| SMOTE | Generates synthetic fraud samples to balance the training set |
| `scale_pos_weight` | XGBoost-native imbalance correction (ratio = 577) |

### Models Compared
| Model | ROC-AUC | Fraud Caught | False Alarms |
|---|---|---|---|
| Logistic Regression (Baseline) | 0.9573 | 63/98 | 13 |
| LR + Class Weights | 0.9722 | 90/98 | 1389 |
| LR + SMOTE | 0.9698 | 90/98 | 1458 |
| Random Forest | 0.9529 | 73/98 | 3 |
| **XGBoost Tuned ✅** | **0.9800** | **82/98** | **14** |

### Threshold Tuning
Default threshold of 0.5 is not optimal for imbalanced fraud detection. Threshold was tuned to **0.85** using F1 score maximization:
- **Precision: 0.91** — 91% of flagged transactions are real fraud
- **Recall: 0.83** — catches 83% of all fraud
- **F1: 0.87**

---

## 🔍 SHAP Explainability

Every prediction is explained using **SHAP (SHapley Additive exPlanations)**:
- Global feature importance — which features matter most overall
- Per-transaction waterfall plots — exactly why a transaction was flagged
- Top fraud signals: **V14, V10, V12** (strong negative correlation with fraud probability)

---

## 🖥️ Streamlit App Features

| Tab | What it shows |
|---|---|
| 📊 EDA & Data Insights | Class distribution, amount analysis, feature correlation heatmap |
| 🤖 Model Comparison | ROC curves, PR curves, confusion matrix for all 5 models |
| 🔍 SHAP Explainability | Feature importance bar chart + beeswarm plot |
| 🚨 Live Fraud Predictor | Enter any transaction → instant verdict + SHAP waterfall explanation |

---

## 🛠️ Tech Stack

- **Data & ML:** Pandas, NumPy, Scikit-learn, XGBoost, imbalanced-learn
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Cloud

---

## 🚀 Run Locally

```bash
git clone https://github.com/Devina0810/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt

# Download dataset from Kaggle:
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place creditcard.csv in the project root

streamlit run app.py
```

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
├── app.py                          # Streamlit web app
├── credit-card-fraud-detection.ipynb  # Full analysis notebook
├── requirements.txt                # Dependencies
└── README.md
```

---

## 📈 Dataset

**[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)** — ULB Machine Learning Group

- 284,807 transactions over 2 days (September 2013, European cardholders)
- 492 fraud cases (0.17%)
- Features V1–V28 are PCA-transformed for confidentiality
- `Time`, `Amount`, and `Class` are the only original features

---

## 👩‍💻 Author

**Devina Yadav** — CSE Student, IIIT Kota  
[GitHub](https://github.com/Devina0810) · [Kaggle](https://kaggle.com/midnight1708)
