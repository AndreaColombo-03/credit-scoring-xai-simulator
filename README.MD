# 🏦 Interactive Credit Scoring Simulator with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green.svg)

## 📌 Context & Project Overview
This project consists of an end-to-end **Credit Scoring Web Application** built to evaluate loan applications and predict the Probability of Default (PD). 

The core architecture and credit risk methodologies were developed following the insights and institutional frameworks discussed during the **"Machine Learning in Industry" training week organized by Cardo AI**. The goal is to address the crucial regulatory requirement of algorithmic transparency in banking (e.g., EBA guidelines) by integrating **SHAP (SHapley Additive exPlanations)** into complex non-linear models.

👉 **[TRY THE LIVE APP HERE](INSERISCI_IL_LINK_DI_STREAMLIT_CHE_OTTERRAI_NELLA_FASE_5)**

## 🛠️ Technical Architecture
1. **Data Engineering (`01_Data_Preparation.ipynb`):** Performed Vintage Analysis on a transactional credit record dataset to construct the binary target variable (`TARGET = 1` for defaults > 60 days past due, aligned with Basel framework principles).
2. **Machine Learning (`train.py`):** Trained an `XGBoost` classifier optimized for ROC-AUC. Addressed extreme class imbalance (1.69% default rate) via cost-sensitive learning (`scale_pos_weight`) to severely penalize False Negatives.
3. **Explainable AI (`app.py`):** Integrated a `SHAP` TreeExplainer to deconstruct individual credit scores into marginal feature contributions via dynamic Waterfall plots.
4. **Quality Assurance (`test_credit_model.py`):** Implemented automated unit tests with `pytest` to guarantee deterministic model behavior and handle structural survival bias within the sample.

## 👨‍💻 Author
**Andrea Colombo** – MSc Student in Quantitative Finance
