streamlit_app: https://creditcardfrauddetection-pe554ntahhpkbscfuvcuwz.streamlit.app/


Credit Card Fraud Detection Project

Authors: Debodip Chowdhury
Project Type: End-to-End Machine Learning / Data Science Project

📌 Project Overview

This project is all about detecting fraudulent credit card transactions using machine learning. The goal is to build an end-to-end system that can:

Preprocess and clean transaction data

Handle imbalanced datasets

Train models like Logistic Regression and XGBoost

Explain predictions with feature importance (SHAP)

Provide a simple dashboard for interactive predictions

We used real-world datasets with anonymized credit card transactions and focused on making the project both practical and understandable.

🗂 Dataset

The dataset comes from European cardholders in September 2013.

It contains 284,807 transactions, out of which 492 are fraudulent.

Features V1–V28 are PCA-transformed components for privacy reasons.

Other features include:

Time → Seconds since the first transaction

Amount → Transaction amount

Class → 1 for fraud, 0 otherwise

The dataset was collected and analyzed by Worldline and the Machine Learning Group (MLG) at ULB, Belgium.

🛠 Features / Models

We explored several models:

Logistic Regression – baseline model

XGBoost – high-performance tree-based model

SMOTE – to handle class imbalance

We also included SHAP explainability to show which features influence predictions the most.

📊 Dashboard

We built a Streamlit dashboard where users can:

Input new transaction data (Time, Amount, PCA features V1–V28)

See fraud probability predictions

View feature importance plots

Users don’t need to know the exact meaning of PCA components—they can use default values for testing/demo.

⚡ How to Run

Clone the repository:

git clone https://github.com/yourusername/credit-card-fraud-detection.git

Install requirements:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Open the browser link to interact with the dashboard.


✅ Key Takeaways

Fraud detection requires handling highly imbalanced datasets.

Tree-based models like XGBoost perform very well.

Explainability (SHAP) is crucial for understanding predictions.

Dashboards make ML models accessible to non-technical users.
