# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# -------------------------
# Streamlit app title
# -------------------------
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection Dashboard")
st.write("""
This dashboard allows you to input credit card transaction features and predict fraud probability.
It also shows feature importance using SHAP explanations.
""")

# -------------------------
# Step 1: Load dataset from Kaggle
# -------------------------
st.header("Step 1: Load Dataset")
st.write("Downloading Credit Card Fraud dataset from Kaggle...")
import os
try:
    import kagglehub   # make sure kagglehub is installed and configured with your Kaggle API token
except ImportError:
    st.error("Please install kagglehub via `pip install kagglehub` and upload your Kaggle API token.")

# Path to dataset
if not os.path.exists("creditcard.csv"):
    try:
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        st.success(f"Dataset downloaded successfully! Path: {path}")
    except Exception as e:
        st.error(f"Failed to download dataset. Please upload manually.\n{e}")

df = pd.read_csv("/home/appuser/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3")
st.write("Dataset preview:")
st.dataframe(df.head())

# -------------------------
# Step 2: Preprocessing
# -------------------------
st.header("Step 2: Preprocessing")
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Handle class imbalance with SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
st.write(f"After SMOTE, training data shape: {X_train_res.shape}")

# -------------------------
# Step 3: Train XGBoost model
# -------------------------
st.header("Step 3: Model Training")
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_res, y_train_res)
st.success("XGBoost model trained successfully!")

# Show model evaluation on test set
y_pred = xgb_model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
st.subheader("Model Evaluation on Test Data")
st.json(report)

# -------------------------
# Step 4: SHAP Explainer
# -------------------------
st.header("Step 4: SHAP Feature Importance")
explainer = shap.TreeExplainer(xgb_model)

# Global SHAP summary
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(plt.gcf())

# -------------------------
# Step 5: User input for prediction
# -------------------------
st.header("Step 5: Predict Fraud for a Transaction")
st.sidebar.header("Transaction Input Features")
input_data = {}
for col in X.columns:
    input_data[col] = st.sidebar.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([input_data])
pred_prob = xgb_model.predict_proba(input_df)[:,1][0]
pred_class = "Fraud" if pred_prob >= 0.5 else "Normal"

st.subheader("Prediction Result")
st.write(f"Predicted Class: **{pred_class}**")
st.write(f"Fraud Probability: **{pred_prob:.2f}**")

# SHAP explanation for this transaction
st.subheader("Top Feature Contributions (SHAP)")
shap_values_input = explainer.shap_values(input_df)

# Force plot
plt.figure()
shap.force_plot(explainer.expected_value, shap_values_input[0,:], input_df, matplotlib=True)
st.pyplot(plt.gcf())

# Top 5 features
shap_df = pd.DataFrame({'Feature': X.columns, 'SHAP Value': shap_values_input[0,:]})
shap_df['Abs'] = shap_df['SHAP Value'].abs()
top_features = shap_df.sort_values('Abs', ascending=False).head(5)
st.write("Top 5 features contributing to prediction:")
st.dataframe(top_features[['Feature', 'SHAP Value']])
