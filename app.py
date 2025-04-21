# app.py

import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the expected feature columns
with open('features.pkl', 'rb') as f:
    model_features = pickle.load(f)

# Streamlit form for user input
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)
tenure = st.slider("Tenure (in months)", 0, 72, 12)

# Convert input into DataFrame
input_dict = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "PhoneService": phone_service,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "tenure": tenure
}

feature_df = pd.DataFrame([input_dict])


# One-hot encode input
feature_df = pd.get_dummies(feature_df)

# Add any missing columns and set to 0
for col in model_features:
    if col not in feature_df.columns:
        feature_df[col] = 0

# Ensure column order matches training
feature_df = feature_df[model_features]

# Predict
prediction = model.predict(feature_df)[0]
prob = model.predict_proba(feature_df)[0]

# Display result
st.write(f"Prediction: {'Churn' if prediction else 'No Churn'}")
st.write(f"Confidence: {max(prob) * 100:.2f}%")
