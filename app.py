import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------
# 1. Load pipeline and feature list
# -------------------------------
pipeline = joblib.load("fraud_pipeline.pkl")
feature_cols = joblib.load("features.pkl")

st.title("Insurance Fraud Prediction App")

st.write("Enter the details of the insurance claim below:")

# -------------------------------
# 2. Create input form
# -------------------------------
def user_input_features():
    data = {}
    
    # Example numeric inputs (replace with actual dataset columns)
    numeric_cols = ["months_as_customer", "age", "policy_deductable",
                    "policy_annual_premium", "umbrella_limit", "capital-gains",
                    "capital-loss", "incident_hour_of_the_day", "number_of_vehicles_involved",
                    "bodily_injuries", "witnesses", "injury_claim", "property_claim",
                    "vehicle_claim", "auto_year"]
    
    for col in numeric_cols:
        data[col] = st.number_input(f"{col}", value=0)

    # Example categorical inputs (replace with actual dataset columns)
    categorical_cols = ["policy_state", "policy_csl", "insured_sex",
                        "insured_education_level", "insured_occupation",
                        "insured_hobbies", "insured_relationship",
                        "incident_type", "collision_type", "incident_severity",
                        "authorities_contacted", "incident_state", "incident_city",
                        "incident_location", "property_damage",
                        "police_report_available", "auto_make", "auto_model"]
    
    for col in categorical_cols:
        data[col] = st.text_input(f"{col}", value="Unknown")
    
    # Date inputs
    date_cols = ["policy_bind_date", "incident_date"]
    for col in date_cols:
        date_str = st.text_input(f"{col} (DD-MM-YYYY)", value="01-01-2020")
        try:
            date_val = pd.to_datetime(date_str, dayfirst=True)
            data[f"{col}_day"] = date_val.day
            data[f"{col}_month"] = date_val.month
            data[f"{col}_year"] = date_val.year
        except:
            data[f"{col}_day"] = 0
            data[f"{col}_month"] = 0
            data[f"{col}_year"] = 0
    
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# -------------------------------
# 3. Align input with training features
# -------------------------------
# Add missing columns with default 0
for col in feature_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Keep only the order of training features
input_df = input_df[feature_cols]

# -------------------------------
# 4. Prediction
# -------------------------------
if st.button("Predict Fraud"):
    try:
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0][1]
        result = "Fraudulent Claim" if prediction == 1 else "Legitimate Claim"
        st.write(f"**Prediction:** {result}")
        st.write(f"**Fraud Probability:** {prediction_proba:.2f}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")





