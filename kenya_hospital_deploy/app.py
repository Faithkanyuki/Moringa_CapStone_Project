import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# Page setup
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    layout="wide"
)

# Title
st.title("Kenya Hospital Readmission Risk Predictor")
st.write("Clinical tool for predicting patient readmission risk")

# Load model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest_model.joblib")
        with open("feature_names.pkl", "rb") as f:
            features = pickle.load(f)
        with open("model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        return model, features, metadata
    except Exception as e:
        st.error(f"Error: {str(e)[:100]}")
        return None, None, None

model, features, metadata = load_model()

if model is None:
    st.stop()

# Get threshold
threshold = metadata.get("optimal_threshold", 0.48)

# Prediction function
def predict(data_dict):
    try:
        df = pd.DataFrame({feat: [0] for feat in features})
        for key, value in data_dict.items():
            if key in df.columns:
                df[key] = value
        prob = model.predict_proba(df)[0, 1]
        return prob
    except:
        return None

# User interface
st.header("Patient Assessment")

col1, col2 = st.columns(2)

with col1:
    days = st.slider("Hospital Stay (days)", 1, 30, 7)
    visits = st.number_input("Hospital Visits (past year)", 0, 50, 3)
    emergency = st.number_input("Emergency Visits", 0, 20, 1)

with col2:
    labs = st.number_input("Lab Procedures", 0, 200, 45)
    meds = st.number_input("Medications", 0, 100, 12)
    age = st.slider("Age", 18, 100, 58)

# Predict button
if st.button("Predict Risk", type="primary"):
    data = {
        "time_in_hospital": days,
        "total_hospital_visits": visits,
        "number_emergency": emergency,
        "num_lab_procedures": labs,
        "num_medications": meds,
        "age_numeric": age
    }
    
    prob = predict(data)
    
    if prob is not None:
        st.success("Assessment Complete")
        
        # Show results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Probability", f"{prob:.1%}")
            st.metric("Threshold", f"{threshold:.1%}")
        
        with col2:
            if prob >= threshold:
                st.metric("Risk Level", "HIGH RISK")
                st.write("Action: Priority follow-up")
            else:
                st.metric("Risk Level", "LOW RISK")
                st.write("Action: Standard care")
        
        with col3:
            st.write("Risk Factors:")
            if visits >= 4:
                st.write(f"- {visits} hospital visits")
            if emergency >= 2:
                st.write(f"- {emergency} ED visits")

# Sidebar
st.sidebar.title("Model Info")
st.sidebar.write(f"Recall: {metadata.get('recall', 0.69):.1%}")
st.sidebar.write(f"Precision: {metadata.get('precision', 0.154):.1%}")
st.sidebar.write(f"Threshold: {threshold:.3f}")

# Footer
st.markdown("---")
st.caption("Kenya Hospital Readmission Predictor")
