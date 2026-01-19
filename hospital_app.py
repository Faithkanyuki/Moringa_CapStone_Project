
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.graph_objects as go
from datetime import datetime

# ============================================
# PAGE SETUP
# ============================================
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL & ARTIFACTS
# ============================================
@st.cache_resource
def load_model_and_artifacts():
    """Load the trained model and supporting files"""
    try:
        # Load the Random Forest model
        model = joblib.load('random_forest_model.joblib')
        
        # Load feature names (48 features)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load model metadata
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load feature importance
        with open('feature_importance.pkl', 'rb') as f:
            feature_importance = pickle.load(f)
        
        return model, feature_names, metadata, feature_importance
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

# Load everything
model, feature_names, metadata, feature_importance = load_model_and_artifacts()

if model is None:
    st.stop()

optimal_threshold = metadata['optimal_threshold']

# ============================================
# APP TITLE & DESCRIPTION
# ============================================
st.title("🏥 Kenya Hospital Readmission Risk Predictor")
st.markdown("""
This tool helps healthcare providers in Kenya identify patients at high risk 
of hospital readmission. The model is optimized for the Kenyan healthcare context 
with **69.0% recall** - capturing most high-risk patients.
""")

# Display model performance
with st.expander("📊 Model Performance Summary", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recall", f"{metadata['recall']:.1%}", "69.0%", delta_color="off")
    with col2:
        st.metric("Precision", f"{metadata['precision']:.1%}", "15.4%", delta_color="off")
    with col3:
        st.metric("Optimal Threshold", f"{optimal_threshold:.3f}")
    with col4:
        st.metric("Features Used", metadata['features_count'])

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Patient", "Batch Upload", "Model Info", "How to Use"]
)

st.sidebar.markdown("---")
st.sidebar.info("**For Kenyan Healthcare:**")
st.sidebar.write("• Optimized for **high recall** (69.0%)")
st.sidebar.write("• Threshold: **0.480**")
st.sidebar.write("• Identifies **high-risk** patients")

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_for_patient(patient_features):
    """Make prediction for a single patient"""
    # Create dataframe with all features set to 0
    patient_df = pd.DataFrame({feature: [0] for feature in feature_names})
    
    # Update with provided features
    for feature, value in patient_features.items():
        if feature in patient_df.columns:
            patient_df[feature] = value
    
    # Get probability
    probability = model.predict_proba(patient_df)[0, 1]
    return probability

# ============================================
# SINGLE PATIENT MODE
# ============================================
if app_mode == "Single Patient":
    st.header("👤 Single Patient Assessment")
    
    # Create input form
    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Hospital History")
            time_in_hospital = st.slider("Current Hospital Stay (days)", 1, 30, 7)
            total_hospital_visits = st.number_input("Total Hospital Visits (past year)", 0, 50, 3)
            number_emergency = st.number_input("Emergency Visits (past year)", 0, 20, 1)
        
        with col2:
            st.subheader("Medical Details")
            num_lab_procedures = st.number_input("Lab Procedures", 0, 200, 45)
            num_medications = st.number_input("Medications", 0, 100, 12)
            age_numeric = st.slider("Age", 18, 100, 58)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Assess Readmission Risk", type="primary")
    
    if submitted:
        with st.spinner("Analyzing patient risk..."):
            # Prepare patient data
            patient_data = {
                'time_in_hospital': time_in_hospital,
                'total_hospital_visits': total_hospital_visits,
                'number_emergency': number_emergency,
                'num_lab_procedures': num_lab_procedures,
                'num_medications': num_medications,
                'age_numeric': age_numeric
            }
            
            # Get prediction
            probability = predict_for_patient(patient_data)
            
            # Determine risk
            if probability >= 0.7:
                risk = "🔴 VERY HIGH"
                action = "**URGENT** - Immediate follow-up within 3 days"
                color = "red"
            elif probability >= optimal_threshold:
                risk = "🟠 HIGH"
                action = "**PRIORITY** - Follow-up within 7 days"
                color = "orange"
            elif probability >= 0.3:
                risk = "🟡 MEDIUM"
                action = "**MONITOR** - Follow-up within 30 days"
                color = "yellow"
            else:
                risk = "🟢 LOW"
                action = "**ROUTINE** - Standard discharge care"
                color = "green"
            
            # Display results
            st.success("Risk Assessment Complete!")
            
            # Results in columns
            res1, res2, res3 = st.columns(3)
            
            with res1:
                # Simple gauge
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f2f6;">
                    <h1 style="color: {color}; font-size: 48px;">{probability:.1%}</h1>
                    <h3>Readmission Risk</h3>
                    <p>Threshold: {optimal_threshold:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with res2:
                st.metric("Risk Level", risk)
                st.metric("Recommendation", "Intervention Needed" if probability >= optimal_threshold else "Routine Care")
                st.metric("Threshold Status", 
                         f"Above by {probability-optimal_threshold:.3f}" if probability >= optimal_threshold 
                         else f"Below by {optimal_threshold-probability:.3f}")
            
            with res3:
                st.info("**Clinical Action Required:**")
                st.write(action)
                
                st.info("**Key Factors:**")
                factors = []
                if total_hospital_visits >= 4:
                    factors.append(f"• {total_hospital_visits} hospital visits (past year)")
                if number_emergency >= 2:
                    factors.append(f"• {number_emergency} emergency visits")
                if time_in_hospital >= 10:
                    factors.append(f"• {time_in_hospital}-day hospital stay")
                
                if factors:
                    for factor in factors:
                        st.write(factor)
                else:
                    st.write("• Standard risk profile")

# ============================================
# BATCH UPLOAD MODE
# ============================================
elif app_mode == "Batch Upload":
    st.header("📁 Batch Patient Upload")
    st.write("Upload a CSV file with multiple patients' data.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df)} patients")
            
            if st.button("Process All Patients"):
                with st.spinner("Processing..."):
                    # Simple processing
                    st.write(f"Sample data would be processed here")
                    st.write("For now, file uploaded successfully!")
                    
                    # Show sample
                    st.dataframe(df.head())
        except:
            st.error("Error reading file")

# ============================================
# MODEL INFO MODE
# ============================================
elif app_mode == "Model Info":
    st.header("🤖 Model Information")
    
    st.write("**Random Forest Model Details:**")
    st.write(f"- Optimal threshold: **{optimal_threshold:.3f}**")
    st.write(f"- Recall: **{metadata['recall']:.3f}** (Primary target)")
    st.write(f"- Precision: **{metadata['precision']:.3f}**")
    st.write(f"- Features: **{metadata['features_count']}**")
    
    if feature_importance:
        st.subheader("Top 5 Risk Factors")
        for i, feat in enumerate(feature_importance[:5], 1):
            st.write(f"{i}. **{feat['feature']}** (importance: {feat['importance']:.4f})")

# ============================================
# HOW TO USE MODE
# ============================================
else:
    st.header("📚 How to Use This Tool")
    
    st.markdown("""
    ### For Healthcare Providers in Kenya:
    
    1. **Single Patient Assessment**
       - Enter patient details in the form
       - Get instant risk prediction
       - View clinical recommendations
    
    2. **Understanding the Results**
       - **RED (≥70%)**: Very High Risk - Urgent action needed
       - **ORANGE (≥48%)**: High Risk - Priority follow-up
       - **YELLOW (30-48%)**: Medium Risk - Enhanced monitoring
       - **GREEN (<30%)**: Low Risk - Routine care
    
    3. **Key Features Considered**
       - Hospital visit history
       - Emergency department use
       - Length of stay
       - Age and medications
    
    4. **Model Performance**
       - **69% Recall**: Captures most high-risk patients
       - **15% Precision**: Balance for resource constraints
       - **Optimized** for Kenyan healthcare context
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("Kenya Hospital Readmission Predictor • v1.0 • For clinical decision support")
