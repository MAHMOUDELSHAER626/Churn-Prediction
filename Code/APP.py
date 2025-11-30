import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import time

# Set page configuration
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS: CYBER-FUTURE AI THEME ---
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');

    /* Global Colors */
    :root {
        --primary-color: #00f260;
        --secondary-color: #0575E6;
        --bg-color: #0f0c29;
        --card-bg: rgba(255, 255, 255, 0.05);
        --text-color: #ffffff;
    }

    /* Background Gradient */
    .stApp {
        background: linear-gradient(to bottom, #0f0c29, #302b63, #24243e);
        color: var(--text-color);
        font-family: 'Roboto', sans-serif;
    }

    /* Headings */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        background: linear-gradient(to right, #00f260, #0575E6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 0px 10px rgba(0, 242, 96, 0.3);
    }
    
    h1 {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 10px;
        letter-spacing: 2px;
    }
    
    .stMarkdown p {
        color: #d1d5db;
        font-size: 1.1rem;
    }

    /* Input Styling - Dark Tech Look */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #00f260 !important;
        font-weight: bold;
        letter-spacing: 1px;
        font-family: 'Orbitron', sans-serif;
    }
    
    /* Style the inner input boxes */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput input {
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: white !important;
        border: 1px solid #302b63 !important;
        border-radius: 8px !important;
    }
    
    /* Cards/Columns Styling */
    div[data-testid="column"] {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    div[data-testid="column"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 0 20px rgba(5, 117, 230, 0.4);
        border-color: var(--secondary-color);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00f260 0%, #0575E6 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-family: 'Orbitron', sans-serif;
        font-size: 18px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-radius: 50px;
        box-shadow: 0 0 15px rgba(0, 242, 96, 0.5);
        transition: all 0.4s ease;
        width: 100%;
        margin-top: 20px;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(5, 117, 230, 0.8);
        background: linear-gradient(90deg, #0575E6 0%, #00f260 100%);
    }

    /* Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        animation: fadeIn 1s ease-in-out;
        margin-top: 30px;
        border: 2px solid;
    }
    
    .danger-glow {
        background: rgba(255, 0, 0, 0.1);
        border-color: #ff3333;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.4);
    }
    
    .safe-glow {
        background: rgba(0, 255, 0, 0.1);
        border-color: #00f260;
        box-shadow: 0 0 30px rgba(0, 242, 96, 0.4);
    }
    
    .result-title {
        font-size: 2rem;
        font-family: 'Orbitron', sans-serif;
        margin-bottom: 10px;
    }
    
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ System Config")
    show_debug = st.checkbox("🔍 Debug Mode", value=False)
    st.info("System Version: 2.0 (Cyber-Glass UI)")

# --- 1. Load Models ---
@st.cache_resource
def load_models():
    # Make sure these files exist in the same directory!
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model_gradient_boosting.pkl', 'rb') as f:
        model = pickle.load(f)
    return scaler, model

try:
    scaler, model = load_models()
except Exception as e:
    st.error(f"System Error: Models could not be loaded. Please ensure 'scaler.pkl' and 'model_gradient_boosting.pkl' are in the same folder. Error details: {e}")
    st.stop()

# --- 2. User Input Form ---
st.markdown("<h1>CHURN PREDICTION AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; margin-bottom: 50px;'>Advanced Neural Analysis for Customer Retention</p>", unsafe_allow_html=True)

# Use columns for layout
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 IDENTITY")
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])

with col2:
    st.markdown("### 🛠️ SERVICES")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    
    with st.expander("➕ Additional Add-ons"):
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

with col3:
    st.markdown("### 💳 BILLING")
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0, step=0.5)

# --- 3. Preprocessing Function (Strictly preserving logic) ---
def preprocess_input(data, model_features, scaler):
    df_input = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
    
    # Scaling Logic
    df_input['tenure'] = float(data['tenure']) / 72.0
    df_input['MonthlyCharges'] = scaler.transform([[data['monthly_charges']]])[0][0]

    # One-Hot Encoding Logic (Drop First)
    if data['senior_citizen'] == 'Yes' and 'SeniorCitizen_1' in model_features: df_input['SeniorCitizen_1'] = 1
    if data['partner'] == 'Yes' and 'Partner_Yes' in model_features: df_input['Partner_Yes'] = 1
    if data['dependents'] == 'Yes' and 'Dependents_Yes' in model_features: df_input['Dependents_Yes'] = 1
        
    if data['multiple_lines'] == 'Yes' and 'MultipleLines_Yes' in model_features: df_input['MultipleLines_Yes'] = 1
    elif data['multiple_lines'] == 'No phone service' and 'MultipleLines_No phone service' in model_features: df_input['MultipleLines_No phone service'] = 1
        
    if data['internet_service'] == 'Fiber optic' and 'InternetService_Fiber optic' in model_features: df_input['InternetService_Fiber optic'] = 1
    elif data['internet_service'] == 'No' and 'InternetService_No' in model_features: df_input['InternetService_No'] = 1
        
    if data['online_security'] == 'Yes' and 'OnlineSecurity_Yes' in model_features: df_input['OnlineSecurity_Yes'] = 1
    elif data['online_security'] == 'No internet service' and 'OnlineSecurity_No internet service' in model_features: df_input['OnlineSecurity_No internet service'] = 1

    if data['online_backup'] == 'Yes' and 'OnlineBackup_Yes' in model_features: df_input['OnlineBackup_Yes'] = 1
    elif data['online_backup'] == 'No internet service' and 'OnlineBackup_No internet service' in model_features: df_input['OnlineBackup_No internet service'] = 1

    if data['device_protection'] == 'Yes' and 'DeviceProtection_Yes' in model_features: df_input['DeviceProtection_Yes'] = 1
    elif data['device_protection'] == 'No internet service' and 'DeviceProtection_No internet service' in model_features: df_input['DeviceProtection_No internet service'] = 1

    if data['tech_support'] == 'Yes' and 'TechSupport_Yes' in model_features: df_input['TechSupport_Yes'] = 1
    elif data['tech_support'] == 'No internet service' and 'TechSupport_No internet service' in model_features: df_input['TechSupport_No internet service'] = 1

    if data['streaming_tv'] == 'Yes' and 'StreamingTV_Yes' in model_features: df_input['StreamingTV_Yes'] = 1
    elif data['streaming_tv'] == 'No internet service' and 'StreamingTV_No internet service' in model_features: df_input['StreamingTV_No internet service'] = 1

    if data['streaming_movies'] == 'Yes' and 'StreamingMovies_Yes' in model_features: df_input['StreamingMovies_Yes'] = 1
    elif data['streaming_movies'] == 'No internet service' and 'StreamingMovies_No internet service' in model_features: df_input['StreamingMovies_No internet service'] = 1

    if data['paperless_billing'] == 'Yes' and 'PaperlessBilling_Yes' in model_features: df_input['PaperlessBilling_Yes'] = 1
        
    if data['payment_method'] == 'Electronic check' and 'PaymentMethod_Electronic check' in model_features: df_input['PaymentMethod_Electronic check'] = 1
    elif data['payment_method'] == 'Mailed check' and 'PaymentMethod_Mailed check' in model_features: df_input['PaymentMethod_Mailed check'] = 1
    elif data['payment_method'] == 'Credit card (automatic)' and 'PaymentMethod_Credit card (automatic)' in model_features: df_input['PaymentMethod_Credit card (automatic)'] = 1

    return df_input

# --- 4. Prediction Logic ---
st.markdown("<br>", unsafe_allow_html=True)
if st.button("INITIATE ANALYSIS"):
    
    with st.spinner("Processing Neural Pathways..."):
        time.sleep(1)
        
        user_input = {
            'senior_citizen': senior_citizen,
            'partner': partner,
            'dependents': dependents,
            'tenure': tenure,
            'multiple_lines': multiple_lines,
            'internet_service': internet_service,
            'online_security': online_security,
            'online_backup': online_backup,
            'device_protection': device_protection,
            'tech_support': tech_support,
            'streaming_tv': streaming_tv,
            'streaming_movies': streaming_movies,
            'paperless_billing': paperless_billing,
            'payment_method': payment_method,
            'monthly_charges': monthly_charges
        }

        model_features = model.feature_names_in_
        processed_data = preprocess_input(user_input, model_features, scaler)

        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[0][1]

        st.markdown("---")
        
        if prediction[0] == 1:
            st.markdown(f"""
                <div class="result-card danger-glow">
                    <div class="result-title">⚠️ HIGH CHURN RISK DETECTED</div>
                    <div style="font-size: 24px; margin: 15px 0; color: #ff4d4d;">PROBABILITY: {probability:.1%}</div>
                    <p style="color: #ccc;">Customer behavior indicates a high likelihood of contract termination.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.balloons()
            st.markdown(f"""
                <div class="result-card safe-glow">
                    <div class="result-title">✅ CUSTOMER SECURE</div>
                    <div style="font-size: 24px; margin: 15px 0; color: #00f260;">PROBABILITY: {probability:.1%}</div>
                    <p style="color: #ccc;">Customer retention metrics are stable.</p>
                </div>
            """, unsafe_allow_html=True)
            
        if show_debug:
            with st.expander("🔍 SYSTEM DIAGNOSTICS"):
                st.write("**Active Feature Signals:**")
                st.dataframe(processed_data.loc[:, (processed_data != 0).any(axis=0)])