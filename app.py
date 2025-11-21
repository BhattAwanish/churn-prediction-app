# app.py — Premium Cyberpunk Churn App (CLEAN & UPDATED)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ------------------------
# Helpers
# ------------------------
def load_lottie(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ------------------------
# Load Model + Scaler
# ------------------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

gender_map = {"Female": 0, "Male": 1}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}

# ------------------------
# Streamlit Config
# ------------------------
st.set_page_config(
    page_title="Premium Churn App",
    page_icon="💎",
    layout="wide",
)

# ------------------------
# Cyberpunk CSS
# ------------------------
st.markdown(
    """
<style>
body {
    background: radial-gradient(circle at center, #000012 0%, #000008 40%, #000000 100%),
                repeating-linear-gradient(90deg, rgba(0,255,255,0.06) 0px, rgba(0,255,255,0.06) 2px, transparent 2px, transparent 40px),
                repeating-linear-gradient(0deg, rgba(255,0,255,0.04) 0px, rgba(255,0,255,0.04) 2px, transparent 2px, transparent 40px);
    animation: backgroundShift 18s linear infinite;
    color: #0ff;
}
@keyframes backgroundShift {
    0% { background-position: 0 0, 0 0, 0 0; }
    100% { background-position: 0 0, 200px 400px, 400px 200px; }
}

.card, .neon-card, .cyber-card {
    padding: 22px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
}

.neon-card {
    background: rgba(0, 0, 0, 0.45);
    border: 2px solid rgba(0, 255, 255, 0.7);
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.6), 0 0 40px rgba(0, 255, 255, 0.35);
}

.cyber-card {
    background: rgba(0, 0, 20, 0.65);
    border: 2px solid rgba(0,255,255,0.18);
    box-shadow: 0 0 22px rgba(0,255,255,0.2), inset 0 0 30px rgba(0,255,255,0.04);
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    padding: 12px 20px;
    border-radius: 12px;
    font-size: 16px;
    border: none;
    transition: 0.25s;
}
div[data-testid="stButton"] > button:hover {
    transform: scale(1.04);
    filter: brightness(1.08);
}

.neon-title {
    font-size: 40px;
    font-weight: 900;
    color: #0ff;
    text-shadow: 0 0 18px #0ff, 0 0 38px #09f, 0 0 80px #0ff;
    text-align: center;
    margin-bottom: 10px;
}

/* Matrix rain */
.rain { position: fixed; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; pointer-events: none; z-index: -1; }
.rain span {
    position: absolute;
    top: -200px;
    width: 2px;
    height: 220px;
    background: linear-gradient(to bottom, transparent, rgba(0,255,180,0.6));
    animation: drop 3s linear infinite;
}
@keyframes drop {
    0% { transform: translateY(0); }
    100% { transform: translateY(120vh); }
}
</style>
""",
    unsafe_allow_html=True,
)

# Rain effect
rain_html = "<div class='rain'>"
for i in range(1, 31):
    rain_html += f"<span style='left:{i*3.2}%; --i:{i};'></span>"
rain_html += "</div>"
st.markdown(rain_html, unsafe_allow_html=True)

# ------------------------
# Sidebar & Navigation
# ------------------------
st.sidebar.title("⚙️ Settings")
page = st.sidebar.selectbox("Navigation", ["Predict", "About App"])

# ------------------------
# Predict Page
# ------------------------
if page == "Predict":
    st.markdown("<div class='neon-title'>💎 Premium Churn Prediction</div>", unsafe_allow_html=True)

    st.markdown("<div class='neon-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    with col2:
        total = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    st.markdown("</div>", unsafe_allow_html=True)

    df = pd.DataFrame({
        "gender": [gender_map[gender]],
        "tenure": [tenure],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total],
        "InternetService": [internet_map[internet]],
        "Contract": [contract_map[contract]],
    })

    df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        df[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    if st.button("🔮 Predict Churn"):
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1] * 100

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        if pred == 1:
            st.markdown(f"<h2 style='color:#ff4b4b;'>⚠️ High Risk: {prob:.2f}%</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:#4bff8f;'>✅ Low Risk: {prob:.2f}%</h2>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# About App Page
# ------------------------
elif page == "About App":
    st.markdown("<h1 class='neon-title'>ℹ️ About This App</h1>", unsafe_allow_html=True)

    st.markdown(
        """
### **🔍 What This App Does**
This premium churn prediction system uses **machine learning** to determine whether a customer is likely to leave a service (churn).  
It has been optimized with:
- A trained **Random Forest / ML model**
- Proper **scaling and encoding**
- Accurate **probability-based predictions**

---

### **🎨 UI & Design Features**
- Full **Cyberpunk Neon Theme**
- Animated **Matrix–style background**
- Glassmorphism cards  
- Responsive layout  
- Premium visual effects  

---

### **🧠 Inside the ML Model**
The model was trained using:
- Tenure  
- Monthly Charges  
- Total Charges  
- Gender  
- Internet Type  
- Contract Type  

---

"""
    )
