import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Smart Farming", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1, h2, h3, h4 {
    color: #2e7d32;
    font-family: 'Segoe UI', sans-serif;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}

.stSidebar {
    background-color: #1b5e20;
}

.stSidebar label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style='background: linear-gradient(90deg, #2e7d32, #66bb6a);
padding:25px;border-radius:15px;color:white'>
<h1>🌱 AI Smart Farming Advisor</h1>
<p>Smart crop recommendation using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- LOAD DATA ----------------
data = pd.read_csv("Crop_recommendation.csv")

# ---------------- MODEL ----------------
X = data[['N','P','K','temperature','humidity','ph','rainfall']]
y = data['label']

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Input Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 50)
K = st.sidebar.slider("Potassium (K)", 0, 150, 50)
temp = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 100)

# ---------------- BUTTON ----------------
if st.sidebar.button("🚀 Predict Crop"):

    sample = [[N, P, K, temp, humidity, ph, rainfall]]

    prediction = model.predict(sample)[0]
    confidence = max(model.predict_proba(sample)[0])

    # ---------------- RESULT CARDS ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style='background:white;padding:25px;border-radius:15px;
        box-shadow:0 4px 10px rgba(0,0,0,0.1)'>
            <h4>🌾 Recommended Crop</h4>
            <h1 style='color:#2e7d32'>{prediction}</h1>
            <p style='color:gray'>Best match for your conditions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style='background:white;padding:25px;border-radius:15px;
        box-shadow:0 4px 10px rgba(0,0,0,0.1)'>
            <h4>📊 Confidence Level</h4>
            <h1 style='color:#1565c0'>{round(confidence*100,2)}%</h1>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- PROGRESS BAR ----------------
    st.markdown("### 📊 Confidence Score")
    st.progress(int(confidence * 100))

    # ---------------- WHY THIS CROP ----------------
    st.markdown("### 🧠 Why this crop?")

    reasons = []

    if rainfall > 150:
        reasons.append("High rainfall is ideal")
    if 6 <= ph <= 7:
        reasons.append("Soil pH is optimal")
    if temp > 25:
        reasons.append("Warm temperature supports growth")

    for r in reasons:
        st.markdown(f"""
        <div style='background:#e8f5e9;padding:10px;border-radius:10px;margin-bottom:5px'>
        ✔ {r}
        </div>
        """, unsafe_allow_html=True)

    # ---------------- GRAPH ----------------
    st.markdown("### 📈 Feature Importance")

    importance = model.feature_importances_

    fig = px.bar(
        x=importance,
        y=X.columns,
        orientation='h',
        color=importance,
        color_continuous_scale="greens"
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    # ---------------- FOOTER ----------------
    st.success("✅ Prediction completed successfully using AI model")