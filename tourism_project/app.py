import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# ── 1. Load saved model from HF ──
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="divyabhangre/tourism-package-predict",
        filename="best_model.pkl",
        repo_type="space",
        token=os.getenv("HF_TOKEN")
    )
    return joblib.load(model_path)

model = load_model()

# ── 2. App Title ──
st.title("🏖️ Tourism Package Prediction")
st.markdown("Fill in the customer details below to predict if they will take a tourism package.")

# ── 3. Input Fields ──
st.subheader("👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    type_of_contact = st.selectbox("Type of Contact", [0, 1])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    duration_of_pitch = st.number_input("Duration of Pitch", min_value=0, max_value=100, value=10)
    occupation = st.selectbox("Occupation", [0, 1, 2, 3])
    gender = st.selectbox("Gender", [0, 1])

with col2:
    number_of_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
    product_pitched = st.selectbox("Product Pitched", [0, 1, 2, 3, 4])
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    marital_status = st.selectbox("Marital Status", [0, 1, 2, 3])
    number_of_trips = st.number_input("Number of Trips", min_value=0, max_value=20, value=2)

st.subheader("💰 Financial Details")
col3, col4 = st.columns(2)

with col3:
    passport = st.selectbox("Passport", [0, 1])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    own_car = st.selectbox("Own Car", [0, 1])

with col4:
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
    designation = st.selectbox("Designation", [0, 1, 2, 3, 4])
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=20000)

# ── 4. Predict ──
if st.button("🔍 Predict", use_container_width=True):
    input_df = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch,
        "Occupation": occupation,
        "Gender": gender,
        "NumberOfPersonVisiting": number_of_persons_visiting,
        "NumberOfFollowups": number_of_followups,
        "ProductPitched": product_pitched,
        "PreferredPropertyStar": preferred_property_star,
        "MaritalStatus": marital_status,
        "NumberOfTrips": number_of_trips,
        "Passport": passport,
        "PitchSatisfactionScore": pitch_satisfaction_score,
        "OwnCar": own_car,
        "NumberOfChildrenVisiting": number_of_children_visiting,
        "Designation": designation,
        "MonthlyIncome": monthly_income
    }])

    try:
        prediction = model.predict(input_df)
        result = "✅ Will Take Package" if prediction[0] == 1 else "❌ Will Not Take Package"
        st.success(result)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
