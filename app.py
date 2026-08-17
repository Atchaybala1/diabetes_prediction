import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return joblib.load("diabetes_prediction_model.joblib")

model = load_model()

# Title
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's information below to predict the diabetes risk.")

st.divider()

# Input fields
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

st.divider()

# Prediction button
if st.button("🔍 Predict Diabetes", use_container_width=True):

    # Create DataFrame in the exact feature order used during training
    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    # Prediction
    prediction = model.predict(input_data)[0]

    # Prediction probability
    probability = model.predict_proba(input_data)[0]

    diabetes_probability = probability[1] * 100
    no_diabetes_probability = probability[0] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Diabetes")
    else:
        st.success("✅ Prediction: No Diabetes")

    st.write(
        f"**Diabetes probability:** {diabetes_probability:.2f}%"
    )

    st.write(
        f"**No diabetes probability:** {no_diabetes_probability:.2f}%"
    )

    st.progress(int(diabetes_probability))

    st.info(
        "This prediction is for educational purposes only and "
        "should not be considered a medical diagnosis."
    )