import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# Load Model & Scaler
# ---------------------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("best_log_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_assets()


def main():

    # ---------------------------
    # Sidebar Inputs
    # ---------------------------
    st.sidebar.title("Patient Clinical Inputs")

    age = st.sidebar.number_input("Age (Years)", value=45, step=1)
    sex = st.sidebar.selectbox("Sex", ["Female (0)", "Male (1)"])
    cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.sidebar.number_input("Resting BP (mm Hg)", value=120)
    chol = st.sidebar.number_input("Cholesterol (mg/dl)", value=230)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.sidebar.number_input("Max Heart Rate", value=150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina", [0, 1])
    oldpeak = st.sidebar.number_input("ST Depression", value=1.0)
    slope = st.sidebar.selectbox("Slope (0–2)", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia (1–3)", [1, 2, 3])

    # ---------------------------
    # Derived Indicators
    # ---------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Derived Clinical Indicators")

    # Risk Index (simple clinical signal combining HR & Depression)
    risk_index = (oldpeak * 10) + (220 - thalach)
    risk_norm = int(min(risk_index / 200 * 100, 100))

    st.sidebar.write(f"Risk Index Score: **{risk_index:.2f}**")
    st.sidebar.progress(risk_norm)

    # Convert sex to numeric
    sex_val = 1 if "Male" in sex else 0

    # ---------------------------
    # Prepare Input Dataframe
    # ---------------------------
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex_val],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    # Align with scaler (if needed)
    try:
        expected_features = scaler.feature_names_in_
        input_data = input_data.reindex(columns=expected_features)
    except AttributeError:
        pass

    # ---------------------------
    # Main UI
    # ---------------------------
    st.title("Heart Disease Prediction Dashboard")
    st.markdown("""
    This system predicts the likelihood of **heart disease**  
    based on key clinical measurements and risk indicators.
    """)

    st.markdown("### Input Overview")
    st.dataframe(input_data, use_container_width=True)
    st.markdown("---")

    # ---------------------------
    # Prediction
    # ---------------------------
    if st.button("Run Heart Disease Prediction", use_container_width=True):

        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        probabilities = model.predict_proba(scaled_input)[0]

        p_no_hd, p_hd = probabilities[0], probabilities[1]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f" High Likelihood of Heart Disease\nProbability: {p_hd:.4f}")
        else:
            st.success(f" Low Likelihood of Heart Disease\nProbability: {p_no_hd:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("No Disease Probability", f"{p_no_hd:.2%}")
        with col2:
            st.metric("Heart Disease Probability", f"{p_hd:.2%}")

        st.info("Chest pain type, ST depression, and maximum heart rate strongly influence cardiac risk.")

    # ---------------------------
    # Footer
    # ---------------------------
    st.markdown("""
    ---
    Developed by: Ben  
    Model: Optimized Logistic Regression  
    Purpose: Early Heart Disease Risk Detection  
    """)


if __name__ == "__main__":
    main()
