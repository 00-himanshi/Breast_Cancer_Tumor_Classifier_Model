import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Breast Cancer Tumor Classifier",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------
# Load Model and Features
# -----------------------------------
model = joblib.load("best_breast_cancer_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# -----------------------------------
# Title
# -----------------------------------
st.title("🩺 Breast Cancer Tumor Classifier")
st.markdown("""
Predict whether a breast tumor is **Benign (Non-Cancerous)**
or **Malignant (Cancerous)** using a trained Machine Learning model.
""")

st.divider()

# -----------------------------------
# Sidebar
# -----------------------------------
st.sidebar.header("Prediction Mode")

mode = st.sidebar.radio(
    "Choose Prediction Type",
    [
        "Single Patient Prediction",
        "CSV Batch Prediction"
    ]
)

# ==========================================================
# SINGLE PATIENT PREDICTION
# ==========================================================
if mode == "Single Patient Prediction":

    st.subheader("Enter Patient Measurements")

    user_input = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(feature_names):

        if i % 2 == 0:
            with col1:
                user_input[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    format="%.4f"
                )
        else:
            with col2:
                user_input[feature] = st.number_input(
                    feature,
                    min_value=0.0,
                    value=0.0,
                    format="%.4f"
                )

    st.divider()

    if st.button("Predict Tumor Type"):

        input_df = pd.DataFrame([user_input])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        benign_prob = probability[0]
        malignant_prob = probability[1]

        st.subheader("Prediction Result")

        if prediction == 0:
            st.success(
                f"Prediction: Benign (Non-Cancerous)\n\n"
                f"Confidence: {benign_prob:.2%}"
            )
        else:
            st.error(
                f"Prediction: Malignant (Cancerous)\n\n"
                f"Confidence: {malignant_prob:.2%}"
            )

        st.subheader("Prediction Probabilities")

        st.write(f"Benign Probability: {benign_prob:.2%}")
        st.progress(float(benign_prob))

        st.write(f"Malignant Probability: {malignant_prob:.2%}")
        st.progress(float(malignant_prob))

# ==========================================================
# CSV BATCH PREDICTION
# ==========================================================
elif mode == "CSV Batch Prediction":

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.subheader("Uploaded Data")
            st.dataframe(uploaded_df)

            st.write("Detected Columns:")
            st.write(uploaded_df.columns.tolist())

            prediction_df = uploaded_df.copy()

            # Remove unnecessary columns if present
            prediction_df = prediction_df.drop(
                columns=[
                    "index",
                    "id",
                    "diagnosis",
                    "Prediction"
                ],
                errors="ignore"
            )

            # Check missing columns
            missing_columns = [
                col for col in feature_names
                if col not in prediction_df.columns
            ]

            if missing_columns:
                st.error(
                    f"Missing Columns: {missing_columns}"
                )
                st.stop()

            # Keep only model features
            prediction_df = prediction_df[
                feature_names
            ]

            # Predictions
            predictions = model.predict(
                prediction_df
            )

            probabilities = model.predict_proba(
                prediction_df
            )

            result_df = uploaded_df.copy()

            result_df["Prediction"] = [
                "Malignant" if p == 1
                else "Benign"
                for p in predictions
            ]

            result_df[
                "Benign Probability (%)"
            ] = (
                probabilities[:, 0] * 100
            ).round(2)

            result_df[
                "Malignant Probability (%)"
            ] = (
                probabilities[:, 1] * 100
            ).round(2)

            st.subheader("Prediction Results")
            st.dataframe(result_df)

            csv = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error while processing file: {e}")

# -----------------------------------
# Footer
# -----------------------------------
st.divider()

st.caption(
    """
    ⚠️ Disclaimer:
    This application is intended for educational and
    demonstration purposes only and should not be used
    as a substitute for professional medical advice.
    """
)