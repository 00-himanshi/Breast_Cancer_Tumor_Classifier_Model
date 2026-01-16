# breast_cancer_app.py
import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Title
# -------------------------
st.title("Breast Cancer Tumor Prediction")
st.write("""
Upload a CSV file with **30 features** to predict whether the tumor is **Benign** or **Malignant** using Logistic Regression.
""")


# -------------------------
# Load trained model and scaler
# -------------------------
@st.cache_resource
def load_model():
    model = joblib.load("Logistic_Regression.pkl")  # your trained Logistic Regression model
    scaler = joblib.load("scaler.pkl")  # your fitted StandardScaler
    return model, scaler


model, scaler = load_model()

# -------------------------
# Upload CSV
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        user_data = pd.read_csv(uploaded_file)
        st.write("Input data preview:")
        st.dataframe(user_data.head())

        # -------------------------
        # Scale features
        # -------------------------
        user_data_scaled = scaler.transform(user_data)

        # -------------------------
        # Make predictions
        # -------------------------
        prediction = model.predict(user_data_scaled)
        prediction_label = ["Benign" if i == 0 else "Malignant" for i in prediction]

        # -------------------------
        # Display results
        # -------------------------
        result_df = user_data.copy()
        result_df['Prediction'] = prediction_label

        st.write("Prediction Results:")
        st.dataframe(result_df)

        # Optional: Download predictions as CSV
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

    except Exception as e:
        st.error(f"Error processing the file: {e}")
