Breast Cancer Tumor Classification

Project Overview

This project aims to **classify breast tumors as Malignant or Benign** using supervised machine learning techniques. Early and accurate detection of breast cancer is critical for effective treatment and patient care. This project leverages clinical and diagnostic features extracted from breast tumor samples to build predictive models that assist healthcare professionals in making informed decisions.

The project includes:

* Data preprocessing and feature scaling
* Model training using Logistic Regression
* Evaluation of model performance
* Deployment-ready **Streamlit application** for interactive tumor classification
  
Dataset Description:

The **Breast Cancer Dataset** contains clinical and diagnostic features extracted from breast tumor samples. It is designed to support **machine learning and data science research in healthcare** ⚕️📊.

**Key characteristics:**

* Each record represents a patient case.
* Features include measurements of tumor characteristics such as **size, texture, shape, smoothness, and concavity**.
* The target variable classifies tumors as **Malignant** or **Benign**.
* The dataset is widely used in **medical diagnosis and cancer research** 🧪🩺.

**Objective:**

* Predict whether a tumor is **Malignant** or **Benign** using machine learning models.

---

## Features

The dataset contains **30 numerical features** extracted from tumor images, including:

* Mean measurements (`radius_mean`, `texture_mean`, `perimeter_mean`, etc.)
* Standard error measurements (`radius_se`, `texture_se`, `perimeter_se`, etc.)
* Worst-case measurements (`radius_worst`, `texture_worst`, `perimeter_worst`, etc.)

These features capture important characteristics of the tumors and are essential for accurate classification.

---

## Model

* **Algorithm used:** Logistic Regression
* **Performance Metrics:**

  * Accuracy: ~97.87%
  * F1 Score: ~97.06%
* **Preprocessing:**

  * Feature scaling using `StandardScaler`
  * No handling of outliers (considered as meaningful clinical data)

---

## Streamlit Application

An interactive **Streamlit web application** has been developed for this project:

* Users can **upload a CSV file** with 30 features.
* The application **predicts whether each tumor is Malignant or Benign**.
* Results can be **viewed and downloaded** as a CSV file for further analysis.

**How it works:**

1. User uploads a CSV file containing tumor features.
2. Features are scaled using the same scaler used during training.
3. Logistic Regression model predicts tumor type.
4. Predictions are displayed in the app and can be downloaded.


## Conclusion

This project demonstrates a **practical application of machine learning in healthcare**. By classifying breast tumors accurately, it supports **early detection and informed diagnostic decisions**, contributing to better patient outcomes.

## Acknowledgements

* Dataset: [Breast Cancer Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
* Libraries: `scikit-learn`, `pandas`, `numpy`, `streamlit`, `joblib`

