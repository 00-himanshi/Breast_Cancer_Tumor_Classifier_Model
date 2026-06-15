# 🩺 Breast Cancer Tumor Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikitlearn)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide, and early detection significantly improves treatment outcomes. This project uses Machine Learning techniques to classify breast tumors as:

* ✅ **Benign (Non-Cancerous)**
* ❌ **Malignant (Cancerous)**

The project includes:

* End-to-end Machine Learning pipeline
* Exploratory Data Analysis (EDA)
* Feature preprocessing and scaling
* Model comparison and evaluation
* Trained classification model
* Interactive Streamlit web application
* Single prediction and batch prediction support

---

## 🎯 Objectives

The primary objectives of this project are:

* Build a reliable breast tumor classification model.
* Perform detailed exploratory data analysis.
* Compare multiple machine learning algorithms.
* Prevent data leakage and overfitting.
* Deploy the trained model through an interactive web application.
* Demonstrate an end-to-end machine learning workflow suitable for production and portfolio projects.

---

## 📊 Dataset Information

The project uses the Breast Cancer Wisconsin Diagnostic Dataset.

### Dataset Characteristics

* Number of samples: 569
* Number of features: 30
* Target classes:

  * Benign
  * Malignant

### Feature Categories

The dataset contains numerical measurements computed from digitized images of breast masses, including:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

Each feature also contains:

* Mean values
* Standard error values
* Worst-case values

---

## 🔍 Exploratory Data Analysis (EDA)

The project includes extensive EDA, including:

* Dataset inspection
* Missing value analysis
* Duplicate checking
* Statistical summary
* Target distribution analysis
* Correlation heatmaps
* Feature relationship analysis
* Distribution plots
* Pairwise feature exploration
* Outlier analysis

Since this is medical data, outlier handling was performed conservatively to avoid removing clinically significant observations.

---

## ⚙️ Data Preprocessing Pipeline

The preprocessing pipeline consists of:

### Missing Value Handling

* Median Imputation using `SimpleImputer`

### Feature Scaling

* Standardization using `StandardScaler`

### Pipeline Construction

A Scikit-Learn Pipeline was created to ensure:

* Reproducibility
* Prevention of data leakage
* Clean and maintainable code
* Production-ready inference pipeline

---

## 🤖 Machine Learning Models Evaluated

Multiple classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Machine (SVM)
4. K-Nearest Neighbors (KNN)
5. Decision Tree Classifier
6. Gradient Boosting Classifier

Models were compared using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Cross Validation Performance

---

## 🏆 Final Model Selection

### Final Model

**Logistic Regression**

### Why Logistic Regression?

* Strong generalization performance
* Stable decision boundaries
* Lower risk of overfitting
* High interpretability
* Excellent performance on structured tabular medical data

---

## 🧠 Final Pipeline Architecture

SimpleImputer (Median Strategy)
↓
StandardScaler
↓
LogisticRegression

---

## 📈 Model Evaluation Metrics

Evaluation metrics considered:

* Accuracy Score
* Precision Score
* Recall Score
* F1 Score
* Confusion Matrix
* ROC Curve
* ROC-AUC Score
* Cross Validation Metrics

The final model demonstrated strong predictive performance while maintaining good generalization capabilities.

---

## 🚀 Streamlit Web Application Features

### Single Patient Prediction

* Manual feature input
* Tumor prediction
* Prediction confidence scores

### Batch Prediction

* CSV upload support
* Predict multiple samples simultaneously
* Download prediction results

### Probability Estimates

The application displays:

* Benign probability
* Malignant probability
* Model confidence

---

## 📂 Project Structure

```text
Breast_Cancer_Tumor_Classifier_Model/
│
├── app.py
├── best_breast_cancer_model.pkl
├── feature_names.pkl
├── Breast_Cancer_Final_Professional.ipynb
├── synthetic_breast_cancer_4000.csv
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 🛠️ Installation

### Clone Repository

```bash
git clone https://github.com/00-himanshi/Breast_Cancer_Tumor_Classifier_Model.git
cd Breast_Cancer_Tumor_Classifier_Model
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit Application

```bash
streamlit run app.py
```

---

## 📌 Important Note

The trained model expects **original breast cancer measurements** as input.

Do not provide already standardized or scaled values because scaling is automatically handled inside the preprocessing pipeline.

---

## 💡 Skills Demonstrated

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Statistical Analysis
* Feature Engineering
* Machine Learning
* Model Evaluation
* Scikit-Learn Pipelines
* Data Leakage Prevention
* Overfitting Analysis
* Model Serialization
* Streamlit Deployment
* Git and GitHub
* End-to-End Machine Learning Project Development

---

## ⚠️ Disclaimer

This project is intended solely for educational, research, and demonstration purposes.

The predictions generated by this application should not be considered medical advice, diagnosis, or treatment recommendations. Always consult qualified healthcare professionals for medical decisions.

---

## 👩‍💻 Author

**Himanshi Sharma**

Data Science | Machine Learning | Python | AI Enthusiast

GitHub: https://github.com/00-himanshi
