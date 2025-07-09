 #app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("random_forest_credit_model.joblib")

st.set_page_config(page_title="Creditworthiness Prediction", layout="centered")
st.title("🔍 Creditworthiness Prediction App")
st.markdown("This app predicts whether a person is **creditworthy** based on financial attributes using a trained Random Forest model.")

# ----- Input Form -----
st.header("📋 Enter Applicant Details")

def user_input():
    Status = st.selectbox("Status of checking account", [0, 1, 2, 3])
    Duration = st.slider("Credit Duration (months)", 4, 72, 24)
    CreditHistory = st.selectbox("Credit History", [0, 1, 2, 3, 4])
    Purpose = st.selectbox("Purpose of credit", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    CreditAmount = st.number_input("Credit Amount", 250, 20000, 5000)
    Savings = st.selectbox("Savings Account", [0, 1, 2, 3, 4])
    Employment = st.selectbox("Years of Employment", [0, 1, 2, 3, 4])
    InstallmentRate = st.slider("Installment Rate (% of income)", 1, 4, 2)
    PersonalStatusSex = st.selectbox("Personal Status & Sex", [0, 1, 2, 3])
    OtherDebtors = st.selectbox("Other Debtors", [0, 1, 2])
    ResidenceSince = st.slider("Years at Residence", 1, 4, 2)
    Property = st.selectbox("Property", [0, 1, 2, 3])
    Age = st.slider("Age", 18, 75, 35)
    OtherInstallmentPlans = st.selectbox("Other Installment Plans", [0, 1, 2])
    Housing = st.selectbox("Housing", [0, 1, 2])
    ExistingCredits = st.slider("Number of Existing Credits", 1, 4, 1)
    Job = st.selectbox("Job", [0, 1, 2, 3])
    NumPeopleMaintenance = st.selectbox("Dependents Maintained", [1, 2])
    Telephone = st.selectbox("Telephone", [0, 1])
    ForeignWorker = st.selectbox("Foreign Worker", [0, 1])

    data = np.array([[
        Status, Duration, CreditHistory, Purpose, CreditAmount, Savings,
        Employment, InstallmentRate, PersonalStatusSex, OtherDebtors,
        ResidenceSince, Property, Age, OtherInstallmentPlans, Housing,
        ExistingCredits, Job, NumPeopleMaintenance, Telephone, ForeignWorker
    ]])
    return data

input_data = user_input()

# ----- Predict -----
if st.button("🔮 Predict Creditworthiness"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]  # Prob of class 1 (creditworthy)

    if prediction == 1:
        st.success("✅ The applicant is **Creditworthy**.")
    else:
        st.error("❌ The applicant is **Not Creditworthy**.")

    st.metric("Confidence (Creditworthy)", f"{proba:.2%}")

# ----- Plots -----
# ----- Optional Visualizations -----
st.markdown("---")
if st.checkbox("📊 Show Model Visualizations (Optional)"):
    st.header("📊 Model Insights")

    # 1. Feature Importance
    st.subheader("🔎 Top 10 Feature Importances")

    features = [
        "Status", "Duration", "CreditHistory", "Purpose", "CreditAmount",
        "Savings", "Employment", "InstallmentRate", "PersonalStatusSex",
        "OtherDebtors", "ResidenceSince", "Property", "Age",
        "OtherInstallmentPlans", "Housing", "ExistingCredits",
        "Job", "NumPeopleMaintenance", "Telephone", "ForeignWorker"
    ]

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values("Importance", ascending=False).head(10)

    fig1, ax1 = plt.subplots()
    sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis", ax=ax1)
    st.pyplot(fig1)

    # 2. Class Distribution (Optional)
    try:
        df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data",
                         delimiter=' ', header=None)
        df.columns = features + ["Target"]
        st.subheader("📈 Original Dataset: Class Distribution")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x="Target", ax=ax2)
        ax2.set_xticklabels(["Not Creditworthy (2)", "Creditworthy (1)"])
        st.pyplot(fig2)
    except:
        st.info("Class distribution plot skipped (dataset not loaded).")


   
       
