# app.py
import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("🔍 Creditworthiness Prediction App")
st.write("This app predicts whether a person is **creditworthy** or **not**, based on financial attributes.")

# Load model
model = joblib.load("random_forest_credit_model.joblib")

# Input fields
def user_input():
    Status = st.selectbox("Status of existing checking account", [0, 1, 2, 3])
    Duration = st.slider("Duration in months", 4, 72, 24)
    CreditHistory = st.selectbox("Credit history", [0, 1, 2, 3, 4])
    Purpose = st.selectbox("Purpose of credit", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    CreditAmount = st.number_input("Credit Amount", 250, 20000, 5000)
    Savings = st.selectbox("Savings account/bonds", [0, 1, 2, 3, 4])
    Employment = st.selectbox("Employment", [0, 1, 2, 3, 4])
    InstallmentRate = st.slider("Installment rate (% of income)", 1, 4, 2)
    PersonalStatusSex = st.selectbox("Personal status and sex", [0, 1, 2, 3])
    OtherDebtors = st.selectbox("Other debtors/guarantors", [0, 1, 2])
    ResidenceSince = st.slider("Present residence since (years)", 1, 4, 2)
    Property = st.selectbox("Property", [0, 1, 2, 3])
    Age = st.slider("Age", 18, 75, 35)
    OtherInstallmentPlans = st.selectbox("Other installment plans", [0, 1, 2])
    Housing = st.selectbox("Housing", [0, 1, 2])
    ExistingCredits = st.slider("Number of existing credits", 1, 4, 1)
    Job = st.selectbox("Job", [0, 1, 2, 3])
    NumPeopleMaintenance = st.selectbox("Number of people being maintained", [1, 2])
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

if st.button("Predict Creditworthiness"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Creditworthy: The applicant is likely to repay the loan.")
    else:
        st.error("❌ Not Creditworthy: The applicant is at risk of default.")
