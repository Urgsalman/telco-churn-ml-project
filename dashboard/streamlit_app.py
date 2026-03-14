import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt

# ----------------------------
# Load model
# ----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "churn_model.pkl")
columns_path = os.path.join(BASE_DIR, "models", "model_columns.pkl")

model = pickle.load(open(model_path, "rb"))
model_columns = pickle.load(open(columns_path, "rb"))

# ----------------------------
# Page configuration
# ----------------------------

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.write(
"This application predicts whether a telecom customer is likely to churn "
"based on tenure, monthly charges and contract type using a Machine Learning model."
)

st.write("Enter customer information")

# ----------------------------
# User Inputs
# ----------------------------

tenure = st.slider("Tenure Months", 0, 72)

monthly_charges = st.slider("Monthly Charges", 0, 150)

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# ----------------------------
# Create input dataframe
# ----------------------------

data = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
}

input_df = pd.DataFrame([data])

# ----------------------------
# Recreate model feature space
# ----------------------------

input_full = pd.DataFrame(columns=model_columns)
input_full.loc[0] = 0

for col in input_df.columns:
    if col in input_full.columns:
        input_full[col] = input_df[col]

contract_col = f"Contract_{contract}"
if contract_col in input_full.columns:
    input_full[contract_col] = 1

payment_col = f"Payment Method_{payment}"
if payment_col in input_full.columns:
    input_full[payment_col] = 1

# ----------------------------
# Prediction
# ----------------------------

if st.button("Predict Churn"):

    prediction = model.predict(input_full)
    probability = model.predict_proba(input_full)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    st.write(f"Churn probability : {round(probability*100,2)} %")

    # ----------------------------
    # Probability Visualization
    # ----------------------------

    fig, ax = plt.subplots()

    ax.bar(["Stay","Churn"], [1-probability, probability])

    ax.set_ylabel("Probability")
    ax.set_title("Churn Probability")

    st.pyplot(fig)