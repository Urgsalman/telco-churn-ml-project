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

st.set_page_config(page_title="Churn Dashboard", layout="wide")

# ----------------------------
# Title & Description
# ----------------------------

st.title("📊 Customer Churn Prediction Dashboard")

st.info(
    "This application predicts whether a telecom customer is likely to churn "
    "using a Machine Learning model."
)

st.markdown("---")

# ----------------------------
# Layout (2 columns)
# ----------------------------

col1, col2 = st.columns(2)

# ----------------------------
# INPUT SECTION (LEFT)
# ----------------------------

with col1:
    st.subheader("📥 Enter Customer Information")

    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 50)

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    )

    st.markdown("")

    predict_button = st.button("🚀 Predict Churn")

# ----------------------------
# DATA PREPARATION
# ----------------------------

data = {
    "Tenure Months": tenure,
    "Monthly Charges": monthly_charges,
}

input_df = pd.DataFrame([data])

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
# RESULT SECTION (RIGHT)
# ----------------------------

with col2:
    st.subheader("📊 Prediction Result")

    if predict_button:

        prediction = model.predict(input_full)
        probability = model.predict_proba(input_full)[0][1]

        # ----------------------------
        # Result Message
        # ----------------------------

        if prediction[0] == 1:
            st.error("⚠️ High churn risk: Customer likely to leave")
        else:
            st.success("✅ Low churn risk: Customer likely to stay")

        # ----------------------------
        # Probability Display
        # ----------------------------

        st.metric("Churn Probability", f"{round(probability * 100, 2)} %")

        st.markdown("")

        # ----------------------------
        # Chart
        # ----------------------------

        fig, ax = plt.subplots()

        ax.bar(
            ["Stay", "Churn"],
            [1 - probability, probability]
        )

        ax.set_ylabel("Probability")
        ax.set_title("Churn Probability Distribution")

        st.pyplot(fig)

    else:
        st.info("Enter customer details and click 'Predict Churn'")

# ----------------------------
# Footer / Insight
# ----------------------------

st.markdown("---")

st.subheader("💡 Insights")

st.markdown("""
- A higher churn probability indicates a higher risk customer  
- Businesses can take proactive actions (offers, support, retention strategies)  
""")