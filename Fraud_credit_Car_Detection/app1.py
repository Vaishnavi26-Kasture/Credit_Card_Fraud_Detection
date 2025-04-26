import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# Title
st.title("💳 Credit Card Fraud Detection App")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()

# Show sample data
if st.checkbox("Show Sample Data"):
    st.write(data.head())

# Prepare data
X = data.drop(columns=["Class"])
y = data["Class"]

# Train model
model = IsolationForest(n_estimators=100, random_state=42)
model.fit(X)

# Important features with explanations
important_features = {
    "Time": "Time since the first transaction (in seconds)",
    "V1": "Anonymized PCA component 1",
    "V2": "Anonymized PCA component 2",
    "V3": "Anonymized PCA component 3",
    "Amount": "Transaction amount in USD"
}

# User input section
st.subheader("🔒 Enter Transaction Details (All inputs are hidden)")

input_data = {}

for feature, explanation in important_features.items():
    user_input = st.text_input(
        label=f"{feature} ({explanation})",
        value=str(round(X[feature].mean(), 2)),
        type="password"
    )
    try:
        input_data[feature] = float(user_input)
    except ValueError:
        st.warning(f"Invalid input for {feature}. Using default mean value.")
        input_data[feature] = X[feature].mean()

# Fill rest with mean
for col in X.columns:
    if col not in input_data:
        input_data[col] = X[col].mean()

# Predict
if st.button("Check Fraud"):
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(input_array)
    result = "🚨 Fraud Detected!" if prediction[0] == -1 else "✅ Transaction is Normal"
    st.success(result)

