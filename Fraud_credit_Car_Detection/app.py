import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score

# Title
st.title("Credit Card Fraud Detection")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data

data = load_data()
st.subheader("Dataset Sample")
st.write(data.head())

# Show basic EDA
st.subheader("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Class', data=data, ax=ax)
st.pyplot(fig)

# Model selection
model_choice = st.selectbox("Select Model", ("Isolation Forest", "Local Outlier Factor", "One-Class SVM"))

if st.button("Run Detection"):
    X = data.drop(columns=["Class"])
    y = data["Class"]

    if model_choice == "Isolation Forest":
        model = IsolationForest(n_estimators=100)
    elif model_choice == "Local Outlier Factor":
        model = LocalOutlierFactor()
    else:
        model = OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)

    if model_choice == "Local Outlier Factor":
        y_pred = model.fit_predict(X)
    else:
        model.fit(X)
        y_pred = model.predict(X)

    y_pred = [1 if x == -1 else 0 for x in y_pred]  # Convert to 0/1
    acc = accuracy_score(y, y_pred)
    st.success(f"Model Accuracy: {acc:.2f}")
