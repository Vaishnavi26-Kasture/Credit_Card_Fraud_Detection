## 💳 Credit Card Fraud Detection App

A simple and interactive Streamlit application to detect credit card fraud using an Isolation Forest model.
Users can enter transaction details, and the app predicts whether the transaction is normal or fraudulent.\

## 🚀 Features

- Load and display a sample of the credit card transaction dataset.

- Use an Isolation Forest model trained on anonymized features.

- User-friendly form for entering transaction details securely (inputs hidden like passwords).

## Predicts transaction status:

- ✅ Normal Transaction

- 🚨 Fraud Detected

## 📁 Project Structure


- ├── app1.py        # Streamlit application file
- ├── creditcard.csv # Dataset file (should be in the same directory)
- └── README.md      # Project documentation

## 🧠 Model Details

- Model Used: Isolation Forest (unsupervised anomaly detection)
- Training: Model trained on all features excluding the Class label.

## Features Highlighted:

- Time (time elapsed)
- V1, V2, V3 (anonymized PCA components)
- Amount (transaction amount)

## ✨ Future Improvements

- Add additional models for comparison (e.g., One-Class SVM).

- Show model performance metrics (Precision, Recall, F1 Score).

- Deploy the app on cloud platforms like Streamlit Cloud or Heroku.

## 🛠️ Tech Stack

- Programming Language: Python
- Framework: Streamlit (for building the web app)

### Libraries:
- Pandas — for data manipulation
- NumPy — for numerical operations
- Scikit-learn — for building the Isolation Forest model

- Machine Learning Model: Isolation Forest (Anomaly Detection)

- Dataset: Credit Card Transactions Dataset (creditcard.csv)



