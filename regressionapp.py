# -*- coding: utf-8 -*-
"""RegressionApp.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10OxWt8RJBYJeHUWjwYC1OYNlwDc_rV8J
"""

# streamlit_regression_app.py
import subprocess

# Install necessary packages
subprocess.run(["pip", "install", "streamlit", "pandas", "numpy", "tensorflow", "scikit-learn"])
pip install --upgrade tensorflow
# Import the required libraries after installation
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_regression_model():
    # Replace 'regmodel_weights.best.hdf5' with your actual model file
    model = tf.keras.models.load_model('regmodel_weights.best.hdf5')
    return model

regression_model = load_regression_model()

st.write("""
# Regression Model Streamlit App
""")

# Create a Streamlit file uploader
file = st.file_uploader("Choose a CSV file", type=["csv"])

# Define a function to make predictions using the regression model
def predict_total_income(data):
    # Your preprocessing logic for regression model goes here
    # Replace the example logic with your actual preprocessing steps
    X = data.drop(['ID', 'Target', 'Total_income'], axis=1)
    X = pd.get_dummies(X)
    X_scaled = scaler.transform(X)
    prediction = regression_model.predict(X_scaled)
    return prediction[0][0]  # Assuming the model output is a single value

# If a file is uploaded, read the CSV file and make predictions
if file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(file)

    # Drop 'ID' and 'Target' for simplicity (you may need to preprocess these features in a real scenario)
    X = df.drop(['ID', 'Target', 'Total_income'], axis=1)
    y = df['Total_income']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Make predictions
    predictions = [predict_total_income(row) for _, row in df.iterrows()]

    # Display the predictions
    st.write("### Predictions:")
    st.table(pd.DataFrame({'Actual Total Income': y, 'Predicted Total Income': predictions}))
