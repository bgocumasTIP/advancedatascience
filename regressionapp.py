import streamlit as st
import pandas as pd
#from tensorflow.keras.models import load_model

# Function to load the HDF5 model
def load_regression_model(model_path):
    model = model_path
    return model

# Function to make predictions
def predict_regression(model, features):
    features_df = pd.DataFrame([features], columns=model.input_names)
    prediction = model.predict(features_df)
    return prediction[0][0]

# Streamlit app
def main():
    st.title("Regression Model Predictor")

    # Load pre-trained model
    model_path = "regmodel_weights_best.hdf5"
    model = load_regression_model(model_path)

    # Make predictions
    if st.button("Predict"):
        prediction = predict_regression(model, user_input)
        st.success(f"Prediction: {prediction}")
