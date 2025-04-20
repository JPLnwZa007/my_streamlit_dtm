# -*- coding: utf-8 -*-
"""
Created on Fri Apr 18 23:15:47 2025
@author: Nongnuch
"""

# dtm_app.py - Streamlit app for Iris Flower Classification

import streamlit as st
import numpy as np
import joblib  # Preferred over pickle for sklearn models
import os

# App title and description
st.set_page_config(page_title="Iris Classifier", page_icon="🌼")
st.title("🌼 Iris Flower Classification")
st.write("Enter the features of the iris flower below to predict its species:")

# Try to load the trained model
MODEL_PATH = 'dtm_trained_model.pkl'

if os.path.exists(MODEL_PATH):
    try:
        dtm_model = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.error(f"Model file '{MODEL_PATH}' not found.")
    st.stop()

# Input sliders for flower features
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict button
if st.button("🌸 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = dtm_model.predict(input_data)

    # Map numeric prediction to species name
    species = ['Setosa', 'Versicolor', 'Virginica']
    result = species[int(prediction[0])]
    
    # Display prediction
    st.success(f"The predicted species is: **{result}**")
