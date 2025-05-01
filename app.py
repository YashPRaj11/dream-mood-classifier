import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('dream_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app title
st.title("Dream Mood Classifier")

# Instructions for the user
st.write("""
    Enter a description of your dream, and the model will predict your mood.
    The possible moods are: **happy, nightmare, weird, visionary, confused/anxious**.
""")

# Text input for the user's dream
dream_input = st.text_area("Describe your dream:")

# Prediction logic
if st.button('Predict Mood'):
    if dream_input:
        prediction = model.predict([dream_input])
        st.write(f"Predicted Mood: **{prediction[0]}**")
    else:
        st.write("Please enter a description of your dream.")
