import streamlit as st
import numpy as np
import pickle
import os

# App configuration
st.set_page_config(page_title="Rock vs Mine Classifier", layout="centered")

# App title with emoji
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎯 Rock vs Mine Prediction App 💣</h1>", unsafe_allow_html=True)
st.write("---")

# Debugging support (optional)
with st.expander("📁 Check Environment (Optional)", expanded=False):
    st.write("**Current Directory:**", os.getcwd())
    st.write("**Files:**", os.listdir())

# Load model safely
model_path = os.path.join("Rock_VS_Mine_Prediction", "forest_model.pkl")
if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure 'forest_model.pkl' is in the correct folder.")
    st.stop()

with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Instructions
st.markdown("### 🔢 Enter the 60 Sonar Features")
st.markdown("Each feature is a float value representing signal strength. You can use sliders or input manually.")

# Feature inputs in columns
inputs = []
cols = st.columns(3)  # 3 columns for better layout
for i in range(60):
    with cols[i % 3]:
        val = st.number_input(f"Feature {i+1}", value=0.0, format="%.4f", key=f"input_{i}")
        inputs.append(val)

# Predict button
if st.button("🚀 Predict"):
    input_array = np.array(inputs).reshape(1, -1)
    prediction = model.predict(input_array)

    st.write("---")
    if prediction[0] == 'R':
        st.success("🎸 **It's a Rock!** Not a threat.")
        st.balloons()
    else:
        st.error("💣 **Warning: It's a Mine!** Potential underwater explosive detected.")

    st.write("🔍 **Model Used:** Random Forest Classifier")
