# app.py
import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load(r"Email_Spam_Detection.joblib")

# Page config
st.set_page_config(page_title="Spam Detector", page_icon="📩")

st.title("📩 Email Spam Classification System")
st.write("Enter a message to check if it is Spam or Not Spam.")

# Text input
user_input = st.text_area("Enter your message here:", height=150)

# Predict button
if st.button("Analyze Message"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        # Convert user input to DataFrame (needed for ColumnTransformer pipelines)
        input_df = pd.DataFrame({"Message_body": [user_input]})
        
        # Get prediction probabilities
        probability = model.predict_proba(input_df)[0]
        
        # Set threshold for Spam detection
        threshold = 0.4

        # Use threshold to decide Spam/Not Spam
        if probability[1] >= threshold:
            st.error(f"🚨 Spam Detected! ({probability[1]*100:.2f}% confidence)")
        else:
            st.success(f"✅ Not Spam ({probability[0]*100:.2f}% confidence)")