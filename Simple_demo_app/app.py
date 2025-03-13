import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a modern look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .spam-box {
        background-color: #ffebee;
        border: 1px solid #ef5350;
    }
    .ham-box {
        background-color: #e8f5e9;
        border: 1px solid #66bb6a;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4527a0;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📱 SMS Spam Detector")
st.markdown("### Check if your SMS is spam or not")
st.markdown("---")

# Function to load the model and vectorizer
@st.cache_resource
def load_model():
    try:
        # Load the model
        model = pickle.load(open('random_forest_model.pkl', 'rb'))
        # Load the vectorizer
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
        return model, vectorizer
    except FileNotFoundError:
        # If model files don't exist yet, create dummy objects for demonstration
        st.warning("Model files not found. Using a demo mode.")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model()

# Text input for SMS
sms_text = st.text_area("Enter the SMS message to analyze:", height=150)

# Prediction function
def predict_spam(text):
    # Transform the text using the vectorizer
    text_vectorized = vectorizer.transform([text])
    # Predict the class
    prediction = model.predict(text_vectorized)[0]
    # Get prediction probability
    proba = model.predict_proba(text_vectorized)[0]
    return prediction, proba

# Analyze button
if st.button("Analyze SMS"):
    if not sms_text.strip():
        st.error("Please enter an SMS message.")
    else:
        # Add a spinner for visual feedback
        with st.spinner("Analyzing..."):
            # Simulate processing time
            time.sleep(0.5)
            
            # Make prediction
            prediction, proba = predict_spam(sms_text)
            
            # Display result
            if prediction == 1:
                spam_prob = proba[1] * 100
                st.markdown(f"""
                <div class="prediction-box spam-box">
                    <h2>⚠️ Spam Detected!</h2>
                    <p>This message appears to be spam with {spam_prob:.1f}% confidence.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Display spam characteristics
                st.markdown("### Why this might be spam:")
                st.markdown("- Contains promotional language or offers")
                st.markdown("- Unusual formatting or excessive punctuation")
                st.markdown("- Requests personal information")
                st.markdown("- Contains URLs or asks you to click on links")
            else:
                ham_prob = proba[0] * 100
                st.markdown(f"""
                <div class="prediction-box ham-box">
                    <h2>✅ This is Not Spam</h2>
                    <p>This message appears to be legitimate with {ham_prob:.1f}% confidence.</p>
                </div>
                """, unsafe_allow_html=True)

# Additional information in sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a Random Forest model trained on SMS messages to detect spam.
    
    The model analyzes the text content and predicts whether the message is spam 
    or legitimate based on patterns it learned during training.
    """)
    
    st.header("Examples")
    st.markdown("**Likely Spam:**")
    st.markdown("- 'CONGRATULATIONS! You've won a free iPhone! Click here to claim now!'")
    st.markdown("- 'URGENT: Your account has been suspended. Verify your details at http://bit.ly/2kLm'")
    
    st.markdown("**Likely Not Spam:**")
    st.markdown("- 'Hey, are we still meeting for coffee at 3pm today?'")
    st.markdown("- 'Your appointment is confirmed for tomorrow at 2:30pm. Reply Y to confirm.'")

# Footer
st.markdown("---")
st.markdown("SMS Spam Detector © 2025")
