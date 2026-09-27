import streamlit as st
import joblib

from src.processing import preprocess_text

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📱",
    layout="centered"
)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.title("📱 AI Spam Message Detector")

st.markdown(
    "Enter an SMS message below and our machine learning model "
    "will predict whether it is **SPAM** or **HAM**."
)

message = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

if st.button("Check Message"):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        cleaned_message = preprocess_text(message)

        message_tfidf = vectorizer.transform([cleaned_message])

        prediction = model.predict(message_tfidf)[0]

        if prediction == 1:
            st.error("SPAM SPAM Message")
        else:
            st.success("Its a HAM Message")