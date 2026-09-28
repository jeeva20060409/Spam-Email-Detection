import streamlit as st
import joblib


# ---------------------------------------
# Load trained model and TF-IDF vectorizer
# ---------------------------------------

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# ---------------------------------------
# Page Configuration
# ---------------------------------------

st.set_page_config(
    page_title="Spam Email Detector",
    layout="centered"
)


# ---------------------------------------
# Title
# ---------------------------------------

st.title("Spam Email Detection System")

st.write(
    "Enter an email message below to check whether it is Spam or Not Spam."
)


# ---------------------------------------
# Email Input
# ---------------------------------------

email = st.text_area(
    "Enter Email Message:",
    height=200,
    placeholder="Type or paste your email here..."
)


# ---------------------------------------
# Prediction Button
# ---------------------------------------

if st.button("🔍 Predict"):

    if email.strip() == "":
        st.warning("⚠️ Please enter an email message.")

    else:

        # Convert email into TF-IDF features
        email_tfidf = vectorizer.transform([email])

        # Make prediction
        prediction = model.predict(email_tfidf)[0]

        # Display result
        if prediction == 1:

            st.error("🚨 SPAM EMAIL")

            st.write(
                "This email has been classified as Spam."
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                "This email has been classified as Not Spam."
            )