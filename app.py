import streamlit as st
import pandas as pd
import pickle
import os
import csv
from sklearn.ensemble import RandomForestClassifier
import pyttsx3

# ---------------------------
# LOAD / TRAIN MODEL
# ---------------------------
@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        df = pd.read_csv("Blood_samples_dataset_balanced_2(f).csv")
        X = df.drop("Disease", axis=1)
        y = df["Disease"]

        model = RandomForestClassifier()
        model.fit(X, y)

        pickle.dump(model, open("model.pkl", "wb"))
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# ---------------------------
# LOGIN SYSTEM (CSV)
# ---------------------------
def create_user(username, password):
    with open("users.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([username, password])

def login_user(username, password):
    if not os.path.exists("users.csv"):
        return False
    with open("users.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == username and row[1] == password:
                return True
    return False

# ---------------------------
# MULTILINGUAL RESPONSES
# ---------------------------
responses = {
    "Diabetes": {
        "en": "⚠️ High risk of Diabetes. Reduce sugar intake.",
        "hi": "⚠️ मधुमेह का खतरा। चीनी कम करें।",
        "te": "⚠️ మధుమేహ ప్రమాదం. చక్కెర తగ్గించండి."
    },
    "Anemia": {
        "en": "⚠️ Low hemoglobin. Eat iron-rich foods.",
        "hi": "⚠️ हीमोग्लोबिन कम है। आयरन लें।",
        "te": "⚠️ హిమోగ్లోబిన్ తక్కువ. ఐరన్ ఆహారం తినండి."
    },
    "Healthy": {
        "en": "✅ You are healthy!",
        "hi": "✅ आप स्वस्थ हैं!",
        "te": "✅ మీరు ఆరోగ్యంగా ఉన్నారు!"
    }
}

# ---------------------------
# UI
# ---------------------------
st.title("🧠 AI Health Assistant")

menu = ["Login", "Signup"]
choice = st.sidebar.selectbox("Menu", menu)

# ---------------------------
# SIGNUP
# ---------------------------
if choice == "Signup":
    st.subheader("Create Account")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Signup"):
        create_user(user, pwd)
        st.success("Account created!")

# ---------------------------
# LOGIN
# ---------------------------
if choice == "Login":
    st.subheader("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if login_user(user, pwd):
            st.success("Logged in!")

            # ---------------------------
            # LANGUAGE SELECT
            # ---------------------------
            lang = st.selectbox("Select Language", ["en", "hi", "te"])

            st.header("🧪 Enter Blood Parameters")

            df = pd.read_csv("Blood_samples_dataset_balanced_2(f).csv")
            mean_vals = df.mean(numeric_only=True)

            user_input = {}
            for col in df.columns:
                if col != "Disease":
                    user_input[col] = st.number_input(col, value=float(mean_vals[col]))

            # ---------------------------
            # PREDICTION
            # ---------------------------
            if st.button("Predict Disease"):
                input_df = pd.DataFrame([user_input])
                pred = model.predict(input_df)[0]

                st.subheader(f"Prediction: {pred}")

                # Chatbot-like response
                st.write(responses.get(pred, {}).get(lang, "Stay healthy!"))

                # Voice Output
                if st.button("🔊 Speak Result"):
                    engine = pyttsx3.init()
                    engine.say(responses.get(pred, {}).get(lang, pred))
                    engine.runAndWait()

            # ---------------------------
            # IMAGE ANALYSIS
            # ---------------------------
            st.header("🖼 Upload Image")
            img = st.file_uploader("Upload medical image")

            if img:
                st.image(img)
                st.info("Image analysis feature (demo)")

        else:
            st.error("Invalid credentials")
