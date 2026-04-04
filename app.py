import streamlit as st
import pandas as pd
import pdfplumber
import docx2txt
import joblib
import base64
import os
import io
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="Resume Job Role Predictor", layout="wide")


def set_background(image_path: str):
    if not os.path.exists(image_path):
        return  # Do NOT block startup

    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/gif;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Load background ONCE (not every rerun)
if "bg_loaded" not in st.session_state:
    set_background("rs.gif")
    st.session_state.bg_loaded = True


@st.cache_resource
def load_model():
    if not os.path.exists("best_model.pkl") or not os.path.exists("vectorizer.pkl"):
        st.error("❌ Model files not found. Deployment is broken.")
        st.stop()

    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()


def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_bytes = io.BytesIO(uploaded_file.read())
        with pdfplumber.open(pdf_bytes) as pdf:
            text = "\n".join(
                page.extract_text()
                for page in pdf.pages
                if page.extract_text()
            )

    elif uploaded_file.type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        text = docx2txt.process(uploaded_file)

    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    return text.strip()


def predict_job_role(resume_text: str):
    resume_vector = vectorizer.transform([resume_text])
    return model.predict(resume_vector)[0]


st.title("📄 Resume Job Role Predictor")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX)",
    type=["pdf", "docx"],
)

if uploaded_file:
    with st.spinner("Extracting text..."):
        resume_text = extract_text_from_file(uploaded_file)

    if not resume_text:
        st.warning("No readable text found in the document.")
        st.stop()

    st.subheader("Extracted Resume Text")
    st.text_area("", resume_text, height=300)

    with st.spinner("Predicting job role..."):
        predicted_role = predict_job_role(resume_text)

    st.success(f"🎯 Predicted Job Role: **{predicted_role}**")
