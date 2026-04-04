import streamlit as st
import pandas as pd
import pdfplumber
import docx2txt
import joblib
import io
import os
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Resume Job Role Predictor", layout="wide")


@st.cache_resource
def load_model():
    """Load the trained model and vectorizer."""
    if not os.path.exists("best_model.pkl") or not os.path.exists("vectorizer.pkl"):
        st.error("❌ Model files not found. Deployment is broken.")
        st.stop()

    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_model()


def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, or plain text files."""
    if uploaded_file.type == "application/pdf":
        pdf_bytes = io.BytesIO(uploaded_file.read())
        with pdfplumber.open(pdf_bytes) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif uploaded_file.type == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        text = docx2txt.process(temp_path)
        os.remove(temp_path)

    else:
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    return text.strip()


def predict_job_role(resume_text: str):
    """Predict job role from resume text."""
    resume_vector = vectorizer.transform([resume_text])
    return model.predict(resume_vector)[0]


# ---- Streamlit UI ----
st.title("📄 Resume Job Role Predictor")

uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX)", type=["pdf", "docx"]
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
