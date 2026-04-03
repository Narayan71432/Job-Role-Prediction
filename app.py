import streamlit as st
import pandas as pd
import pdfplumber
import docx2txt
import joblib
import base64
from sklearn.feature_extraction.text import TfidfVectorizer

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    
    background_style = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_style, unsafe_allow_html=True)

image_path = "rs.gif"
set_background(image_path)

@st.cache_resource
def load_model():
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = docx2txt.process(uploaded_file)
    else:
        text = str(uploaded_file.read(), "utf-8")
    return text

def predict_job_role(resume_text):
    resume_vector = vectorizer.transform([resume_text])
    predicted_category = model.predict(resume_vector)[0]
    return predicted_category

st.title("Resume Job Role Predictor")

uploaded_file = st.file_uploader("Upload Your Resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded_file:
    resume_text = extract_text_from_file(uploaded_file)
    st.subheader("Extracted Resume Text:")
    st.text_area("", resume_text, height=250)
    
    predicted_role = predict_job_role(resume_text)
    st.success(f"Predicted Job Role: {predicted_role}")
