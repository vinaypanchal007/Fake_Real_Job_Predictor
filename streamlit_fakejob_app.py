import streamlit as st
import joblib
import re

MODEL_PATH = "fakejob_pipeline.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

def decide_label(fake_prob):
    if fake_prob >= 0.7:
        return "Fake Job"
    elif fake_prob <= 0.3:
        return "Real Job"
    else:
        return "Unsure"

def is_gibberish(text):
    letters = re.findall(r"[a-zA-Z]", text)
    return len(letters) / max(len(text), 1) < 0.5

st.set_page_config(page_title="Fake Job Detector", layout="centered")
st.title("Fake Job Posting Detection")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Requirements")
benefits = st.text_area("Benefits")

if st.button("Predict"):

    combined_text = " ".join([
        title,
        description, description,
        requirements, requirements,
        company_profile,
        benefits
    ]).strip()

    word_count = len(combined_text.split())

    if word_count < 20:
        st.error("FAKE JOB POSTING")
        st.caption("Insufficient job details. Legitimate postings provide full descriptions.")
        st.stop()

    if is_gibberish(combined_text):
        st.error("FAKE JOB POSTING")
        st.caption("Input text is not meaningful.")
        st.stop()

    vector = model.named_steps['tfidf'].transform([combined_text])
    if vector.nnz < 5:
        st.error("FAKE JOB POSTING")
        st.caption("Too few recognizable job-related terms.")
        st.stop()

    fake_prob = model.predict_proba([combined_text])[0][1]
    result = decide_label(fake_prob)

    st.markdown("### Prediction Result")
    st.write(f"Fake probability: **{fake_prob:.2f}**")

    if result == "Fake Job":
        st.error("FAKE JOB POSTING")
    elif result == "Real Job":
        st.success("REAL JOB POSTING")
    else:
        st.warning("UNSURE — NEEDS MANUAL REVIEW")

    st.caption("Prediction is based on learned fraud patterns and content quality.")