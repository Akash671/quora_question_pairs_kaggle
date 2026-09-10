"""Streamlit UI for the Quora Duplicate Question Detector."""

import streamlit as st

from src.model import predict

st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="?")

st.title("Quora Duplicate Question Detector")
st.caption("Enter two questions and check whether they're asking the same thing.")

col1, col2 = st.columns(2)
with col1:
    question1 = st.text_area("Question 1", height=120, placeholder="How do I learn Python?")
with col2:
    question2 = st.text_area("Question 2", height=120, placeholder="What's the best way to learn Python?")

if st.button("Check for duplicate", type="primary", use_container_width=True):
    if not question1.strip() or not question2.strip():
        st.error("Please fill in both questions.")
    else:
        with st.spinner("Calling model..."):
            try:
                result = predict(question1, question2)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
            else:
                prob = result["probability"]
                if result["is_duplicate"]:
                    st.success(f" Likely duplicates — {prob:.1%} confidence")
                else:
                    st.info(f" Likely different questions — {prob:.1%} confidence they're duplicates")
                st.progress(prob)