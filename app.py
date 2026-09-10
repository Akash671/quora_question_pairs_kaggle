"""
app.py

Streamlit UI for the Quora Duplicate Question Detector.
Calls the FastAPI backend (api.py) over HTTP — run both separately:

    uvicorn api:app --reload          # terminal 1
    streamlit run app.py              # terminal 2
"""

import os

import requests
import streamlit as st

st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="#")


def _get_secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return default
    except Exception:
        # StreamlitSecretNotFoundError when no secrets.toml exists at all
        return default


DEFAULT_API_URL = _get_secret("API_URL", os.environ.get("API_URL", "http://localhost:8000"))

st.title("Quora Duplicate Question Detector")
st.caption("Enter two questions and check whether they're asking the same thing.")

with st.sidebar:
    st.subheader("Backend settings")
    api_url = st.text_input(
        "FastAPI backend URL",
        value=DEFAULT_API_URL,
        help="e.g. http://localhost:8000 locally, or your deployed API URL",
    ).rstrip("/")

col1, col2 = st.columns(2)
with col1:
    question1 = st.text_area("Question 1", height=120, placeholder="How do I learn Python?")
with col2:
    question2 = st.text_area("Question 2", height=120, placeholder="What's the best way to learn Python?")

if st.button("Check for duplicate", type="primary", use_container_width=True):
    if not api_url:
        st.error("No backend URL configured.")
    elif not question1.strip() or not question2.strip():
        st.error("Please fill in both questions.")
    else:
        with st.spinner("Calling model..."):
            try:
                resp = requests.post(
                    f"{api_url}/predict",
                    json={"question1": question1, "question2": question2},
                    timeout=30,
                )
                resp.raise_for_status()
                result = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Couldn't reach the backend at {api_url}. ({e})")
            else:
                prob = result["probability"]
                if result["is_duplicate"]:
                    st.success(f" Likely duplicates — {prob:.1%} confidence")
                else:
                    st.info(f" Likely different questions — {prob:.1%} confidence they're duplicates")
                st.progress(prob)