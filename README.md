# quora_question_pairs_kaggle

Quora question pairs duplicate detector.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub. Keep the `model/` directory and both `.joblib` files in the repository.
2. Open [share.streamlit.io](https://share.streamlit.io/) and select **Deploy an app**.
3. Choose the repository and branch, then set **Main file path** to `app.py`.
4. Deploy. Streamlit Cloud installs the packages from `requirements.txt` automatically.

The Streamlit app loads the model directly, so no FastAPI server or `API_URL` secret is required. The FastAPI service in `src/api.py` remains available for separate API hosting with Uvicorn.

## Run locally

```text
streamlit run app.py
```
