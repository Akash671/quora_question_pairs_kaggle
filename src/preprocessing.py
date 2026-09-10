"""
src/preprocessing.py

Text cleaning for the Quora duplicate-question pipeline.
Must produce IDENTICAL output at inference time as it did during training,
since the model was trained on features derived from this cleaned text.
"""

import re
import os
import zipfile

import nltk
from nltk.stem import WordNetLemmatizer

# --- NLTK data setup ---
NLTK_DATA_DIR = os.path.expanduser("~/nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


def _ensure_nltk_resource(pkg_id: str, subdir: str) -> None:
    """Download an NLTK resource and make sure it's actually extracted."""
    nltk.download(pkg_id, download_dir=NLTK_DATA_DIR, quiet=True)
    zip_path = os.path.join(NLTK_DATA_DIR, subdir, f"{pkg_id}.zip")
    extracted_path = os.path.join(NLTK_DATA_DIR, subdir, pkg_id)
    if os.path.exists(zip_path) and not os.path.exists(extracted_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(os.path.join(NLTK_DATA_DIR, subdir))


_ensure_nltk_resource("wordnet", "corpora")
_ensure_nltk_resource("omw-1.4", "corpora")

# Fail loudly at import time, not deep inside a .apply() call in production
from nltk.corpus import wordnet  # noqa: E402
assert wordnet.synsets("running"), (
    "WordNet failed to load — check NLTK_DATA_DIR contents"
)

_lemmatizer = WordNetLemmatizer()


def quora_preprocess(text: str) -> str:
    """
    Clean a single question string: lowercase, expand contractions,
    normalise numbers/currency/percent symbols, strip punctuation
    (keeping '?'), and lemmatize.

    Must match training exactly — this is the same function used to
    build `clean_q1` / `clean_q2` before training.
    """
    text = str(text).lower().strip()

    # Expand common contractions
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'l", " will", text)
    text = re.sub(r"\'ve", " have", text)

    # Normalise numbers and currency/percent shortcuts
    text = re.sub(r"\b([0-9]+)k\b", r"\1000", text)
    text = re.sub(r"\b([0-9]+)m\b", r"\1000000", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"₹", " rupee ", text)
    text = re.sub(r"€", " euro ", text)
    text = re.sub(r"%", " percent ", text)

    # Strip punctuation but keep letters, numbers, and the question mark
    text = re.sub(r"[^a-zA-Z0-9?\s]", "", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Lemmatize (single shared lemmatizer instance, not re-created per call)
    text = " ".join(_lemmatizer.lemmatize(word) for word in text.split())

    return text


def clean_dataframe(df, q1_col="question1", q2_col="question2",
                     out_q1_col="clean_q1", out_q2_col="clean_q2"):
    """
    Convenience wrapper for batch cleaning during training/offline use.
    (At inference in api.py you'll call quora_preprocess() directly on
    the two incoming strings instead of using this.)
    """
    df = df.copy()
    df[q1_col] = df[q1_col].fillna("")
    df[q2_col] = df[q2_col].fillna("")
    df[out_q1_col] = df[q1_col].apply(quora_preprocess)
    df[out_q2_col] = df[q2_col].apply(quora_preprocess)
    return df