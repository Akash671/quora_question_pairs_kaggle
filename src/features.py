"""
src/features.py

Turns one raw (question1, question2) pair into the exact feature vector
the model was trained on. Feature order MUST match training's `feature_cols`.
"""

from rapidfuzz import fuzz

try:
    from .preprocessing import quora_preprocess
except ImportError as exc:
    if "attempted relative import" not in str(exc):
        raise
    from preprocessing import quora_preprocess  # fallback if run as a script

# Must match training's feature_cols order exactly
FEATURE_COLUMNS = [
    "q1_len", "q2_len", "q1_n_words", "q2_n_words", "abs_len_diff", "avg_len",
    "common_word_count", "word_share", "jaccard",
    "fuzz_ratio", "fuzz_partial_ratio", "fuzz_token_sort_ratio", "fuzz_token_set_ratio",
    "tfidf_cosine_sim",
    "same_first_word", "same_last_word",
    "q1_freq", "q2_freq", "neighbor_overlap",
]


def build_features(raw_q1: str, raw_q2: str,
                    tfidf=None, question_freq: dict = None,
                    question_neighbors: dict = None) -> list:
    """
    Build one feature row for a (question1, question2) pair.

    tfidf              : a FITTED TfidfVectorizer from training.
                          If None, tfidf_cosine_sim is returned as 0.0.
    question_freq      : dict {raw question text -> degree in training graph}.
                          If None, q1_freq/q2_freq default to 0.
    question_neighbors : dict {raw question text -> set of paired questions}.
                          If None, neighbor_overlap defaults to 0.

    NOTE: tfidf/question_freq/question_neighbors are NOT currently in your
    model/ folder (only model.joblib and scaler.joblib are there). Until
    you export and load them, this function still runs, but those three
    columns will silently be 0 for every prediction — which will skew
    results if the model leaned on them during training. See the note
    at the bottom of this message.
    """
    clean_q1 = quora_preprocess(raw_q1)
    clean_q2 = quora_preprocess(raw_q2)

    w1, w2 = clean_q1.split(), clean_q2.split()
    s1, s2 = set(w1), set(w2)
    common = s1 & s2
    total = s1 | s2

    q1_len, q2_len = len(clean_q1), len(clean_q2)
    q1_n_words, q2_n_words = len(w1), len(w2)
    abs_len_diff = abs(q1_n_words - q2_n_words)
    avg_len = (q1_n_words + q2_n_words) / 2

    common_word_count = len(common)
    word_share = len(common) / (len(s1) + len(s2) + 1e-6)
    jaccard = len(common) / (len(total) + 1e-6)

    fuzz_ratio = fuzz.ratio(clean_q1, clean_q2)
    fuzz_partial_ratio = fuzz.partial_ratio(clean_q1, clean_q2)
    fuzz_token_sort_ratio = fuzz.token_sort_ratio(clean_q1, clean_q2)
    fuzz_token_set_ratio = fuzz.token_set_ratio(clean_q1, clean_q2)

    if tfidf is not None:
        q1_vec = tfidf.transform([clean_q1])
        q2_vec = tfidf.transform([clean_q2])
        denom = (q1_vec.multiply(q1_vec).sum() ** 0.5) * (q2_vec.multiply(q2_vec).sum() ** 0.5)
        tfidf_cosine_sim = float(q1_vec.multiply(q2_vec).sum() / denom) if denom > 0 else 0.0
    else:
        tfidf_cosine_sim = 0.0

    same_first_word = int(bool(w1) and bool(w2) and w1[0] == w2[0])
    same_last_word = int(bool(w1) and bool(w2) and w1[-1] == w2[-1])

    question_freq = question_freq or {}
    question_neighbors = question_neighbors or {}
    q1_freq = question_freq.get(raw_q1, 0)
    q2_freq = question_freq.get(raw_q2, 0)
    n1 = question_neighbors.get(raw_q1, set())
    n2 = question_neighbors.get(raw_q2, set())
    neighbor_overlap = len(n1 & n2)

    return [
        q1_len, q2_len, q1_n_words, q2_n_words, abs_len_diff, avg_len,
        common_word_count, word_share, jaccard,
        fuzz_ratio, fuzz_partial_ratio, fuzz_token_sort_ratio, fuzz_token_set_ratio,
        tfidf_cosine_sim,
        same_first_word, same_last_word,
        q1_freq, q2_freq, neighbor_overlap,
    ]