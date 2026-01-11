# helper_nlp.py

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import string
from itertools import islice

import numpy as np
import pandas as pd

import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sentence_transformers import SentenceTransformer


# -------------------------
# spaCy init (keep simple)
# -------------------------

_NLP = {
    "en": spacy.load("en_core_web_sm"),
    "de": spacy.load("de_core_news_sm"),
}


# -------------------------
# Text preprocessing
# -------------------------

def preprocess_text(
    text: str,
    thresh_token_len: int = 2,
    lang: str = "en",
) -> List[str]:
    """
    Lemmatize + remove stopwords/punct + filter short tokens.
    Returns list of lemmas.
    """
    if lang not in _NLP:
        raise ValueError("Only english (en) and german (de) are supported")

    doc = _NLP[lang](str(text))
    tokens = []
    for tok in doc:
        if tok.is_stop or tok.is_punct:
            continue
        lemma = tok.lemma_.strip()
        if len(lemma) <= thresh_token_len:
            continue
        tokens.append(lemma.lower())
    return tokens


def preprocess_text_str(
    text: str,
    thresh_token_len: int = 2,
    lang: str = "en",
) -> str:
    """Convenience wrapper returning space-joined tokens."""
    return " ".join(preprocess_text(text, thresh_token_len=thresh_token_len, lang=lang))


# -------------------------
# Simple meta-features
# -------------------------

def extract_meta_features(text: str) -> List[float]:
    """
    Very lightweight string-level features (language-agnostic).

    Returns:
        [
          length_no_spaces,
          digit_ratio,
          punctuation_ratio,
          uppercase_ratio,
          vowel_to_consonant_ratio,  # only meaningful for Latin scripts
          word_density,              # words / chars_no_spaces
        ]
    """
    s = str(text)
    stripped = s.replace(" ", "")
    if not stripped:
        return [0.0] * 6

    length = len(stripped)

    # Ratios
    digit_ratio = sum(ch.isdigit() for ch in stripped) / length
    punctuation_ratio = sum(ch in string.punctuation for ch in stripped) / length
    uppercase_ratio = sum(ch.isupper() for ch in stripped) / length

    # Rough vowel/consonant (English-ish, but ok as crude signal)
    vowels = set("aeiou")
    num_vowels = sum(ch.lower() in vowels for ch in stripped if ch.isalpha())
    num_consonants = sum(ch.isalpha() and ch.lower() not in vowels for ch in stripped)
    vowel_to_consonant_ratio = (num_vowels / num_consonants) if num_consonants > 0 else 0.0

    word_density = (len(s.split()) / length) if length > 0 else 0.0

    return [
        float(length),
        float(digit_ratio),
        float(punctuation_ratio),
        float(uppercase_ratio),
        float(vowel_to_consonant_ratio),
        float(word_density),
    ]


def add_meta_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    tmp = df.copy()
    cols = [
        "len_no_spaces",
        "digit_ratio",
        "punct_ratio",
        "uppercase_ratio",
        "vowel_to_consonant_ratio",
        "word_density",
    ]
    tmp[cols] = tmp[text_column].apply(extract_meta_features).apply(pd.Series)
    return tmp


# -------------------------
# n-grams helpers
# -------------------------

def generate_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    ngrams = zip(*(islice(tokens, i, None) for i in range(n)))
    return [" ".join(ng) for ng in ngrams]


def generate_uni_di_tri_grams(
    text: str,
    lang: str = "en",
    thresh_token_len: int = 2,
) -> Tuple[Counter, Counter, Counter]:
    tokens = preprocess_text(text, thresh_token_len=thresh_token_len, lang=lang)
    return (
        Counter(tokens),
        Counter(generate_ngrams(tokens, n=2)),
        Counter(generate_ngrams(tokens, n=3)),
    )


# -------------------------
# Sentence embeddings
# -------------------------

class Embeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], llm_context: str = "") -> np.ndarray:
        texts = [llm_context + str(t) for t in texts]
        return self.model.encode(texts, show_progress_bar=False)


# -------------------------
# Models
# -------------------------

@dataclass
class RFTextModelResult:
    test_pred_proba: np.ndarray
    y_test: np.ndarray
    model: RandomForestClassifier
    vectorizer: Optional[Any] = None
    X_test_transformed: Optional[Any] = None


def embedding_rfprob_features(
    df: pd.DataFrame,
    col_name: str,
    llm_context: str = "",
    target_col: str = "scam_flag",
    rf_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    seed: int = 42,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> RFTextModelResult:
    """
    Sentence embeddings -> RandomForest -> P(y=1) on test set.
    """
    rf_params = rf_params or {}
    tmp = df.copy()

    emb = Embeddings(model_name=embedding_model)
    X = emb.encode(tmp[col_name].astype(str).tolist(), llm_context=llm_context)
    y = tmp[target_col].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    clf = RandomForestClassifier(**rf_params, random_state=seed)
    clf.fit(X_train, y_train)

    test_pred = clf.predict_proba(X_test)[:, 1]
    return RFTextModelResult(test_pred_proba=test_pred, y_test=y_test, model=clf)


def nlp_classifier_prob(
    df: pd.DataFrame,
    col_name: str,
    target_col: str = "scam_flag",
    lang: str = "en",
    max_features: int = 1000,
    rf_params: Optional[Dict[str, Any]] = None,
    ngram_range: Tuple[int, int] = (1, 2),
    bow: bool = True,
    test_size: float = 0.2,
    seed: int = 42,
    thresh_token_len: int = 2,
) -> RFTextModelResult:
    """
    Preprocess -> (BoW or TF-IDF) -> RF -> P(y=1) on test set.

    Returns enough objects to later do SHAP:
      - model
      - vectorizer
      - X_test_transformed
      - test_pred_proba
    """
    rf_params = rf_params or {}
    tmp = df.copy()

    tmp["_proc"] = tmp[col_name].astype(str).apply(
        lambda s: preprocess_text_str(s, thresh_token_len=thresh_token_len, lang=lang)
    )

    X = tmp["_proc"]
    y = tmp[target_col].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    vectorizer = (CountVectorizer if bow else TfidfVectorizer)(
        ngram_range=ngram_range,
        max_features=max_features,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = RandomForestClassifier(**rf_params, random_state=seed)
    clf.fit(X_train_vec, y_train)

    test_pred = clf.predict_proba(X_test_vec)[:, 1]

    return RFTextModelResult(
        test_pred_proba=test_pred,
        y_test=y_test,
        model=clf,
        vectorizer=vectorizer,
        X_test_transformed=X_test_vec,
    )
