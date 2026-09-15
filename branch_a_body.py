# branch_a_body.py
# Branch A: Body text preprocessing and TF-IDF feature extraction
# Implements Baccouche et al. (2020) text cleaning pipeline
# adapted for TF-IDF vectorization (instead of Word2Vec + LSTM).
#
# Keyphrase extraction is implemented without YAKE (not available in this env)
# using a TF-based discriminative vocabulary selector that approximates the
# same effect: prioritize n-grams with high corpus-wide term frequency
# in the spam class relative to ham class.

import re
import string
import math
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_SUBLINEAR_TF, STOP_WORDS
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text cleaning (Baccouche et al. 2020 pipeline — Section 4.2)
# Steps: emoji strip → lowercase → URL removal → number/punct removal
#        → repeated-char compression → stopword removal → simple lemmatization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Simple suffix-stripping lemmatizer (no NLTK required)
# Handles common English inflections well enough for phishing keywords
_LEMMA_RULES = [
    (r"ying$",  "y"),
    (r"ied$",   "y"),
    (r"ies$",   "y"),
    (r"ing$",   ""),
    (r"tion$",  "te"),
    (r"ations$","te"),
    (r"ness$",  ""),
    (r"ment$",  ""),
    (r"ed$",    ""),
    (r"er$",    ""),
    (r"est$",   ""),
    (r"ly$",    ""),
    (r"s$",     ""),
]

_MIN_STEM_LEN = 4   # don't strip suffixes from short words

def _simple_lemmatize(word: str) -> str:
    if len(word) <= _MIN_STEM_LEN:
        return word
    for pattern, replacement in _LEMMA_RULES:
        candidate = re.sub(pattern, replacement, word)
        if len(candidate) >= _MIN_STEM_LEN and candidate != word:
            return candidate
    return word

# Slang/abbreviation map (common in phishing / social content)
_SLANG_MAP = {
    "u": "you", "r": "are", "ur": "your", "4": "for", "2": "to",
    "thx": "thanks", "pls": "please", "plz": "please",
    "gr8": "great", "b4": "before", "btw": "by the way",
    "asap": "as soon as possible", "fyi": "for your information",
    "idk": "i do not know", "omg": "oh my god",
    "lol": "laugh", "brb": "be right back",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of",
    "cant": "cannot", "wont": "will not", "dont": "do not",
    "ive": "i have", "im": "i am", "id": "i would",
    "youre": "you are", "youve": "you have", "youll": "you will",
    "its": "it is", "thats": "that is", "hes": "he is", "shes": "she is",
    "weve": "we have", "theyre": "they are", "theyll": "they will",
}

def clean_text(text: str) -> str:
    """
    Full Baccouche et al. (2020) preprocessing pipeline.
    Returns cleaned, lemmatized string ready for TF-IDF.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: strip emoji and non-ASCII decorative chars
    text = text.encode("ascii", "ignore").decode("ascii")

    # Step 2: lowercase
    text = text.lower()

    # Step 3: remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 4: remove email addresses from body (not headers — already extracted)
    text = re.sub(r"[\w.\-+]+@[\w.\-]+", " ", text)

    # Step 5: remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 6: remove numbers
    text = re.sub(r"\d+", " ", text)

    # Step 7: remove punctuation / non-alpha chars (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 8: compress repeated characters (heeello → hello)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Step 9: tokenize (simple whitespace split — no NLTK needed)
    tokens = text.split()

    # Step 10: apply slang map
    tokens = [_SLANG_MAP.get(t, t) for t in tokens]

    # Step 11: remove stop words and very short tokens
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    # Step 12: simple lemmatization
    tokens = [_simple_lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Discriminative vocabulary builder (approximates YAKE keyphrase filtering)
# Selects n-grams that are most informative by class-conditional frequency
# (log-odds ratio of spam vs ham occurrence), which is the core insight YAKE
# approximates with its statistical scoring.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_discriminative_vocab(
        texts: list,
        labels: list,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple = TFIDF_NGRAM_RANGE,
        top_k_per_class: int = None,
) -> list:
    """
    Build a vocabulary biased toward terms that strongly distinguish
    spam (label=1) from ham (label=0).

    Strategy: fit a preliminary TfidfVectorizer, compute per-term
    log-odds ratio between spam and ham corpora, return the top
    max_features terms by |log-odds|.  Terms that appear roughly
    equally in both classes are penalized (similar to YAKE scoring).
    """
    if top_k_per_class is None:
        top_k_per_class = max_features

    # 1. Fit a broad preliminary vectorizer (large vocab, no limit)
    pre_vec = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features * 3,   # cast wide net first
        sublinear_tf=True,
        min_df=2,
        max_df=0.97,
        stop_words=None,   # already cleaned
    )
    X_pre = pre_vec.fit_transform(texts)
    vocab  = pre_vec.get_feature_names_out()

    labels_arr = [int(l) for l in labels]
    spam_mask  = [i for i, l in enumerate(labels_arr) if l == 1]
    ham_mask   = [i for i, l in enumerate(labels_arr) if l == 0]

    # 2. Sum TF-IDF weights per class
    spam_sums = X_pre[spam_mask].sum(axis=0).A1
    ham_sums  = X_pre[ham_mask].sum(axis=0).A1

    eps = 1e-9
    n_spam = max(len(spam_mask), 1)
    n_ham  = max(len(ham_mask), 1)

    # 3. Log-odds of term appearing in spam vs ham (normalized by class size)
    log_odds = [
        abs(math.log((spam_sums[i] / n_spam + eps) /
                     (ham_sums[i]  / n_ham  + eps)))
        for i in range(len(vocab))
    ]

    # 4. Sort by discriminative power, keep top max_features
    ranked    = sorted(enumerate(log_odds), key=lambda x: -x[1])
    top_idx   = [idx for idx, _ in ranked[:max_features]]
    top_vocab = [vocab[i] for i in top_idx]
    return top_vocab


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TF-IDF vectorizer wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BodyFeaturizer:
    """
    Wraps preprocessing + discriminative vocab selection + TF-IDF fitting.

    Usage:
        bfeat = BodyFeaturizer()
        X_body_train = bfeat.fit_transform(train_texts, train_labels)
        X_body_test  = bfeat.transform(test_texts)
    """

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range:  tuple = TFIDF_NGRAM_RANGE,
        sublinear_tf: bool  = TFIDF_SUBLINEAR_TF,
        use_discriminative_vocab: bool = True,
    ):
        self.max_features  = max_features
        self.ngram_range   = ngram_range
        self.sublinear_tf  = sublinear_tf
        self.use_disc_vocab = use_discriminative_vocab
        self._vectorizer   = None
        self._vocab        = None

    def _preprocess(self, texts: list) -> list:
        return [clean_text(t) for t in texts]

    def fit_transform(self, texts: list, labels: list):
        """Clean, select vocab, fit TF-IDF, return feature matrix."""
        cleaned = self._preprocess(texts)

        if self.use_disc_vocab:
            self._vocab = _build_discriminative_vocab(
                cleaned, labels,
                max_features=self.max_features,
                ngram_range=self.ngram_range,
            )
            self._vectorizer = TfidfVectorizer(
                vocabulary=self._vocab,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
            )
        else:
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                sublinear_tf=self.sublinear_tf,
                min_df=2,
                max_df=0.97,
            )

        return self._vectorizer.fit_transform(cleaned)

    def transform(self, texts: list):
        """Clean and transform new texts with the fitted vectorizer."""
        if self._vectorizer is None:
            raise RuntimeError("Call fit_transform first.")
        cleaned = self._preprocess(texts)
        return self._vectorizer.transform(cleaned)

    def get_feature_names(self) -> list:
        if self._vectorizer is None:
            return []
        return list(self._vectorizer.get_feature_names_out())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick self-test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    sample_texts = [
        "URGENT! Your PayPal account has been suspended. Click here to verify NOW!!",
        "Your BDO account requires immediate verification. Confirm your identity.",
        "Hey, are we still meeting for lunch on Friday? Let me know!",
        "Team meeting notes attached. Please review before Thursday.",
        "Congratulations! You have won $1000. Click to claim your prize immediately.",
        "Hi, I am following up on our project discussion from last week.",
        "Unusual login detected on your account. Update your password now.",
        "The quarterly report is ready for your review. Let us schedule a call.",
    ]
    sample_labels = [1, 1, 0, 0, 1, 0, 1, 0]

    bfeat = BodyFeaturizer(max_features=50, use_discriminative_vocab=True)
    X = bfeat.fit_transform(sample_texts, sample_labels)

    print("Body featurizer self-test")
    print(f"  Input texts:   {len(sample_texts)}")
    print(f"  Output shape:  {X.shape}")
    print(f"  Vocab size:    {len(bfeat.get_feature_names())}")
    print(f"\n  Top 15 features (discriminative):")
    for f in bfeat.get_feature_names()[:15]:
        print(f"    {f}")

    print("\nCleaning examples:")
    examples = [
        "URGENT!! Click HERE to verify ur account NOW!!!",
        "Please find attached the Q3 report for your review.",
        "Confirm ur identity asap or ur account will be suspended!",
    ]
    for e in examples:
        print(f"  Raw:   {e[:60]}")
        print(f"  Clean: {clean_text(e)}\n")
