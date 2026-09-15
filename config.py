# config.py — Early Fusion Phishing Detection Pipeline
# Abundo, Custodio, Pastoral — Mapua University 2026
# Combines: Kulkarni et al. (2020) header features
#           Baccouche et al. (2020) body preprocessing
#           Early fusion via feature vector concatenation

import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR,    exist_ok=True)
os.makedirs(RAW_DIR,     exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

#Dataset paths 
# Each dataset needs spam/ and ham/ subdirectories under raw/<name>/
# SpamAssassin is downloaded by 1_download_data.py
# Nazario and TREC-7 must be placed manually (see README)
DATASETS = {
    "SpamAssassin": {
        "spam_dir": os.path.join(RAW_DIR, "SpamAssassin", "spam"),
        "ham_dir":  os.path.join(RAW_DIR, "SpamAssassin", "ham"),
        "csv":      os.path.join(DATA_DIR, "SpamAssassin_features.csv"),
        "kulkarni_rf_accuracy": 93.56,   # published benchmark (RF + Relief)
    },
    "Nazario": {
        "spam_dir": os.path.join(RAW_DIR, "Nazario", "spam"),
        "ham_dir":  os.path.join(RAW_DIR, "Nazario", "ham"),
        "csv":      os.path.join(DATA_DIR, "Nazario_features.csv"),
        "kulkarni_rf_accuracy": None,
    },
    "TREC7": {
        "spam_dir": os.path.join(RAW_DIR, "TREC7", "spam"),
        "ham_dir":  os.path.join(RAW_DIR, "TREC7", "ham"),
        "csv":      os.path.join(DATA_DIR, "TREC7_features.csv"),
        "kulkarni_rf_accuracy": None,
    },
}

#Kulkarni et al. (2020) 17 header feature names
KULKARNI_FEATURES = [
    "To_empty",
    "BCC_notempty_To_empty",
    "Total_Recp",
    "Received_count",
    "Span_time",
    "From_symbol",
    "Subject_symbol",
    "Received_SPF",
    "Authentication_Results",
    "MessageID_domainname",
    "MessageID_dollar",
    "Reply_To_empty_symbol",
    "Reply_To_domain",
    "X_Mailer_exist",
    "Content_Transfer_Encoding",
    "Return_Path_empty",
    "Return_Path_From_domain",
]

#Extended header features (our additions address decomposition)
EXTENDED_HEADER_FEATURES = [
    # From username analysis
    "from_username_length",
    "from_username_digit_ratio",
    "from_username_special_ratio",
    "from_username_suspicious_keyword",
    # From domain analysis
    "from_domain_typosquat_score",
    "from_domain_subdomain_depth",
    "from_tld_suspicious",
    # Reply-To address decomposition
    "replyto_username_length",
    "replyto_domain_typosquat_score",
    "replyto_tld_suspicious",
    # Return-Path address decomposition
    "returnpath_username_length",
    "returnpath_domain_typosquat_score",
    "returnpath_tld_suspicious",
    # Cross-field domain mismatch (granular)
    "from_vs_replyto_edit_distance",
    "from_vs_returnpath_edit_distance",
    "from_vs_msgid_edit_distance",
]

ALL_HEADER_FEATURES = KULKARNI_FEATURES + EXTENDED_HEADER_FEATURES

#Body / TF-IDF settings =
TFIDF_MAX_FEATURES  = 5000
TFIDF_NGRAM_RANGE   = (1, 3)   # unigrams, bigrams, trigrams
TFIDF_SUBLINEAR_TF  = True     # log-normalise term frequency

#ML settings =
CV_FOLDS    = 10
RANDOM_SEED = 42

# Known-legitimate domains for typosquatting scoring
KNOWN_LEGIT_DOMAINS = [
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
    "apple.com", "google.com", "microsoft.com", "amazon.com",
    "paypal.com", "facebook.com", "twitter.com", "linkedin.com",
    "bdo.com.ph", "bpi.com.ph", "metrobank.com.ph",
    "gcash.com", "paymaya.com",
]

# Suspicious TLDs (high abuse rate)
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",   # free Freenom TLDs
    "xyz", "top", "club", "click", "download", "loan",
    "work", "party", "gdn", "stream", "bid", "win",
    "review", "faith", "date", "trade", "accountant",
}

# Suspicious keywords in usernames/local parts
SUSPICIOUS_USERNAME_KEYWORDS = {
    "security", "secure", "alert", "update", "verify", "confirm",
    "account", "login", "support", "helpdesk", "noreply", "no-reply",
    "admin", "service", "notification", "billing",
}

# Stop words (stdlib replacement for NLTK)
STOP_WORDS = {
    "a","about","above","after","again","against","all","am","an","and",
    "any","are","aren't","as","at","be","because","been","before","being",
    "below","between","both","but","by","can't","cannot","could","couldn't",
    "did","didn't","do","does","doesn't","doing","don't","down","during",
    "each","few","for","from","further","get","got","had","hadn't","has",
    "hasn't","have","haven't","having","he","he'd","he'll","he's","her",
    "here","here's","hers","herself","him","himself","his","how","how's",
    "i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it",
    "it's","its","itself","let's","me","more","most","mustn't","my",
    "myself","no","nor","not","of","off","on","once","only","or","other",
    "ought","our","ours","ourselves","out","over","own","same","shan't",
    "she","she'd","she'll","she's","should","shouldn't","so","some","such",
    "than","that","that's","the","their","theirs","them","themselves","then",
    "there","there's","these","they","they'd","they'll","they're","they've",
    "this","those","through","to","too","under","until","up","very","was",
    "wasn't","we","we'd","we'll","we're","we've","were","weren't","what",
    "what's","when","when's","where","where's","which","while","who","who's",
    "whom","why","why's","will","with","won't","would","wouldn't","you",
    "you'd","you'll","you're","you've","your","yours","yourself","yourselves",
    # extras relevant to email
    "http","www","com","also","like","one","would","may","said","new",
    "use","used","using","just","well","way","even","know","make","see",
    "back","much","many","give","most","tell","also","us","however",
}
