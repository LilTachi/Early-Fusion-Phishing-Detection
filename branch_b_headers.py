# branch_b_headers.py
# Branch B: Extended header feature extraction
# Implements all 17 Kulkarni et al. (2020) features + address decomposition
# extensions described in the thesis methodology.
#
# All features return floats in [0, 1] (binary features are 0.0 or 1.0).
# No external NLP libraries required — uses stdlib only.

import re
import difflib
from email import message_from_string, message_from_bytes
from email.utils import parsedate_to_datetime

from config import (
    ALL_HEADER_FEATURES,
    KULKARNI_FEATURES,
    EXTENDED_HEADER_FEATURES,
    KNOWN_LEGIT_DOMAINS,
    SUSPICIOUS_TLDS,
    SUSPICIOUS_USERNAME_KEYWORDS,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Low-level helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_domain(address: str) -> str:
    """Return lowercased domain from 'Name <user@domain.com>' or 'user@domain.com'."""
    if not address:
        return ""
    m = re.search(r"@([\w.\-]+)", address)
    return m.group(1).lower().strip() if m else ""

def _extract_username(address: str) -> str:
    """Return local part (before @) from an email address string."""
    if not address:
        return ""
    m = re.search(r"[\w.+\-]+(?=@)", address)
    return m.group(0).lower() if m else ""

def _extract_tld(domain: str) -> str:
    """Return last label of domain as TLD."""
    if not domain:
        return ""
    return domain.rsplit(".", 1)[-1].lower() if "." in domain else ""

def _all_addresses(header_val: str) -> list:
    """Return list of email address strings from a comma-separated header."""
    return re.findall(r"[\w.+\-]+@[\w.\-]+", header_val or "")

def _typosquat_score(domain: str, known: list = KNOWN_LEGIT_DOMAINS) -> float:
    """
    Typosquatting similarity score in [0, 1].
    1.0 = identical to a known-legit domain (not suspicious).
    Low score = very different from all legit domains (suspicious).
    We invert so that 1.0 = suspicious (high edit-distance to everything legit).
    Uses difflib SequenceMatcher (pure stdlib, O(n*m)).
    """
    if not domain:
        return 0.0
    best = max(
        difflib.SequenceMatcher(None, domain, ld).ratio()
        for ld in known
    )
    # A score close to 1.0 means domain IS legit → not suspicious
    # A mid-range score (e.g. 0.7–0.85) is the typosquatting sweet spot
    # Invert so high output = suspicious (either too different OR exact match)
    # The thesis wants: flag domains that are SIMILAR but NOT identical
    if best >= 0.999:
        return 0.0   # exact match = not suspicious
    elif best >= 0.65:
        return round(1.0 - best, 4)   # close but not identical = suspicious
    else:
        return 0.0   # totally unrelated = unknown domain, don't flag as typosquat


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 17 Kulkarni et al. (2020) features — exact implementations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def feat_To_empty(msg) -> float:
    to_val = msg.get("To", "")
    if not to_val:
        return 1.0
    tl = to_val.lower()
    return 1.0 if ("undisclosed" in tl or "<>" in to_val) else 0.0

def feat_BCC_notempty_To_empty(msg) -> float:
    bcc = msg.get("Bcc", "") or msg.get("BCC", "")
    return 1.0 if (_all_addresses(bcc) and feat_To_empty(msg) == 1.0) else 0.0

def feat_Total_Recp(msg) -> float:
    to  = _all_addresses(msg.get("To", ""))
    cc  = _all_addresses(msg.get("Cc", "") or msg.get("CC", ""))
    bcc = _all_addresses(msg.get("Bcc", "") or msg.get("BCC", ""))
    return 1.0 if (len(to) + len(cc) + len(bcc)) >= 2 else 0.0

def feat_Received_count(msg) -> float:
    return 1.0 if len(msg.get_all("Received") or []) > 3 else 0.0

def feat_Span_time(msg) -> float:
    received = msg.get_all("Received") or []
    if len(received) < 2:
        return 0.0
    times = []
    for r in received:
        m = re.search(r";\s*(.+)$", r, re.MULTILINE)
        if m:
            try:
                times.append(parsedate_to_datetime(m.group(1).strip()))
            except Exception:
                pass
    if len(times) < 2:
        return 0.0
    span = abs((max(times) - min(times)).total_seconds())
    return 1.0 if span > 10 else 0.0

def feat_From_symbol(msg) -> float:
    from_val = msg.get("From", "")
    display  = re.sub(r"<[^>]+>", "", from_val).strip()
    return 1.0 if re.search(r"[?!<>]", display) else 0.0

def feat_Subject_symbol(msg) -> float:
    subj = msg.get("Subject", "") or ""
    try:
        from email.header import decode_header
        parts = decode_header(subj)
        subj = " ".join(
            p[0].decode(p[1] or "utf-8", errors="ignore")
            if isinstance(p[0], bytes) else p[0]
            for p in parts
        )
    except Exception:
        pass
    return 1.0 if re.search(r"[?=]", subj) else 0.0

def feat_Received_SPF(msg) -> float:
    spf = (msg.get("Received-SPF", "") or "").lower()
    return 1.0 if any(v in spf for v in ["fail", "softfail", "bad", "none"]) else 0.0

def feat_Authentication_Results(msg) -> float:
    auth = (msg.get("Authentication-Results", "") or "").lower()
    if not auth:
        return 1.0
    m = re.search(r"dkim\s*=\s*(\w+)", auth)
    if m:
        return 1.0 if m.group(1) in ("fail", "softfail", "bad", "none", "neutral") else 0.0
    return 0.0

def feat_MessageID_domainname(msg) -> float:
    from_d = _extract_domain(msg.get("From", ""))
    mid    = msg.get("Message-ID", "") or ""
    m      = re.search(r"@([\w.\-]+)>", mid)
    mid_d  = m.group(1).lower() if m else ""
    if not from_d or not mid_d:
        return 0.0
    return 1.0 if from_d != mid_d else 0.0

def feat_MessageID_dollar(msg) -> float:
    return 1.0 if "$" in (msg.get("Message-ID", "") or "") else 0.0

def feat_Reply_To_empty_symbol(msg) -> float:
    rt = msg.get("Reply-To", None)
    if rt is None:
        return 1.0
    return 1.0 if "?" in rt else 0.0

def feat_Reply_To_domain(msg) -> float:
    fd = _extract_domain(msg.get("From", ""))
    rd = _extract_domain(msg.get("Reply-To", ""))
    if not fd or not rd:
        return 0.0
    return 1.0 if fd != rd else 0.0

def feat_X_Mailer_exist(msg) -> float:
    xm = msg.get("X-Mailer", None)
    if xm is None:
        return 1.0
    return 1.0 if xm.strip() == "" else 0.0

def feat_Content_Transfer_Encoding(msg) -> float:
    cte = msg.get("Content-Transfer-Encoding", None)
    if cte is None:
        return 1.0
    return 1.0 if cte.strip() == "" else 0.0

def feat_Return_Path_empty(msg) -> float:
    rp = (msg.get("Return-Path", "") or "").lower()
    if not rp or rp.strip() in ("<>", "", "<bounce>"):
        return 1.0
    return 1.0 if "bounce" in rp else 0.0

def feat_Return_Path_From_domain(msg) -> float:
    fd  = _extract_domain(msg.get("From", ""))
    rpd = _extract_domain(msg.get("Return-Path", ""))
    if not fd or not rpd:
        return 0.0
    return 1.0 if fd != rpd else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extended features — email address component decomposition
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _username_features(address: str) -> dict:
    """Return username-level sub-features for a given address string."""
    uname = _extract_username(address)
    if not uname:
        return {
            "length":            0.0,
            "digit_ratio":       0.0,
            "special_ratio":     0.0,
            "suspicious_keyword": 0.0,
        }
    digits   = sum(1 for c in uname if c.isdigit())
    specials  = sum(1 for c in uname if not c.isalnum() and c not in (".", "_", "-"))
    n = max(len(uname), 1)
    has_kw = float(any(kw in uname for kw in SUSPICIOUS_USERNAME_KEYWORDS))
    return {
        "length":             min(len(uname) / 40.0, 1.0),   # normalized
        "digit_ratio":        digits / n,
        "special_ratio":      specials / n,
        "suspicious_keyword": has_kw,
    }

def _domain_features(address: str) -> dict:
    """Return domain-level sub-features for a given address string."""
    domain = _extract_domain(address)
    if not domain:
        return {"typosquat_score": 0.0, "subdomain_depth": 0.0, "tld_suspicious": 0.0}
    tld    = _extract_tld(domain)
    parts  = domain.split(".")
    depth  = max(0, len(parts) - 2)   # subdomain count above SLD+TLD
    return {
        "typosquat_score": _typosquat_score(domain),
        "subdomain_depth": min(depth / 4.0, 1.0),   # normalize to [0,1]
        "tld_suspicious":  float(tld in SUSPICIOUS_TLDS),
    }

def _domain_edit_distance_score(domain_a: str, domain_b: str) -> float:
    """
    Normalized edit-distance between two domain strings, in [0, 1].
    1.0 = completely different, 0.0 = identical.
    Used to flag cross-field mismatches that look manually spoofed.
    """
    if not domain_a or not domain_b:
        return 0.0
    ratio = difflib.SequenceMatcher(None, domain_a, domain_b).ratio()
    return round(1.0 - ratio, 4)

def _extended_features(msg) -> dict:
    from_addr   = msg.get("From", "") or ""
    replyto     = msg.get("Reply-To", "") or ""
    returnpath  = msg.get("Return-Path", "") or ""
    from_domain = _extract_domain(from_addr)
    rt_domain   = _extract_domain(replyto)
    rp_domain   = _extract_domain(returnpath)
    mid         = msg.get("Message-ID", "") or ""
    m           = re.search(r"@([\w.\-]+)>", mid)
    mid_domain  = m.group(1).lower() if m else ""

    fu = _username_features(from_addr)
    fd = _domain_features(from_addr)
    ru = _username_features(replyto)
    rd = _domain_features(replyto)
    pu = _username_features(returnpath)
    pd_ = _domain_features(returnpath)

    return {
        # From username
        "from_username_length":           fu["length"],
        "from_username_digit_ratio":      fu["digit_ratio"],
        "from_username_special_ratio":    fu["special_ratio"],
        "from_username_suspicious_keyword": fu["suspicious_keyword"],
        # From domain
        "from_domain_typosquat_score":    fd["typosquat_score"],
        "from_domain_subdomain_depth":    fd["subdomain_depth"],
        "from_tld_suspicious":            fd["tld_suspicious"],
        # Reply-To
        "replyto_username_length":        ru["length"],
        "replyto_domain_typosquat_score": rd["typosquat_score"],
        "replyto_tld_suspicious":         rd["tld_suspicious"],
        # Return-Path
        "returnpath_username_length":     pu["length"],
        "returnpath_domain_typosquat_score": pd_["typosquat_score"],
        "returnpath_tld_suspicious":      pd_["tld_suspicious"],
        # Cross-field domain mismatch distances
        "from_vs_replyto_edit_distance":     _domain_edit_distance_score(from_domain, rt_domain),
        "from_vs_returnpath_edit_distance":  _domain_edit_distance_score(from_domain, rp_domain),
        "from_vs_msgid_edit_distance":       _domain_edit_distance_score(from_domain, mid_domain),
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Master extractor
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_KULKARNI_FNS = {
    "To_empty":                  feat_To_empty,
    "BCC_notempty_To_empty":     feat_BCC_notempty_To_empty,
    "Total_Recp":                feat_Total_Recp,
    "Received_count":            feat_Received_count,
    "Span_time":                 feat_Span_time,
    "From_symbol":               feat_From_symbol,
    "Subject_symbol":            feat_Subject_symbol,
    "Received_SPF":              feat_Received_SPF,
    "Authentication_Results":    feat_Authentication_Results,
    "MessageID_domainname":      feat_MessageID_domainname,
    "MessageID_dollar":          feat_MessageID_dollar,
    "Reply_To_empty_symbol":     feat_Reply_To_empty_symbol,
    "Reply_To_domain":           feat_Reply_To_domain,
    "X_Mailer_exist":            feat_X_Mailer_exist,
    "Content_Transfer_Encoding": feat_Content_Transfer_Encoding,
    "Return_Path_empty":         feat_Return_Path_empty,
    "Return_Path_From_domain":   feat_Return_Path_From_domain,
}


def extract_header_features(msg) -> dict:
    """
    Extract all header features (Kulkarni 17 + extended 16) from
    a parsed email.message.Message object.
    Returns an ordered dict matching ALL_HEADER_FEATURES.
    """
    kulkarni = {name: fn(msg) for name, fn in _KULKARNI_FNS.items()}
    extended = _extended_features(msg)
    return {**kulkarni, **extended}


def extract_from_file(filepath: str) -> dict:
    """Parse a raw .eml file and return all header features."""
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        msg = message_from_bytes(raw)
    except Exception:
        try:
            with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
                msg = message_from_string(f.read())
        except Exception:
            return {name: 0.0 for name in ALL_HEADER_FEATURES}
    return extract_header_features(msg)


def extract_body_text(filepath: str) -> str:
    """
    Extract the plain-text body from a raw email file.
    Returns empty string on failure.
    """
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
        msg = message_from_bytes(raw)
    except Exception:
        try:
            with open(filepath, "r", encoding="latin-1", errors="ignore") as f:
                msg = message_from_string(f.read())
        except Exception:
            return ""

    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "latin-1"
                    body_parts.append(payload.decode(charset, errors="ignore"))
                except Exception:
                    pass
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "latin-1"
                body_parts.append(payload.decode(charset, errors="ignore"))
        except Exception:
            pass
    return " ".join(body_parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Quick self-test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    SAMPLE = """From: security@paypa1.com
To:
Message-ID: $abc123@evil-server.tk
Subject: Urgent! Verify your account now?
Received: from a by b; Mon, 1 Jan 2024 00:00:01 +0000
Received: from c by d; Mon, 1 Jan 2024 00:00:05 +0000
Received: from e by f; Mon, 1 Jan 2024 00:01:00 +0000
Received: from g by h; Mon, 1 Jan 2024 00:05:00 +0000
Reply-To: support@different-domain.tk
Received-SPF: fail (sender not authorized)
BCC: victim@example.com
Return-Path: bounce@yet-another.xyz

Click here to verify your account immediately or it will be suspended!
"""
    from email import message_from_string
    msg   = message_from_string(SAMPLE)
    feats = extract_header_features(msg)
    print("Self-test: all header features")
    print(f"{'Feature':<40} Value")
    print("-" * 50)
    for k, v in feats.items():
        print(f"  {k:<38} {v}")
    print(f"\nTotal: {len(feats)} features | Non-zero: {sum(1 for v in feats.values() if v > 0)}")
