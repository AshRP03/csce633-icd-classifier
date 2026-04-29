"""
Data pipeline — Stages 1 through 5.

Stage 1 : Build ICD vocabulary from D_ICD_DIAGNOSES + D_ICD_PROCEDURES
Stage 2 : Filter NOTEEVENTS (drop errors, keep high-signal categories, sample)
Stage 3 : Sentence-split each note's TEXT field
Stage 4 : Three-bucket heuristic labeler (class 1 / class 0 / discard)
Stage 5 : Balance classes, combine with gold labels, write pseudo_labeled.csv
"""

import re
import random
import nltk
import pandas as pd
from pathlib import Path

import config

# Download punkt tokenizer model on first run (offline after that)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Medical stopwords ──────────────────────────────────────────────────────────
# Common words in ICD titles that carry no discriminative signal
_MEDICAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "with", "for", "due",
    "to", "not", "nos", "nec", "other", "unspecified", "without", "type",
    "by", "at", "as", "is", "on", "other", "specified", "following",
    "complicating", "complication", "complications", "associated",
    "secondary", "subsequent", "encounter", "initial", "history",
    "personal", "family", "care", "status", "examination",
}

# ── Class 1 regex patterns ─────────────────────────────────────────────────────
_PATIENT_PATTERNS = re.compile(
    r"\b(patient|pt)[\s\w]{0,30}\b"
    r"(has|had|is|was|were|presents?|presented|reports?|reported|"
    r"denies?|denied|developed|diagnosed|underwent|complains?|states?)\b",
    re.IGNORECASE,
)
_CLINICAL_FINDING_PATTERNS = re.compile(
    r"\b(history of|h/o|hx of|consistent with|evidence of|"
    r"findings? of|diagnosis of|impression:|assessment:|plan:)\b",
    re.IGNORECASE,
)

# ── Class 0 regex patterns ─────────────────────────────────────────────────────
_DOSAGE_PATTERN = re.compile(
    r"\b\d+\.?\d*\s*(mg|mcg|ml|mL|units?|tabs?|capsules?|gm|g|mEq|mcg/kg)\b",
    re.IGNORECASE,
)
_HEADER_PATTERN = re.compile(
    r"^[A-Z][A-Z\s\-/]{3,}:\s*$|^\s*[A-Z][A-Z\s]{3,}\s*$"
)
_COMPARATIVE_PATTERN = re.compile(
    r"\b(compared to|no change|unchanged|stable since|similar to prior|"
    r"no significant change|no interval change)\b",
    re.IGNORECASE,
)
_ADMIN_PATTERN = re.compile(
    r"\b(discharge to|transferred to|admitted to|patient expired|"
    r"follow.?up|see you in|will return|appointment scheduled)\b",
    re.IGNORECASE,
)
_PURE_NUMERIC = re.compile(r"^\s*[\d\.\,\s:/\-]+\s*$")


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — ICD vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def build_icd_term_set() -> frozenset:
    """
    Load both ICD tables and extract a set of significant medical terms.
    Terms are lowercased, length >= 4, and not in the medical stopword list.
    """
    dfs = []
    for path in (config.ICD_DIAGNOSES_CSV, config.ICD_PROCEDURES_CSV):
        df = pd.read_csv(path, usecols=["SHORT_TITLE", "LONG_TITLE"], dtype=str)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    all_titles = combined["SHORT_TITLE"].dropna().tolist() + \
                 combined["LONG_TITLE"].dropna().tolist()

    terms = set()
    for title in all_titles:
        for word in re.findall(r"[a-zA-Z]+", title.lower()):
            if len(word) >= 4 and word not in _MEDICAL_STOPWORDS:
                terms.add(word)

    print(f"[Stage 1] ICD vocabulary: {len(terms):,} unique terms")
    return frozenset(terms)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Filter NOTEEVENTS
# ─────────────────────────────────────────────────────────────────────────────

def load_noteevents() -> pd.DataFrame:
    """
    Load NOTEEVENTS, drop error rows, filter to high-signal categories,
    and return a random sample of MAX_NOTES_TO_SAMPLE rows.
    """
    print(f"[Stage 2] Reading NOTEEVENTS (this may take a moment)...")
    df = pd.read_csv(
        config.NOTEEVENTS_CSV,
        usecols=["CATEGORY", "TEXT", "ISERROR"],
        dtype=str,
        low_memory=False,
    )

    # Drop flagged error notes
    df = df[df["ISERROR"].isna() | (df["ISERROR"].str.strip() != "1")]

    # Keep only high-signal note types
    df = df[df["CATEGORY"].isin(config.NOTE_CATEGORIES)]
    df = df.dropna(subset=["TEXT"])

    # Sample
    n = min(config.MAX_NOTES_TO_SAMPLE, len(df))
    df = df.sample(n=n, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"[Stage 2] Kept {len(df):,} notes after filtering and sampling")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """
    Split a clinical note into individual sentences.
    Filters out fragments (< 5 words) and sentences exceeding MAX_LENGTH tokens.
    """
    sentences = nltk.sent_tokenize(text)
    result = []
    for s in sentences:
        s = s.strip()
        word_count = len(s.split())
        if 5 <= word_count <= config.MAX_LENGTH:
            result.append(s)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Heuristic labeler
# ─────────────────────────────────────────────────────────────────────────────

def label_sentence(sentence: str, icd_terms: frozenset) -> int | None:
    """
    Returns:
        1    — confident class 1 (ICD-codable)
        0    — confident class 0 (not codable)
        None — uncertain, discard
    """
    lower = sentence.lower()
    words = set(re.findall(r"[a-zA-Z]+", lower))

    # ── Class 0 checks (run first — fast filters) ──────────────────────────
    if _PURE_NUMERIC.match(sentence):
        return 0
    if _HEADER_PATTERN.match(sentence):
        return 0
    if _DOSAGE_PATTERN.search(sentence) and len(sentence.split()) < 8:
        # Short dosage line with no other content
        return 0
    if _COMPARATIVE_PATTERN.search(sentence):
        return 0
    if _ADMIN_PATTERN.search(sentence):
        return 0

    # ── Class 1 checks ─────────────────────────────────────────────────────
    icd_overlap = len(words & icd_terms)
    if icd_overlap >= config.ICD_OVERLAP_THRESH:
        return 1
    if _PATIENT_PATTERNS.search(sentence):
        return 1
    if _CLINICAL_FINDING_PATTERNS.search(sentence):
        return 1

    return None  # discard


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Build pseudo-labeled dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_pseudo_dataset(notes_df: pd.DataFrame, icd_terms: frozenset) -> pd.DataFrame:
    """
    Sentence-split all notes, apply heuristic labeler, balance classes,
    and return a DataFrame with columns ['text', 'label'].
    """
    class1, class0 = [], []

    for text in notes_df["TEXT"]:
        for sentence in split_sentences(text):
            if len(class1) >= config.MAX_PER_CLASS and \
               len(class0) >= config.MAX_PER_CLASS:
                break
            label = label_sentence(sentence, icd_terms)
            if label == 1 and len(class1) < config.MAX_PER_CLASS:
                class1.append(sentence)
            elif label == 0 and len(class0) < config.MAX_PER_CLASS:
                class0.append(sentence)

    print(f"[Stage 4/5] Pseudo-labels — class 1: {len(class1):,}, class 0: {len(class0):,}")

    rows = [(s, 1) for s in class1] + [(s, 0) for s in class0]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    """
    Execute all 5 stages and write pseudo_labeled.csv.
    Also appends the 20 gold training examples so the final CSV
    contains everything the training loop needs.
    Returns the combined DataFrame.
    """
    random.seed(config.RANDOM_SEED)

    # Stage 1
    icd_terms = build_icd_term_set()

    # Stage 2
    notes_df = load_noteevents()

    # Stages 3–5
    pseudo_df = build_pseudo_dataset(notes_df, icd_terms)

    # Load gold examples and append
    gold_df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    # Normalise column names — instructor CSV may vary
    gold_df.columns = [c.strip().lower() for c in gold_df.columns]
    text_col  = next(c for c in gold_df.columns if "text" in c)
    label_col = next(c for c in gold_df.columns if "label" in c)
    gold_df = gold_df[[text_col, label_col]].rename(
        columns={text_col: "text", label_col: "label"}
    )
    gold_df["label"] = gold_df["label"].astype(int)

    combined = pd.concat([pseudo_df, gold_df], ignore_index=True)
    combined.to_csv(config.PSEUDO_LABEL_CSV, index=False)
    print(f"[Pipeline] Wrote {len(combined):,} rows → {config.PSEUDO_LABEL_CSV}")
    return combined


if __name__ == "__main__":
    run_pipeline()
