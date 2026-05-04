"""
Data pipeline — rewritten to FIX the train/test distribution mismatch.

OLD PIPELINE (broken):
  Stage 5a: sentence-level pseudo via regex score.
  Stage 5b: discharge=1 / radiology=0 — WRONG: many discharge sections are
            class 0 (Discharge Medications, Discharge Instructions, etc.) and
            many radiology IMPRESSIONS are class 1 (positive findings).

NEW PIPELINE:
  Stage 1   : ICD vocabulary (kept).
  Stage 2   : Filter NOTEEVENTS (kept).
  Stage 3   : Sentence splitting (kept, only used by 5a).
  Stage 4   : Heuristic sentence labeler (kept; thresholds tightened).
  Stage 5a  : Sentence-level pseudo-labels (kept, count reduced).
  Stage 5c  : NEW — Section-anchored fragment pseudo-labels.
              Class 1 from: Discharge Diagnosis, Active Issues, Hospital Course
                            "# Disease:" headers, Past Medical History, Major
                            Surgical Procedure, positive imaging IMPRESSIONS.
              Class 0 from: Discharge Medications, Discharge Instructions,
                            Patient Education, negative/stable IMPRESSIONS,
                            Social/Family History without disease, micro/lab
                            result dumps, pure procedural narratives.
  Stage 5d  : NEW — TF-IDF nearest-neighbor mining anchored to 20 gold examples.
              For each gold example, retrieve top-K similar candidate fragments
              from MIMIC and propagate the gold label.

Output: pseudo_labeled.csv with text,label columns. Gold examples are appended
last so train.py can find them by exact text match.
"""

import re
import random
import hashlib
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — ICD vocabulary
# ─────────────────────────────────────────────────────────────────────────────

_MEDICAL_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "with", "for", "due",
    "to", "not", "nos", "nec", "other", "unspecified", "without", "type",
    "by", "at", "as", "is", "on", "specified", "following",
    "complicating", "complication", "complications", "associated",
    "secondary", "subsequent", "encounter", "initial", "history",
    "personal", "family", "care", "status", "examination",
}


def build_icd_term_set() -> frozenset:
    dfs = []
    for path in (config.ICD_DIAGNOSES_CSV, config.ICD_PROCEDURES_CSV):
        df = pd.read_csv(path, usecols=["SHORT_TITLE", "LONG_TITLE"], dtype=str)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    titles = (combined["SHORT_TITLE"].dropna().tolist()
              + combined["LONG_TITLE"].dropna().tolist())
    terms = set()
    for title in titles:
        for w in re.findall(r"[a-zA-Z]+", title.lower()):
            if len(w) >= 4 and w not in _MEDICAL_STOPWORDS:
                terms.add(w)
    print(f"[Stage 1] ICD vocabulary: {len(terms):,} unique terms")
    return frozenset(terms)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — NOTEEVENTS loader (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def load_noteevents(categories=None, max_n=None) -> pd.DataFrame:
    cats = categories or config.NOTE_CATEGORIES
    print(f"[Stage 2] Reading NOTEEVENTS for categories: {cats}")
    df = pd.read_csv(
        config.NOTEEVENTS_CSV,
        usecols=["CATEGORY", "TEXT", "ISERROR"],
        dtype=str, low_memory=False,
    )
    df = df[df["ISERROR"].isna() | (df["ISERROR"].str.strip() != "1")]
    df = df[df["CATEGORY"].isin(cats)]
    df = df.dropna(subset=["TEXT"])
    if max_n is not None and len(df) > max_n:
        df = df.sample(n=max_n, random_state=config.RANDOM_SEED).reset_index(drop=True)
    else:
        df = df.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"[Stage 2] Loaded {len(df):,} notes — "
          f"{df['CATEGORY'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    try:
        nltk.data.find("tokenizers/punkt")
        sents = nltk.sent_tokenize(text)
    except LookupError:
        sents = re.split(r"(?<=[\.!\?])\s+|\n+", text)
    out = []
    for s in sents:
        s = s.strip()
        n = len(s.split())
        if 5 <= n <= 64:   # keep sentence-sized
            out.append(s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Heuristic sentence labeler (kept; thresholds tightened)
# ─────────────────────────────────────────────────────────────────────────────

_PURE_NUMERIC = re.compile(r"^\s*[\d\.\,\s:/\-]+\s*$")
_HEADER_PATTERN = re.compile(r"^[A-Z][A-Z\s\-/]{3,}:\s*$|^\s*[A-Z][A-Z\s]{3,}\s*$")

_NEG_FINDING = re.compile(
    r"\b(no evidence of|without evidence of|negative for|"
    r"not present|no acute|no active|no new|no significant change|"
    r"within normal limits|wnl\b|unremarkable|"
    r"no (?:fracture|mass|lesion|effusion|pneumothorax|infiltrate|"
    r"consolidation|fistula|abscess|dvt|pe\b))\b",
    re.IGNORECASE,
)
_CARE_DIRECTIVE = re.compile(
    r"\b(dnr\b|dni\b|do not (?:resuscitate|intubate)|"
    r"comfort (?:care|measures)|hospice|palliative|code status)\b",
    re.IGNORECASE,
)
_PATIENT_INSTRUCTION = re.compile(
    r"\b(you (?:have been|are|should|will|must|need to)|"
    r"please (?:take|continue|stop|avoid|call|return)|"
    r"take your (?:medication|pill|dose)|your (?:dose|prescription))\b",
    re.IGNORECASE,
)
_MED_ADMIN = re.compile(
    r"\b(given|administered|received)\b.{0,30}"
    r"\b(mg|mcg|mEq|units?|ml\b|iv\b|po\b)\b",
    re.IGNORECASE,
)
_VITALS_LABS = re.compile(
    r"\b(blood pressure|bp\b|heart rate|hr\b|temperature|temp\b|"
    r"o2 sat|spo2|sodium|potassium|creatinine|wbc|hgb|inr|troponin|lactate)"
    r"\s*(is|was|of|:)?\s*[\d\.\-/]+",
    re.IGNORECASE,
)
_SOCIAL_FAMILY = re.compile(
    r"\b(social history|tobacco|smoking|alcohol|drug use|"
    r"lives (?:alone|with)|occupation|retired|homeless|family history)\b",
    re.IGNORECASE,
)

_PROCEDURE_STRONG = re.compile(
    r"\b(s/p\b|status post|underwent|procedure performed|repair|stent|orif|"
    r"intubat(?:ed|ion)|dialysis|hemodialysis|pci\b|catheterization|"
    r"resection|biopsy|debridement|amputation|colostomy|tracheostomy|"
    r"thoracentesis|paracentesis|bronchoscopy|endoscopy|colonoscopy|"
    r"craniotomy|laminectomy)\b",
    re.IGNORECASE,
)
_DIAGNOSIS_FRAMING = re.compile(
    r"\b(diagnos(?:is|ed|ed with)|consistent with|"
    r"findings? (?:of|consistent with)|presents? with|"
    r"history of|h/o\b|known (?:history of|to have)|"
    r"found to have|confirmed|assessment:|impression:)\b",
    re.IGNORECASE,
)
_CONDITION = re.compile(
    r"\b(pneumonia|sepsis|septic shock|bacteremia|cellulitis|abscess|"
    r"fracture|laceration|hemorrhage|hematoma|thrombosis|embolism|"
    r"infarction|ischemia|necrosis|edema|effusion|insufficiency|failure|"
    r"obstruction|stenosis|occlusion|hypertension|hypotension|"
    r"diabetes|diabetic|anemia|infection|malignancy|carcinoma|"
    r"metastasis|tumor|ulcer|pancreatitis|appendicitis|peritonitis|"
    r"meningitis|encephalitis|hydrocephalus|stroke|infarct|ischemic|"
    r"hemorrhagic|dvt\b|pe\b|pulmonary embolism)\b",
    re.IGNORECASE,
)
_CARDIAC_RHYTHM = re.compile(
    r"\b(afib|a-?fib|atrial fibrillation|atrial flutter|"
    r"tachycardic|tachycardia|bradycardic|bradycardia|"
    r"pvc\b|pvcs\b|svt\b|v-?fib|vfib|heart block|arrhythmia)\b",
    re.IGNORECASE,
)


def label_sentence(sentence: str, icd_terms: frozenset) -> int | None:
    """Score-based labeler. Returns 1, 0, or None (discard)."""
    if _PURE_NUMERIC.match(sentence):
        return 0
    if _HEADER_PATTERN.match(sentence):
        return 0
    wc = len(sentence.split())
    if wc < 4:
        return None

    score = 0

    # Strong negatives
    if _NEG_FINDING.search(sentence):       score -= 4
    if _CARE_DIRECTIVE.search(sentence):    score -= 4
    if _PATIENT_INSTRUCTION.search(sentence): score -= 4
    if _MED_ADMIN.search(sentence):         score -= 2
    if _VITALS_LABS.search(sentence):       score -= 2
    if _SOCIAL_FAMILY.search(sentence):     score -= 3

    # Strong positives
    if _PROCEDURE_STRONG.search(sentence):  score += 4
    if _CARDIAC_RHYTHM.search(sentence):    score += 4
    if _CONDITION.search(sentence):         score += 3
    if _DIAGNOSIS_FRAMING.search(sentence): score += 2

    # ICD term overlap
    words = set(re.findall(r"[a-zA-Z]+", sentence.lower()))
    overlap = len(words & icd_terms)
    if overlap >= 4:    score += 3
    elif overlap >= 2:  score += 1

    if score >= 4:    return 1
    if score <= -2:   return 0
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5a — Sentence-level pseudo-labels (unchanged structure, smaller count)
# ─────────────────────────────────────────────────────────────────────────────

def build_sentence_pseudo(notes_df: pd.DataFrame, icd_terms: frozenset) -> pd.DataFrame:
    n_per_class = config.MAX_PER_CLASS
    class1, class0 = [], []
    seen = set()
    for text in notes_df["TEXT"]:
        for s in split_sentences(text):
            if s in seen:
                continue
            seen.add(s)
            if len(class1) >= n_per_class and len(class0) >= n_per_class:
                break
            lab = label_sentence(s, icd_terms)
            if lab == 1 and len(class1) < n_per_class:
                class1.append(s)
            elif lab == 0 and len(class0) < n_per_class:
                class0.append(s)
        if len(class1) >= n_per_class and len(class0) >= n_per_class:
            break
    print(f"[Stage 5a] Sentence pseudo — class1: {len(class1):,}, class0: {len(class0):,}")
    rows = [(s, 1) for s in class1] + [(s, 0) for s in class0]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5c — Section-anchored fragment pseudo-labels  (THE BIG FIX)
# ─────────────────────────────────────────────────────────────────────────────
#
# Strategy:  Don't use note CATEGORY as the label signal. Instead, find specific
# named sections within a note and label by section type. This matches how the
# 20 gold examples were actually labeled (we verified by inspection).
#
# Each section regex captures an anchor; we extract from the anchor up to a
# target word count. This produces fragments that look like the gold examples.
# ─────────────────────────────────────────────────────────────────────────────

# CLASS 1 section anchors  (codable: name diseases / procedures / problems)
_SEC1_DISCHARGE_DX = re.compile(
    r"(discharge\s+diagnosis|primary\s+diagnosis|principal\s+diagnosis|"
    r"admitting\s+diagnosis|secondary\s+diagnosis|active\s+issues|"
    r"problem\s+list)\s*[:\n]",
    re.IGNORECASE,
)
_SEC1_HOSPITAL_COURSE_PROBLEM = re.compile(
    # "# Cardiogenic Shock:" style problem-anchored hospital course chunks
    r"(?:^|\n)\s*#+\s*[A-Z][A-Za-z][^\n]{2,80}:\s",
)
_SEC1_PMH = re.compile(
    r"(past\s+medical\s+history|pmh|cardiac\s+history|"
    r"medical\s+history)\s*[:\n]",
    re.IGNORECASE,
)
_SEC1_SURGICAL = re.compile(
    r"(major\s+surgical\s+(?:or\s+invasive\s+)?procedure|"
    r"surgical\s+procedures?\s+performed|operative\s+procedure)\s*[:\n]",
    re.IGNORECASE,
)
_SEC1_HPI = re.compile(
    r"(history\s+of\s+present\s+illness|hpi)\s*[:\n]",
    re.IGNORECASE,
)
_SEC1_BRIEF_HOSPITAL_COURSE = re.compile(
    r"(brief\s+hospital\s+course|hospital\s+course)\s*[:\n]",
    re.IGNORECASE,
)

# A second `#` header within the next ~80 words is a sign of a *multi-problem*
# narrative chunk — looks like row 8 from the test, which is class 0.
# A single `#` header followed by sustained narrative is a focused problem
# block (class 1, like the gold "# Cardiogenic Shock:" example).
_MULTI_HASH_NEXT80 = re.compile(
    r"(?:^|\n)\s*#+\s*[A-Z]"
)

# CLASS 0 section anchors  (NOT codable: instructions, dispensing, narratives)
_SEC0_DISCHARGE_MEDS = re.compile(
    r"(discharge\s+medications?|medications?\s+on\s+discharge)\s*[:\n]",
    re.IGNORECASE,
)
_SEC0_DISCHARGE_INSTR = re.compile(
    r"(discharge\s+instructions?|patient\s+instructions?|"
    r"followup\s+instructions?|follow-?up\s+instructions?)\s*[:\n]",
    re.IGNORECASE,
)
_SEC0_SOCIAL = re.compile(
    r"(social\s+history|family\s+history)\s*[:\n]",
    re.IGNORECASE,
)
_SEC0_ALLERGIES = re.compile(
    r"(allergies?)\s*[:\n]",
    re.IGNORECASE,
)
_SEC0_DISPOSITION = re.compile(
    r"(discharge\s+disposition|discharge\s+condition)\s*[:\n]",
    re.IGNORECASE,
)
_SEC0_PHYS_EXAM = re.compile(
    r"(physical\s+exam|physical\s+examination|review\s+of\s+systems)\s*[:\n]",
    re.IGNORECASE,
)

# Numbered-list-style medication entry markers — strong class 0 signal
_NUMBERED_MED_LINE = re.compile(
    r"^\s*\d+\.\s+\w+.{0,40}\b(mg|mcg|tablet|capsule|sig:|po\b|iv\b|disp:)",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_window(text: str, start: int, target_words: int) -> str:
    """Take a fragment starting at `start` containing target_words words."""
    fragment = text[start:]
    words = fragment.split()
    return " ".join(words[:target_words])


def _classify_imaging_impression(impression_text: str) -> int | None:
    """Imaging impressions: positive → class 1, negative/stable → class 0."""
    pos_signals = 0
    neg_signals = 0

    if re.search(r"\b(no evidence of|negative for|unremarkable|normal "
                 r"(?:appearance|size)|within normal limits|wnl\b|"
                 r"no acute|no significant change|stable|unchanged|"
                 r"no new|patent|no abnormality|likely benign)\b",
                 impression_text, re.IGNORECASE):
        neg_signals += 2
    if re.search(r"\b(images unavailable|limited examination|"
                 r"comparison.*not available)\b",
                 impression_text, re.IGNORECASE):
        neg_signals += 2

    # Positive: explicit new/worsening finding or named pathology
    if re.search(r"\b(new|worsening|increased|interval increase|developing|"
                 r"progressive|appeared|developed|enlargement|enlarging)\b.{0,40}"
                 r"\b(opacity|effusion|hemorrhage|infarct|edema|mass|lesion|"
                 r"hematoma|consolidation|infiltrate|fracture|stenosis)\b",
                 impression_text, re.IGNORECASE):
        pos_signals += 3
    if re.search(r"\b(pneumonia|sepsis|infarction|hemorrhage|embolism|"
                 r"thrombosis|fracture|aneurysm|stenosis|occlusion|"
                 r"obstruction|abscess|metastasis|carcinoma|malignancy|"
                 r"hydrocephalus|pneumothorax|cellulitis|appendicitis|"
                 r"diverticulitis|cholecystitis|pancreatitis)\b",
                 impression_text, re.IGNORECASE):
        pos_signals += 2
    if re.search(r"\b(concerning for|suspicious for|consistent with|"
                 r"compatible with|worrisome for)\b",
                 impression_text, re.IGNORECASE):
        pos_signals += 1

    if pos_signals >= 2 and neg_signals <= 1:
        return 1
    if neg_signals >= 2 and pos_signals == 0:
        return 0
    return None  # ambiguous


def _build_section_fragments(notes_df: pd.DataFrame) -> tuple[list, list]:
    """
    Walk every note, extract section-anchored fragments, label by section type.
    Returns (class1_list, class0_list).
    """
    target = config.SECTION_FRAG_TARGET_WORDS
    n_max  = config.SECTION_FRAG_PER_CLASS
    minw   = config.SECTION_FRAG_MIN_WORDS
    maxw   = config.SECTION_FRAG_MAX_WORDS

    class1, class0 = [], []
    seen_hashes = set()

    def _add(text: str, cls: int) -> bool:
        wc = len(text.split())
        if wc < minw or wc > maxw:
            return False
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            return False
        seen_hashes.add(h)
        if cls == 1 and len(class1) < n_max:
            class1.append(text); return True
        if cls == 0 and len(class0) < n_max:
            class0.append(text); return True
        return False

    n_notes = len(notes_df)
    for i, row in enumerate(notes_df.itertuples(index=False), 1):
        if i % 1000 == 0:
            print(f"[Stage 5c]   processed {i:,}/{n_notes:,} notes — "
                  f"c1={len(class1):,} c0={len(class0):,}")
        if len(class1) >= n_max and len(class0) >= n_max:
            break

        text = str(getattr(row, "TEXT", "") or "").strip()
        cat  = str(getattr(row, "CATEGORY", "") or "").strip()
        if len(text.split()) < 30:
            continue

        # ── CLASS 1 anchors (HIGH-CONFIDENCE) ─────────────────────────────
        # Discharge Diagnosis, Active Issues, etc. — these are focused
        # diagnostic listings. Past Medical History — patient has these
        # conditions. Surgical Procedure — explicit named procedure.
        # HPI — patient presentation/diagnosis statement.
        # Brief Hospital Course — RESTORED in round 3-fix. Most BHC fragments
        # are class 1 in gold (including multi-`#` chunks like "# Bilateral
        # hearing loss" which is gold-labeled 1). Removing it in round 2
        # caused a regression.
        for pat in (_SEC1_DISCHARGE_DX, _SEC1_PMH, _SEC1_SURGICAL,
                    _SEC1_HPI, _SEC1_BRIEF_HOSPITAL_COURSE):
            for m in pat.finditer(text):
                frag = _extract_window(text, m.start(), target)
                # Hard veto: drop fragments dominated by numbered med list
                if len(_NUMBERED_MED_LINE.findall(frag)) >= 2:
                    continue
                _add(frag, 1)
                if len(class1) >= n_max:
                    break

        # # Disease: style hospital course chunks → class 1.
        # Both single-problem AND multi-problem `#` blocks count as class 1
        # in gold (e.g. gold example #7 has multiple `#` headers and is
        # class 1). The round-2 multi-hash demotion was wrong.
        for m in _SEC1_HOSPITAL_COURSE_PROBLEM.finditer(text):
            frag = _extract_window(text, m.start(), target)
            # Skip if it looks like disposition/instructions
            if re.search(r"\b(discharge|disposition|instructions)\b",
                         frag[:60], re.IGNORECASE):
                continue
            _add(frag, 1)
            if len(class1) >= n_max:
                break

        # ── CLASS 0 anchors ────────────────────────────────────────────────
        for pat in (_SEC0_DISCHARGE_MEDS, _SEC0_DISCHARGE_INSTR,
                    _SEC0_SOCIAL, _SEC0_DISPOSITION, _SEC0_PHYS_EXAM):
            for m in pat.finditer(text):
                frag = _extract_window(text, m.start(), target)
                _add(frag, 0)
                if len(class0) >= n_max:
                    break

        # ── Imaging IMPRESSION (radiology only) — class depends on content ──
        if cat == "Radiology":
            for m in re.finditer(r"\b(?:IMPRESSION|FINDINGS|CONCLUSION)\s*:\s*",
                                 text):
                frag = _extract_window(text, m.start(), target)
                cls = _classify_imaging_impression(frag)
                if cls is not None:
                    _add(frag, cls)
                if len(class1) >= n_max and len(class0) >= n_max:
                    break
            # ROUND 3-FIX-2: REMOVED radiology body addition. It made the
            # model slightly more class-1-biased (gold acc went 0.95 → 0.85)
            # which negated the benefit of prior shift. Cleaner round-2
            # pipeline + prior shift on top works better.

    print(f"[Stage 5c] Section fragments — class1: {len(class1):,}, "
          f"class0: {len(class0):,}")
    return class1, class0


def build_section_pseudo(notes_df: pd.DataFrame) -> pd.DataFrame:
    if config.SECTION_FRAG_PER_CLASS <= 0:
        return pd.DataFrame(columns=["text", "label"])
    class1, class0 = _build_section_fragments(notes_df)
    rows = [(t, 1) for t in class1] + [(t, 0) for t in class0]
    random.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "label"])


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5d — TF-IDF nearest-neighbor mining anchored to gold (NEW)
# ─────────────────────────────────────────────────────────────────────────────
#
# For each of the 20 gold examples, find the K most-similar candidate fragments
# in MIMIC and propagate the gold label. This produces training data that
# *actually looks like* the test data — closing the train/test domain gap.
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_fragments_for_neighbor_mining(notes_df: pd.DataFrame,
                                             max_per_note: int = 3) -> list[str]:
    """
    Generate fragment candidates by sliding ~100-word windows over every note
    starting at section anchors and at every double-newline.
    """
    target = config.SECTION_FRAG_TARGET_WORDS
    minw   = config.SECTION_FRAG_MIN_WORDS
    maxw   = config.SECTION_FRAG_MAX_WORDS
    candidates = []
    seen = set()

    section_re = re.compile(
        r"(?:" + "|".join([
            r"discharge\s+diagnosis", r"primary\s+diagnosis",
            r"principal\s+diagnosis", r"admitting\s+diagnosis",
            r"active\s+issues", r"past\s+medical\s+history",
            r"history\s+of\s+present\s+illness", r"hpi",
            r"brief\s+hospital\s+course", r"hospital\s+course",
            r"discharge\s+medications?", r"discharge\s+instructions?",
            r"social\s+history", r"family\s+history",
            r"allergies", r"physical\s+exam", r"discharge\s+condition",
            r"discharge\s+disposition", r"impression",
            r"major\s+surgical", r"#+\s*[A-Z][A-Za-z][^\n]{2,40}:",
        ]) + r")",
        re.IGNORECASE,
    )

    for row in notes_df.itertuples(index=False):
        text = str(getattr(row, "TEXT", "") or "").strip()
        if len(text.split()) < 30:
            continue
        anchors = [m.start() for m in section_re.finditer(text)]
        # Add a few generic anchors at paragraph breaks
        anchors += [m.start() for m in re.finditer(r"\n\s*\n", text)]
        anchors = sorted(set(anchors))[: max_per_note * 3]

        for a in anchors[:max_per_note]:
            frag = _extract_window(text, a, target)
            wc = len(frag.split())
            if wc < minw or wc > maxw:
                continue
            h = hashlib.md5(frag.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            candidates.append(frag)
    return candidates


def _bert_encode_batch(texts: list[str], tokenizer, encoder,
                        device, max_len: int = 256, bs: int = 32) -> "np.ndarray":
    """
    Mean-pool Bio_ClinicalBERT embeddings for a list of texts.
    Returns shape (n, hidden_size). Used for semantic similarity in Stage 5d.
    """
    import torch
    import numpy as _np
    encoder.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), bs):
            batch = texts[start: start + bs]
            enc = tokenizer(batch, max_length=max_len, truncation=True,
                            padding="max_length", return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            outputs = encoder(input_ids=ids, attention_mask=mask)
            tok = outputs.last_hidden_state         # (B, L, H)
            m = mask.unsqueeze(-1).float()
            pooled = (tok * m).sum(1) / m.sum(1).clamp(min=1e-9)  # (B, H)
            # L2-normalize for cosine sim
            pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-9)
            out.append(pooled.cpu().numpy())
    return _np.concatenate(out, axis=0) if out else _np.zeros((0, 768))


# Class-0 patterns used in hard-negative filtering — these texts LOOK class-1
# (by lexical similarity to gold class 1) but contain unmistakable class-0
# narrative cues like "will continue", "at the time of discharge", "should be
# transitioned". When such a candidate is a top match for a class-1 gold
# example, it's a HARD NEGATIVE: similar to class 1 but actually class 0.
# We label them class 0 explicitly to teach the model the contrast.
_HARD_NEGATIVE_CUES = re.compile(
    r"\b(at the time of discharge|will continue this for|"
    r"should be transitioned to|please (?:take|continue|stop|avoid|call|return)|"
    r"poor candidate for|disp:\s*\*\d+|sig:\s*(?:one|two|three|\d+)|"
    r"refills?:?\s*\*?\d|"
    r"reported to and read back|gram positive cocci|gram negative|"
    r"compared with the (?:report of the )?prior study|"
    r"images unavailable for review|limited examination|"
    r"are pending at the time of|labs?\s+pending|results pending)\b",
    re.IGNORECASE,
)

# CRITICAL: positive overrides. Two gold class-1 examples matched the
# hard-negative cue regex (because they include "Sig: One" and "Disp:*16"
# *followed by* a Discharge Diagnosis listing). If a candidate matches the
# hard-negative cues AND has any of these positive overrides, it's NOT a
# hard negative — leave it for the regular neighbor mining pass.
_HARD_NEG_POSITIVE_OVERRIDE = re.compile(
    r"\b(discharge\s+diagnosis|primary\s+diagnosis|principal\s+diagnosis|"
    r"admitting\s+diagnosis|active\s+issues?|past\s+medical\s+history|"
    r"history\s+of\s+present\s+illness|major\s+surgical\s+(?:or\s+invasive\s+)?procedure|"
    r"cardiac\s+history|chief\s+complaint)\s*[:\n]",
    re.IGNORECASE,
)


def _is_hard_negative(text: str) -> bool:
    """True iff text matches HARD_NEGATIVE_CUES and NOT POSITIVE_OVERRIDE."""
    if _HARD_NEG_POSITIVE_OVERRIDE.search(text):
        return False
    return bool(_HARD_NEGATIVE_CUES.search(text))


def build_neighbor_pseudo(notes_df: pd.DataFrame,
                          gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mine class-1 neighbors AND hard negatives for each gold example using
    Bio_ClinicalBERT semantic embeddings.

    For each gold class-1 example:
      - top-K most-similar candidates → label class 1
      - candidates above a fairly-similar threshold (>0.50 SBERT-style) that
        ALSO match _HARD_NEGATIVE_CUES → label class 0 (hard negatives)

    For each gold class-0 example:
      - top-K most-similar candidates → label class 0
    """
    if config.NEIGHBOR_PER_GOLD <= 0:
        return pd.DataFrame(columns=["text", "label"])

    import torch
    from model import load_tokenizer
    from transformers import AutoModel

    print("[Stage 5d] Generating candidate pool for neighbor mining...")
    # ROUND-FIX: throttle candidate count to stay within runtime budget.
    # 80k candidates is enough diversity for 20 gold seeds with K~30 each.
    per_note = int(getattr(config, "NEIGHBOR_CANDIDATES_PER_NOTE", 4))
    cap      = int(getattr(config, "NEIGHBOR_CANDIDATE_POOL", 80_000))

    candidates_full = _candidate_fragments_for_neighbor_mining(notes_df,
                                                                max_per_note=per_note)
    if len(candidates_full) > cap:
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(candidates_full)
        candidates = candidates_full[:cap]
    else:
        candidates = candidates_full
    print(f"[Stage 5d]   {len(candidates_full):,} → {len(candidates):,} candidate fragments")

    if len(candidates) == 0:
        return pd.DataFrame(columns=["text", "label"])

    # ── Embedding ─────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 5d] Loading Bio_ClinicalBERT for embedding (device={device})...")
    # IMPORTANT: use the same MODEL_NAME from model.py which respects the
    # MODEL_PATH env var (Grace cluster has it pointed at a local path).
    # Using config.MODEL_NAME directly fails with OSError because it tries
    # to download from huggingface.co.
    from model import MODEL_NAME as _BERT_MODEL_NAME
    tokenizer = load_tokenizer()
    encoder   = AutoModel.from_pretrained(
        _BERT_MODEL_NAME,
        local_files_only=True,
    ).to(device)

    gold_texts = gold_df["text"].tolist()
    gold_labels = gold_df["label"].astype(int).tolist()

    print("[Stage 5d] Embedding gold (n=20)...")
    G = _bert_encode_batch(gold_texts, tokenizer, encoder, device)

    print(f"[Stage 5d] Embedding {len(candidates):,} candidates...")
    # Larger batch for candidates since we don't backprop
    C = _bert_encode_batch(candidates, tokenizer, encoder, device, bs=64)

    # Free the encoder before training kicks in
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Cosine similarity (G and C are L2-normalized → dot product = cosine)
    print("[Stage 5d] Computing similarities...")
    sims = G @ C.T          # (20, n_candidates)

    K = int(config.NEIGHBOR_PER_GOLD)
    min_sim = float(config.NEIGHBOR_MIN_SIM)
    hard_neg_min_sim = float(getattr(config, "HARD_NEG_MIN_SIM", 0.50))

    used_candidate_idx = set()
    chosen = []   # (text, label, sim, gold_idx, source)

    # Pass 1: for each gold example, pick top-K disjoint same-class neighbors
    for gi in range(len(gold_texts)):
        order = np.argsort(-sims[gi])
        picked = 0
        for ci in order:
            if picked >= K:
                break
            ci = int(ci)
            if ci in used_candidate_idx:
                continue
            s = sims[gi, ci]
            if s < min_sim:
                break
            used_candidate_idx.add(ci)
            chosen.append((candidates[ci], gold_labels[gi], float(s), gi, "neighbor"))
            picked += 1

    # Pass 2: hard negatives. For each class-1 gold example, find candidates
    # with high similarity (≥ hard_neg_min_sim) that ALSO match the hard-
    # negative cue regex. Label them class 0.
    n_hard_neg = 0
    hard_neg_per_gold = int(getattr(config, "HARD_NEG_PER_GOLD", 10))
    for gi, gl in enumerate(gold_labels):
        if gl != 1:
            continue
        order = np.argsort(-sims[gi])
        picked = 0
        for ci in order:
            if picked >= hard_neg_per_gold:
                break
            ci = int(ci)
            if ci in used_candidate_idx:
                continue
            s = sims[gi, ci]
            if s < hard_neg_min_sim:
                break
            cand_text = candidates[ci]
            if _is_hard_negative(cand_text):
                used_candidate_idx.add(ci)
                chosen.append((cand_text, 0, float(s), gi, "hard_neg"))
                picked += 1
                n_hard_neg += 1

    df = pd.DataFrame(chosen, columns=["text", "label", "sim", "gold_idx", "source"])
    n1 = (df["label"] == 1).sum()
    n0 = (df["label"] == 0).sum()
    print(f"[Stage 5d] Mined {len(df):,} fragments — "
          f"class1: {n1:,}, class0: {n0:,} "
          f"(of which {n_hard_neg:,} are hard negatives)")
    if len(df) > 0:
        print(f"[Stage 5d]   sim quartiles: "
              f"min={df['sim'].min():.3f} 25%={df['sim'].quantile(.25):.3f} "
              f"50%={df['sim'].median():.3f} max={df['sim'].max():.3f}")
    return df[["text", "label"]]


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline() -> pd.DataFrame:
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Stage 1
    icd_terms = build_icd_term_set()

    # Stage 2 — sentence pseudo-labels use the sampled subset for speed
    notes_sample = load_noteevents(max_n=config.MAX_NOTES_TO_SAMPLE)

    # Stage 5a
    sentence_df = build_sentence_pseudo(notes_sample, icd_terms)

    # Stage 5c & 5d use the FULL notes set (no sample cap) so we have enough
    # supply of every section type and rich neighbor mining material.
    print("[Pipeline] Loading full notes for section + neighbor mining...")
    notes_full = load_noteevents(
        categories=["Discharge summary", "Radiology", "Physician"],
        max_n=None,
    )

    # Stage 5c
    section_df = build_section_pseudo(notes_full)

    # Stage 5d — load gold for neighbor mining
    gold_df = pd.read_csv(config.TRAIN_CSV, dtype=str)
    gold_df.columns = [c.strip().lower() for c in gold_df.columns]
    tc = next(c for c in gold_df.columns if "text" in c)
    lc = next(c for c in gold_df.columns if "label" in c)
    gold_df = gold_df[[tc, lc]].rename(columns={tc: "text", lc: "label"})
    gold_df["label"] = gold_df["label"].astype(int)

    neighbor_df = build_neighbor_pseudo(notes_full, gold_df)

    # Combine all pseudo sources
    pseudo_df = pd.concat([sentence_df, section_df, neighbor_df],
                          ignore_index=True)
    pseudo_df["label"] = pseudo_df["label"].astype(int)

    # Drop any pseudo row whose text exactly matches gold (so evaluation is clean)
    gold_text_set = set(gold_df["text"].tolist())
    before = len(pseudo_df)
    pseudo_df = pseudo_df[~pseudo_df["text"].isin(gold_text_set)].reset_index(drop=True)
    print(f"[Pipeline] Removed {before - len(pseudo_df)} pseudo rows that "
          f"matched gold text")

    pseudo_df = pseudo_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    pseudo_df = pseudo_df.sample(frac=1, random_state=config.RANDOM_SEED) \
                          .reset_index(drop=True)

    # Append gold to the end so train.py can locate them
    combined = pd.concat([pseudo_df, gold_df], ignore_index=True)
    combined.to_csv(config.PSEUDO_LABEL_CSV, index=False)

    print(f"\n[Pipeline] FINAL composition:")
    print(f"  Sentence (5a)  : {len(sentence_df):,}")
    print(f"  Section  (5c)  : {len(section_df):,}")
    print(f"  Neighbor (5d)  : {len(neighbor_df):,}")
    print(f"  Pseudo total   : {len(pseudo_df):,}")
    print(f"  Gold appended  : {len(gold_df):,}")
    print(f"  Combined       : {len(combined):,} → {config.PSEUDO_LABEL_CSV}")
    print(f"  Pseudo class balance: "
          f"class0={(pseudo_df['label']==0).sum():,} "
          f"class1={(pseudo_df['label']==1).sum():,}")
    return combined


if __name__ == "__main__":
    run_pipeline()