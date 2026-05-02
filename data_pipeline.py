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
    r"\b(diagnosis of|assessment:|plan:|principal diagnosis|"
    r"primary diagnosis|acute|chronic)\b",
    re.IGNORECASE,
)

# Imaging reports often contain lots of ICD-like terms but are labeled non-codable
# when the impression is normal/negative or hedged.
_IMAGING_CONTEXT_PATTERN = re.compile(
    r"\b(impression|findings|final report|comparison:|compared with|"
    r"ct\b|mri\b|cta\b|mra\b|ultrasound|u/s|doppler|cxr\b|x-?ray|"
    r"echocardiogram|tte\b|lvef|ventricular|valve|regurgitation)\b",
    re.IGNORECASE,
)
_NEG_IMAGING_PATTERN = re.compile(
    r"\b(no evidence of|normal appearance|unremarkable|no acute|"
    r"unchanged|stable|patent|limited examination|images unavailable|"
    r"not present|negative for|without evidence of|no significant change|"
    r"probable degenerative changes|likely benign|no abnormality)\b",
    re.IGNORECASE,
)

_PROCEDURE_PATTERN = re.compile(
    r"\b(s/p|status post|underwent|procedure|repair|stent|orif|"
    r"intubat(ed|ion)|dialysis|hemodialysis|pci|cath lab)\b",
    re.IGNORECASE,
)

# Strong positive: explicit diagnosis/condition statements
_DIAGNOSIS_PATTERN = re.compile(
    r"\b(diagnos(is|ed|ed with)|assessment|impression|presents? with|"
    r"consistent with|findings? (of|consistent)|history of|h/o\b|"
    r"known (history of|to have)|found to have|confirmed|"
    r"ruled out|r/o\b|positive for|negative for)\b",
    re.IGNORECASE,
)

# Strong positive: procedures with action verbs
_PROCEDURE_STRONG = re.compile(
    r"\b(s/p\b|status post|underwent|procedure|repair|stent|orif|"
    r"intubat(ed|ion)|dialysis|hemodialysis|pci|cath\b|catheterization|"
    r"resection|biopsy|debridement|amputation|colostomy|tracheostomy|"
    r"thoracentesis|paracentesis|bronchoscopy|endoscopy|colonoscopy)\b",
    re.IGNORECASE,
)

# Specific medical condition terms (high precision)
_CONDITION_PATTERN = re.compile(
    r"\b(pneumonia|sepsis|septic|bacteremia|cellulitis|abscess|"
    r"fracture|laceration|contusion|hemorrhage|hematoma|thrombosis|"
    r"embolism|infarction|ischemia|necrosis|edema|effusion|"
    r"insufficiency|failure|obstruction|stenosis|occlusion|"
    r"hypertension|hypotension|diabetes|diabetic|anemia|"
    r"infection|infective|inflammatory|malignancy|carcinoma|"
    r"metastasis|tumor|mass|lesion|ulcer|wound|injury|trauma)\b",
    re.IGNORECASE,
)

# Class 0: medication administration (NOT codable by itself)
_MEDICATION_ADMIN = re.compile(
    r"\b(given|administered|prescribed|started on|continued on|"
    r"increased|decreased|titrated|held|discontinued|ordered)\b"
    r"[\s\w]{0,20}"
    r"\b(mg|mcg|units?|tabs?|capsules?|drip|infusion|iv\b|po\b|prn\b|qd\b|bid\b|tid\b|qid\b)\b",
    re.IGNORECASE,
)

# Class 0: vitals and lab values (NOT codable)
_VITALS_LABS_PATTERN = re.compile(
    r"\b(blood pressure|bp\b|heart rate|hr\b|temperature|temp\b|"
    r"respiratory rate|rr\b|oxygen saturation|spo2|o2 sat|"
    r"sodium|potassium|creatinine|bun\b|glucose|hemoglobin|hematocrit|"
    r"wbc\b|platelets?|inr\b|pt\b|ptt\b|ph\b|pco2|po2)\s*"
    r"[\d\.\-/]+",
    re.IGNORECASE,
)

# Class 0: pure plan/instruction statements
_PLAN_INSTRUCTION = re.compile(
    r"\b(will (monitor|check|follow|continue|start|hold|obtain|consult)|"
    r"please (monitor|check|follow|ensure|note|see)|"
    r"recommend(ed|ation)?|advised|counseled|educated|"
    r"will be discharged|plan to discharge|anticipate discharge)\b",
    re.IGNORECASE,
)

# Catches "afib", "tachycardic", "bradycardic" etc. — these ARE codable
_CARDIAC_RHYTHM_PATTERN = re.compile(
    r"\b(afib|atrial fibrillation|tachycardic|tachycardia|"
    r"bradycardic|bradycardia|flutter|svt\b|vfib|v-?fib|"
    r"heart block|arrhythmia|palpitations)\b",
    re.IGNORECASE,
)

# Negative findings — NOT codable even if disease word present
_NEGATIVE_FINDING = re.compile(
    r"\b(no evidence of|without evidence of|negative for|"
    r"not present|no acute|no active|no new|no significant change|"
    r"within normal limits|wnl\b|unremarkable|"
    r"no (?:fracture|mass|lesion|effusion|pneumothorax|"
    r"infiltrate|consolidation|fistula|abscess|dvt|pe\b))\b",
    re.IGNORECASE,
)

# Care directives — NOT codable
_CARE_DIRECTIVE = re.compile(
    r"\b(do not (resuscitate|intubate|re-?intubate)|dnr\b|dni\b|"
    r"comfort (care|measures)|hospice|palliative|"
    r"is not to be|not to be (re-?intubated|resuscitated)|"
    r"taken off|weaned off|discontinued|code status)\b",
    re.IGNORECASE,
)

# Medication change instructions — NOT codable
_MED_INSTRUCTION = re.compile(
    r"\b(you (have been|are|should|will|must|need to)|"
    r"please (take|continue|stop|avoid|call|return)|"
    r"take your|your (dose|medication|prescription)|"
    r"been (started|switched|changed|taken) (on|off|to))\b",
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
    Filters out fragments (< 5 words) and sentences exceeding MAX_LENGTH words.
    """
    try:
        nltk.data.find("tokenizers/punkt")
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        # Offline-friendly fallback: split on punctuation and newlines.
        sentences = re.split(r"(?<=[\.!\?])\s+|\n+", text)
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
    Score-based labeler. Accumulate evidence for/against codability.
    Returns 1, 0, or None (discard ambiguous).
    """
    lower = sentence.lower()
    words = set(re.findall(r"[a-zA-Z]+", lower))
    word_count = len(sentence.split())

    # ── Hard discard (structural non-sentences) ───────────────────────────
    if _PURE_NUMERIC.match(sentence):
        return 0
    if _HEADER_PATTERN.match(sentence):
        return 0
    if word_count < 4:
        return None

    # ── Scoring ───────────────────────────────────────────────────────────
    score = 0  # positive = class 1, negative = class 0

    # === STRONG CLASS 0 SIGNALS (hard veto) ===
    # Explicit negative findings
    if re.search(
        r"\b(no evidence of|without evidence of|negative for|"
        r"not present|no acute|no active|no new|"
        r"no (?:fracture|mass|lesion|effusion|pneumothorax|"
        r"infiltrate|consolidation|fistula|abscess|dvt|pe\b)|"
        r"within normal limits|wnl\b)\b",
        sentence, re.IGNORECASE
    ):
        score -= 4

    # Admission/discharge administrative headers
    if re.search(
        r"\b(admission date|discharge date|date of admission|"
        r"date of discharge|date of birth|dob\b)\b",
        sentence, re.IGNORECASE
    ):
        score -= 5

    # Radiology exam headers (timestamp + exam type)
    if re.search(
        r"\b(portable ap|chest ap|chest pa|ap chest|pa and lateral|"
        r"chest \(portable\)|chest\s*\(ap\)|upright ap)\b",
        sentence, re.IGNORECASE
    ):
        score -= 4

    # Stable existing findings (not new)
    if re.search(
        r"\b(stable|unchanged|persistent|known|chronic|existing|"
        r"previously (noted|seen|identified|described))\b.{0,30}"
        r"\b(effusion|opacity|atelectasis|lesion|mass|hematoma|"
        r"opacification|infiltrate|consolidation)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Physical exam headers
    if re.search(
        r"^(\[\*\*[\w\s]+\*\*\]\s*)?(physical exam|pe:|vital signs|"
        r"review of systems|ros:|general:|heent:|cardiovascular:|"
        r"respiratory:|abdomen:|extremities:|neuro:)\s*$",
        sentence, re.IGNORECASE
    ):
        score -= 5

    # Monitoring/assessment plans without confirmed diagnosis
    if re.search(
        r"\b(cont(inue)? to (assess|monitor|watch|follow)|"
        r"assess for (s/s|signs|symptoms)|"
        r"monitor for|watch for|follow for)\b",
        sentence, re.IGNORECASE
    ) and not re.search(
        r"\b(sepsis|pneumonia|failure|hemorrhage|infarction|"
        r"embolism|thrombosis|ischemia)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Normal structures
    if re.search(
        r"\b(unremarkable|normal (size|appearance|limits?)|"
        r"heart is normal|normal cardiomediastinal|"
        r"no pericardial|patent and normal|grossly normal)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Stable/unchanged imaging
    if re.search(
        r"\b(stable|unchanged|no interval change|no significant change)"
        r".{0,30}(appearance|since|from|compared)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Care directives
    if re.search(
        r"\b(dnr\b|dni\b|do not (resuscitate|intubate)|"
        r"comfort (care|measures)|hospice|palliative|"
        r"not to be re-?intubated|code status)\b",
        sentence, re.IGNORECASE
    ):
        score -= 4

    # Patient-facing instructions
    if re.search(
        r"\b(you (have been|are|should|will|must)|"
        r"please (take|continue|stop|avoid|call|return)|"
        r"take your (pain|blood|heart|water)|your medication)\b",
        sentence, re.IGNORECASE
    ):
        score -= 4

    # Medication administration
    if re.search(
        r"\b(given|administered|received)\b.{0,30}"
        r"\b(mg|mcg|mEq|units?|ml\b|iv\b|po\b)\b",
        sentence, re.IGNORECASE
    ):
        score -= 2

    # Vitals with numbers
    if re.search(
        r"\b(blood pressure|bp\b|heart rate|hr\b|"
        r"temperature|temp\b|o2 sat|spo2)\s*[\d/\.]+",
        sentence, re.IGNORECASE
    ):
        score -= 2

    # Lab values with numbers
    if re.search(
        r"\b(sodium|potassium|creatinine|wbc|hgb|inr|troponin|lactate)"
        r"\s*(is|was|of|:)?\s*[\d\.]+",
        sentence, re.IGNORECASE
    ):
        score -= 2

    # Social history
    if re.search(
        r"\b(social history|tobacco|smoking|alcohol|drug use|"
        r"lives (alone|with)|occupation|retired|homeless)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Follow-up only
    if re.search(
        r"\b(follow.?up (with|in|at)|return (to|in|for)|"
        r"next appointment|outpatient|primary care)\b",
        sentence, re.IGNORECASE
    ):
        score -= 2

    # Allergy lines
    if re.search(
        r"\b(allergies?|nkda|no known (drug )?allergies?|allergic to)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # "Normal" findings with specific structures
    if re.search(
        r"\b(normal (flow|signal|appearing|caliber|contour|echotexture)|"
        r"is normal|are normal|appears normal|appear normal|"
        r"grossly normal|essentially normal)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Waiting/monitoring instructions
    if re.search(
        r"\b(waiting for|wait for|awaiting|monitoring for|"
        r"to wake|to be transferred|to go back|will wake)\b",
        sentence, re.IGNORECASE
    ):
        score -= 3

    # Third-party context (visiting, family member, etc.)
    if re.search(
        r"\b(his wife|her husband|his mother|her father|"
        r"family member|visiting|visitor|accompan)\b",
        sentence, re.IGNORECASE
    ):
        score -= 4

    # Medication list lines (numbered drug entries)
    if re.search(
        r"^\s*\d+[\.\)]\s+\w+.{5,50}\b(mg|mcg|tablet|capsule|sig:)\b",
        sentence, re.IGNORECASE
    ):
        score -= 4

    # === STRONG CLASS 1 SIGNALS ===
    # Explicit procedures
    if re.search(
        r"\b(s/p\b|status post|underwent|procedure performed|"
        r"repair|stent|orif|intubat(ed|ion)|dialysis|hemodialysis|"
        r"pci\b|catheterization|resection|biopsy|debridement|"
        r"amputation|colostomy|tracheostomy|thoracentesis|"
        r"paracentesis|bronchoscopy|endoscopy|colonoscopy|"
        r"thoracentesis|craniotomy|laminectomy)\b",
        sentence, re.IGNORECASE
    ):
        score += 4

    # Sinus tachycardia/bradycardia written in monitoring shorthand
    if re.search(
        r"\b(sinus tachy|sinus brady|s\.tachy|s\.brady|"
        r"cvs[:\s].{0,20}(tachy|brady|afib|flutter)|"
        r"tele[:\s].{0,20}(tachy|brady|afib|pvcs?))\b",
        sentence, re.IGNORECASE
    ):
        score += 4

    # Imaging with new/worsening finding
    if re.search(
        r"\b(new|worsening|increasing|progressive|interval increase|"
        r"interval development|newly|developed|appeared)\b.{0,40}"
        r"\b(opacity|opacification|effusion|infiltrate|"
        r"consolidation|atelectasis|mass|lesion|hematoma)\b",
        sentence, re.IGNORECASE
    ):
        score += 4

    # Cardiac rhythms/arrhythmias
    if re.search(
        r"\b(afib|a-?fib|atrial fibrillation|atrial flutter|"
        r"tachycardic|tachycardia|bradycardic|bradycardia|"
        r"pvc\b|pvcs\b|pac\b|pacs\b|svt\b|v-?fib|vfib|"
        r"heart block|arrhythmia|sinus tach|sinus brady|"
        r"st elevation|st depression|lbbb|rbbb)\b",
        sentence, re.IGNORECASE
    ):
        score += 4

    # Specific diagnoses
    if re.search(
        r"\b(pneumonia|sepsis|septic shock|bacteremia|cellulitis|"
        r"abscess|fracture|laceration|hemorrhage|hematoma|"
        r"thrombosis|embolism|infarction|ischemia|necrosis|"
        r"edema|effusion|insufficiency|failure|obstruction|"
        r"stenosis|occlusion|hypertension|hypotension|"
        r"diabetes|diabetic|anemia|infection|malignancy|"
        r"carcinoma|metastasis|tumor|ulcer|pancreatitis|"
        r"appendicitis|peritonitis|meningitis|encephalitis|"
        r"hydrocephalus|stroke|infarct|ischemic|hemorrhagic|"
        r"dvt\b|pe\b|pulmonary embolism|deep vein)\b",
        sentence, re.IGNORECASE
    ):
        score += 3

    # Abnormal physical exam findings
    if re.search(
        r"\b(\+|positive)\s*(tenderness|guarding|rigidity|rebound)|"
        r"\b(distended|rigid|tender)\s+(abdomen|belly|abd\b)|"
        r"\b(decreased|diminished|absent)\s+(breath sounds|bowel sounds|pulses?)|"
        r"\bsigns? of (infection|inflammation|ischemia|failure)\b",
        sentence, re.IGNORECASE
    ):
        score += 3

    # Diagnosis framing language
    if re.search(
        r"\b(diagnos(is|ed|ed with)|consistent with|"
        r"findings? (of|consistent with)|presents? with|"
        r"history of|h/o\b|known (history of|to have)|"
        r"found to have|confirmed|assessment:)\b",
        sentence, re.IGNORECASE
    ):
        score += 2

    # ICD term overlap bonus
    icd_overlap = len(words & icd_terms)
    if icd_overlap >= 4:
        score += 3
    elif icd_overlap >= 2:
        score += 1

    # Imaging with positive finding (not negated)
    if re.search(
        r"\b(impression:|impression\n|IMPRESSION:)\b",
        sentence, re.IGNORECASE
    ) and score > 0:
        score += 2

    # ── Decision ──────────────────────────────────────────────────────────
    if score >= 4:
        return 1
    elif score <= -2:
        return 0
    else:
        return None   # discard ambiguous — cleaner labels beat more data


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Build pseudo-labeled dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_pseudo_dataset(notes_df: pd.DataFrame, icd_terms: frozenset) -> pd.DataFrame:
    class1, class0 = [], []
    seen = set()  # deduplicate

    for text in notes_df["TEXT"]:
        for sentence in split_sentences(text):
            if sentence in seen:
                continue
            seen.add(sentence)

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
