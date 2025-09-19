#!/usr/bin/env python
"""
Phase 2 (initial): Preprocessing for Knowledge Extraction

This script sets up the Phase 2 pipeline skeleton and implements the first step:
- Read curated corpus from Phase 1 (corpus_sorted.csv)
- Normalize text (case, whitespace, abbreviations)
- Biomedical-aware sentence tokenization using scispaCy
- Write preprocessed output for downstream steps (JSONL)

Inputs/Outputs (defaults use repo root paths):
- Input CSV:   outputs/corpus_sorted.csv (PMID, Title, Abstract, ..., Missing_Abstract)
- Output JSONL: outputs/phase2_preprocessed.jsonl
- Log file:     outputs/phase2_log.txt

Dependencies:
- spacy
- scispacy
- a scispaCy model (e.g., en_core_sci_sm or en_core_sci_lg)

Example:
    python 2_extract_data.py \
        --input d:\\Misc\\BioMedRedundancy\\outputs\\corpus_sorted.csv \
        --output-dir d:\\Misc\\BioMedRedundancy\\outputs \
        --spacy-model en_core_sci_lg
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import spacy  # type: ignore
    from scispacy.abbreviation import AbbreviationDetector  # type: ignore
except Exception as e:  # pragma: no cover
    print(
        "Missing dependencies: please `pip install spacy scispacy` and a scispaCy model, e.g., `pip install https://github.com/allenai/scispacy/releases/download/v0.5.4/en_core_sci_lg-0.5.4.tar.gz`.",
        file=sys.stderr,
    )
    raise


# -----------------------------
# Text Normalization
# -----------------------------

def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_unicode(text: str) -> str:
    # Keep a canonical form (NFC) for consistent processing
    return unicodedata.normalize("NFC", text)


def normalize_sentence(text: str, abbr_map: Dict[str, str]) -> str:
    """Normalize a sentence deterministically:
    - Unicode normalize (NFC)
    - Lowercase
    - Expand abbreviations using provided mapping (word-boundary exact match)
    - Collapse whitespace
    """
    s = _normalize_unicode(text)
    s = s.lower()
    # Apply abbreviation expansions (replace short forms with long forms)
    # Sort by length (desc) to avoid partial overlaps like "ra" vs "rct"
    for short, long in sorted(abbr_map.items(), key=lambda kv: len(kv[0]), reverse=True):
        if not short or not long:
            continue
        pattern = re.compile(rf"\b{re.escape(short.lower())}\b")
        s = pattern.sub(long.lower(), s)
    s = _collapse_ws(s)
    return s


# -----------------------------
# NLP Pipeline
# -----------------------------

def build_nlp(model_name: str):
    """Load a scispaCy model and attach AbbreviationDetector.
    We keep the parser enabled for robust sentence segmentation; disable heavy components if present.
    """
    try:
        nlp = spacy.load(model_name, disable=["ner", "lemmatizer", "textcat"])  # keep parser for sents
    except Exception as e:
        logging.error(f"Failed to load spaCy model '{model_name}': {e}")
        print(f"Failed to load spaCy model '{model_name}'. Ensure it is installed.", file=sys.stderr)
        raise
    # Add abbreviation detector for biomedical abbreviations
    if "abbreviation_detector" not in nlp.pipe_names:
        nlp.add_pipe("abbreviation_detector")  # type: ignore
    return nlp


def extract_abbreviation_map(doc) -> Dict[str, str]:
    """Build a mapping of short-form -> long-form using scispaCy AbbreviationDetector.
    If multiple long forms appear for the same short, prefer the longest string (heuristic).
    """
    mapping: Dict[str, str] = {}
    abbrs = getattr(doc._, "abbreviations", [])
    for ab in abbrs:
        short = ab.text.strip()
        long = ab._.long_form.text.strip() if hasattr(ab._, "long_form") and ab._.long_form is not None else ""
        if not short or not long:
            continue
        # Prefer the longest expansion if duplicates arise
        if short not in mapping or len(long) > len(mapping[short]):
            mapping[short] = long
    return mapping


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 2 (initial): Normalization & biomedical sentence tokenization")
    parser.add_argument("--input", default=str(Path.cwd() / "outputs" / "corpus_sorted.csv"), help="Path to Phase 1 output CSV")
    parser.add_argument("--output-dir", default=str(Path.cwd() / "outputs"), help="Directory to place outputs/logs")
    parser.add_argument("--spacy-model", default="en_core_sci_lg", help="scispaCy model name (e.g., en_core_sci_sm, en_core_sci_lg)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N records (for quick tests)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=str(output_dir / "phase2_log.txt"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("=== Phase 2 (initial) run started ===")
    logging.info(f"Input file: {input_path}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(f"spaCy model: {args.spacy_model}")

    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        logging.error(msg)
        print(msg, file=sys.stderr)
        sys.exit(2)

    # Load model
    nlp = build_nlp(args.spacy_model)

    # TEMPORARY: Use dummy data for testing
    USE_DUMMY_DATA = True  # Set to False to revert to CSV processing
    
    if USE_DUMMY_DATA:
        # Dummy test data
        dummy_data = {
            "PMID": ["33789270"],
            "Title": ["Cochlear Implantation and Other Treatments in Single-Sided Deafness and Asymmetric Hearing Loss: Results of a National Multicenter Study Including a Randomized Controlled Trial"],
            "Abstract": ["""Introduction: Cochlear implantation is a recent approach proposed to treat single-sided deafness (SSD) and asymmetric hearing loss (AHL). Several cohort studies showed its effectiveness on tinnitus and variable results on binaural hearing. The main objective of this study is to assess the outcomes of cochlear implantation and other treatment options in SSD/AHL on quality of life. Methods: This prospective multicenter study was conducted in 7 tertiary university hospitals and included an observational cohort study of SSD/AHL adult patients treated using contralateral routing of the signal (CROS) hearing aids or bone-anchored hearing systems (BAHSs) or who declined all treatments, and a randomized controlled trial in subjects treated by cochlear implantation, after failure of CROS and BAHS trials. In total, 155 subjects with SSD or AHL, with or without associated tinnitus, were enrolled. After 2 consecutive trials with CROS hearing aids and BAHSs on headband, all subjects chose any of the 4 treatment options (abstention, CROS, BAHS, or cochlear implant [CI]). The subjects who opted for a CI were randomized between 2 arms (CI vs. initial observation). Six months after the treatment choice, quality of life was assessed using both generic (EuroQoL-5D, EQ-5D) and auditory-specific quality-of-life indices (Nijmegen Cochlear implant Questionnaire [NCIQ] and Visual Analogue Scale [VAS] for tinnitus severity). Performances for speech-in-noise recognition and localization were measured as secondary outcomes. Results: CROS was chosen by 75 subjects, while 51 opted for cochlear implantation, 18 for BAHSs, and 11 for abstention. Six months after treatment, both EQ-5D VAS and auditory-specific quality-of-life indices were significantly better in the "CI" arm versus "observation" arm. The mean effect of the CI was particularly significant in subjects with associated severe tinnitus (mean improvement of 20.7 points ± 19.7 on EQ-5D VAS, 20.4 ± 12.4 on NCIQ, and 51.4 ± 35.4 on tinnitus). No significant effect of the CI was found on binaural hearing results. Before/after comparisons showed that the CROS and BAHS also improved significantly NCIQ scores (for CROS: +7.7, 95% confidence interval [95% CI] = [4.5; 10.8]; for the BAHS: +14.3, 95% CI = [7.9; 20.7]). Conclusion: Cochlear implantation leads to significant improvements in quality of life in SSD and AHL patients, particularly in subjects with associated severe tinnitus, who are thereby the best candidates to an extension of CI indications."""]
        }
        df = pd.DataFrame(dummy_data)
        logging.info("Using dummy data for testing")
    else:
        # Original CSV processing
        df = pd.read_csv(input_path)
        if "PMID" not in df.columns or "Title" not in df.columns or "Abstract" not in df.columns:
            msg = "Input CSV missing required columns: PMID, Title, Abstract"
            logging.error(msg)
            print(msg, file=sys.stderr)
            sys.exit(2)

        # Skip papers with missing abstracts (as per Phase 1 policy)
        if "Missing_Abstract" in df.columns:
            df = df[~df["Missing_Abstract"].astype(bool)]

    if args.limit is not None:
        df = df.head(args.limit)

    out_jsonl = output_dir / "phase2_preprocessed.jsonl"
    written = 0

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing", unit="paper"):
            pmid = str(row.get("PMID", "")).strip()
            title = str(row.get("Title", "")).strip()
            abstract = str(row.get("Abstract", "")).strip()

            # Combine title + abstract for abbreviation detection and sentence segmentation
            base_text = title
            if abstract:
                base_text = f"{title}. {abstract}" if not title.endswith(".") else f"{title} {abstract}"

            # NLP pass
            doc = nlp(base_text)
            abbr_map = extract_abbreviation_map(doc)

            # Sentence tokenization
            sentences = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]

            # Normalization per sentence (case + abbreviation expansion + whitespace)
            normalized_sentences = [normalize_sentence(s, abbr_map) for s in sentences]

            record = {
                "PMID": pmid,
                "sentences": sentences,
                "normalized_sentences": normalized_sentences,
                "abbreviation_map": abbr_map,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    logging.info(f"Wrote {out_jsonl} with {written} records")
    logging.info("=== Phase 2 (initial) run completed ===")
    print(f"Phase 2 preprocessing complete. Wrote {written} records to: {out_jsonl}")


if __name__ == "__main__":
    main()