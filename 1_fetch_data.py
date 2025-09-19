#!/usr/bin/env python
"""
Phase 1: Data Curation & Validation

Reads PMIDs from pmids.txt, fetches PubMed metadata via NCBI Entrez (BioPython),
standardizes publication dates with imputations, validates records, sorts
chronologically, and outputs:
- corpus_sorted.csv (PMID, Title, Abstract, Authors, Journal, PubDate, Missing_Abstract)
- phase1_log.txt (actions, imputations, issues)
- phase1_summary.csv (summary statistics)

Configuration for Entrez credentials (priority high → low):
1) CLI flags: --email, --api-key
2) Environment variables: NCBI_EMAIL/ENTREZ_EMAIL, NCBI_API_KEY/ENTREZ_API_KEY
3) .env.local file in project root with the same keys (simple parser inside)

Usage:
    python 1_fetch_data.py --input pmids.txt

Requirements: pandas, biopython, tqdm
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio import Entrez
from tqdm import tqdm

# -----------------------------
# Logging setup (configured in main after output_dir is created)
# -----------------------------
logger = logging.getLogger(__name__)

# -----------------------------
# Utils: .env.local loading
# -----------------------------
ENV_FILE = ".env.local"


def load_dot_env_local(env_path: Path) -> Dict[str, str]:
    """Tiny .env parser (KEY=VALUE lines). Returns found key-values without altering os.environ.
    Lines starting with # are ignored. Quotes around values are stripped.
    """
    result: Dict[str, str] = {}
    if not env_path.exists():
        return result
    try:
        for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = k.strip()
            val = v.strip().strip('"').strip("'")
            result[key] = val
    except Exception as e:
        logger.warning(f"Failed to read {env_path}: {e}")
    return result


# -----------------------------
# Entrez configuration
# -----------------------------
ACCEPTED_EMAIL_KEYS = ["NCBI_EMAIL", "ENTREZ_EMAIL"]
ACCEPTED_API_KEYS = ["NCBI_API_KEY", "ENTREZ_API_KEY"]


def resolve_entrez_credentials(cli_email: Optional[str], cli_api_key: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve Entrez email and api key using CLI args, env, or .env.local."""
    if cli_email:
        email = cli_email
    else:
        # env
        email = None
        for k in ACCEPTED_EMAIL_KEYS:
            if os.getenv(k):
                email = os.getenv(k)
                break
        # .env.local
        if email is None:
            env = load_dot_env_local(Path(ENV_FILE))
            for k in ACCEPTED_EMAIL_KEYS:
                if env.get(k):
                    email = env.get(k)
                    break

    if cli_api_key:
        api_key = cli_api_key
    else:
        # env
        api_key = None
        for k in ACCEPTED_API_KEYS:
            if os.getenv(k):
                api_key = os.getenv(k)
                break
        # .env.local
        if api_key is None:
            env = load_dot_env_local(Path(ENV_FILE))
            for k in ACCEPTED_API_KEYS:
                if env.get(k):
                    api_key = env.get(k)
                    break

    return email, api_key


# -----------------------------
# Input reading & deduplication
# -----------------------------

def read_pmids(input_path: Path) -> List[str]:
    """Read PMIDs as strings, strip, keep only digits, preserve first occurrence order.
    Logs duplicate removals.
    """
    if not input_path.exists():
        logger.error(f"PMID file not found: {input_path}")
        raise FileNotFoundError(f"PMID file not found: {input_path}")

    raw_ids: List[str] = []
    with input_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # keep numeric tokens only
            m = re.match(r"^\d+$", s)
            if m:
                raw_ids.append(s)

    # log duplicates
    counts = Counter(raw_ids)
    dups = [pmid for pmid, c in counts.items() if c > 1]
    if dups:
        logger.info(f"Found duplicate PMIDs (will keep first occurrence): {len(dups)} unique duplicated PMIDs")

    # stable unique
    seen = set()
    unique_pmids: List[str] = []
    for pmid in raw_ids:
        if pmid in seen:
            # log each duplicate removal
            logger.debug(f"Duplicate PMID removed: {pmid}")
            continue
        seen.add(pmid)
        unique_pmids.append(pmid)

    logger.info(f"Loaded PMIDs: {len(raw_ids)} | Unique: {len(unique_pmids)} | Removed: {len(raw_ids) - len(unique_pmids)}")
    return unique_pmids


# -----------------------------
# PubDate parsing & imputation
# -----------------------------
MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _coerce_date(y: Optional[int], m: Optional[int], d: Optional[int], source: str, pmid: str) -> date:
    """Apply imputation rules and log them.
    - If year and month only: set day=15
    - If year only: set month=7, day=1
    - If full date given, use as is
    """
    if y and m and d:
        return date(y, m, d)
    if y and m and not d:
        logger.info(f"Impute day=15 for PMID={pmid} from {source} (year+month only)")
        return date(y, m, 15)
    if y and not m:
        logger.info(f"Impute month=7, day=1 for PMID={pmid} from {source} (year only)")
        return date(y, 7, 1)
    # As a last resort, pick mid-year of current year to keep ordering deterministic, but log warning
    logger.warning(f"Failed to parse date for PMID={pmid}; imputing 2000-07-01")
    return date(2000, 7, 1)


def _int_or_none(s: Optional[str]) -> Optional[int]:
    try:
        return int(s) if s is not None else None
    except Exception:
        return None


def _parse_month_token(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    t = token.strip().lower()
    # Sometimes token is a number
    if t.isdigit():
        try:
            m = int(t)
            if 1 <= m <= 12:
                return m
        except Exception:
            return None
    return MONTHS.get(t)


def parse_pubdate_fields(article: dict, pmid: str) -> date:
    """Extract publication date with priority: PubDate > EPubDate > MedlineDate.
    Works on the dictionary produced by Entrez.read (parsed XML).
    """
    # Helpers to navigate nested dicts/lists safely
    def _get(path: List[str], node):
        cur = node
        for key in path:
            if isinstance(cur, list):
                # take first item by default
                cur = cur[0] if cur else None
            if cur is None:
                return None
            cur = cur.get(key) if isinstance(cur, dict) else None
        return cur

    # Prefer Journal -> JournalIssue -> PubDate
    pub = _get(["MedlineCitation", "Article", "Journal", "JournalIssue", "PubDate"], article)
    if isinstance(pub, dict):
        y = _int_or_none(pub.get("Year"))
        m = _parse_month_token(pub.get("Month"))
        d = _int_or_none(pub.get("Day"))
        if y:
            return _coerce_date(y, m, d, source="PubDate", pmid=pmid)
        # MedlineDate fallback inside PubDate
        medline_str = pub.get("MedlineDate")
        if medline_str:
            dt = parse_medline_date(medline_str, pmid, source="MedlineDate(PubDate)")
            if dt:
                return dt

    # ArticleDate (Electronic / EPub)
    art_dates = _get(["MedlineCitation", "Article", "ArticleDate"], article)
    if isinstance(art_dates, list) and art_dates:
        # Prefer DateType == Electronic
        selected = None
        for dct in art_dates:
            if isinstance(dct, dict) and dct.get("DateType", "").lower().startswith("elect"):
                selected = dct
                break
        if selected is None:
            selected = art_dates[0]
        y = _int_or_none(selected.get("Year"))
        m = _parse_month_token(selected.get("Month"))
        d = _int_or_none(selected.get("Day"))
        if y:
            return _coerce_date(y, m, d, source="EPubDate/ArticleDate", pmid=pmid)

    # MedlineDate at MedlineCitation level (less structured)
    medline_date = _get(["MedlineCitation", "Article", "Journal", "JournalIssue", "PubDate", "MedlineDate"], article)
    if medline_date:
        dt = parse_medline_date(medline_date, pmid, source="MedlineDate")
        if dt:
            return dt

    # Last resort: DateCompleted or DateCreated
    date_completed = _get(["MedlineCitation", "DateCompleted"], article)
    if isinstance(date_completed, dict):
        y = _int_or_none(date_completed.get("Year"))
        m = _int_or_none(date_completed.get("Month"))
        d = _int_or_none(date_completed.get("Day"))
        if y:
            return _coerce_date(y, m, d, source="DateCompleted", pmid=pmid)

    date_created = _get(["MedlineCitation", "DateCreated"], article)
    if isinstance(date_created, dict):
        y = _int_or_none(date_created.get("Year"))
        m = _int_or_none(date_created.get("Month"))
        d = _int_or_none(date_created.get("Day"))
        if y:
            return _coerce_date(y, m, d, source="DateCreated", pmid=pmid)

    # default
    return _coerce_date(None, None, None, source="Unknown", pmid=pmid)


def parse_medline_date(text: str, pmid: str, source: str) -> Optional[date]:
    """Parse MedlineDate strings like '2023 Mar-Apr', '2023', '2023 Winter', '1998 Dec 12'.
    Follows imputation rules from the README.
    """
    s = (text or "").strip()
    # Year only
    m = re.match(r"^(\d{4})$", s)
    if m:
        y = int(m.group(1))
        logger.info(f"Impute month=7, day=1 for PMID={pmid} from {source} (year only: '{s}')")
        return date(y, 7, 1)

    # Year + month (possibly ranges)
    m = re.match(r"^(\d{4})\s+([A-Za-z]{3,9})(?:-[A-Za-z]{3,9})?$", s)
    if m:
        y = int(m.group(1))
        mon = _parse_month_token(m.group(2))
        if mon:
            logger.info(f"Impute day=15 for PMID={pmid} from {source} (year+month: '{s}')")
            return date(y, mon, 15)

    # Year + month + day
    m = re.match(r"^(\d{4})\s+([A-Za-z]{3,9})\s+(\d{1,2})$", s)
    if m:
        y = int(m.group(1))
        mon = _parse_month_token(m.group(2))
        d = int(m.group(3))
        if mon:
            return date(y, mon, d)

    # Seasons or unrecognized tokens → mid-year
    m = re.match(r"^(\d{4})\s+(Spring|Summer|Fall|Autumn|Winter).*$", s, flags=re.IGNORECASE)
    if m:
        y = int(m.group(1))
        logger.info(f"Impute month=7, day=1 for PMID={pmid} from {source} (seasonal: '{s}')")
        return date(y, 7, 1)

    logger.warning(f"Unrecognized MedlineDate format for PMID={pmid}: '{s}'; imputing 2000-07-01")
    return date(2000, 7, 1)


# -----------------------------
# Fetching with retry & backoff
# -----------------------------

def fetch_pubmed_batch(pmid_batch: List[str], retries: int = 3, backoff_base: float = 1.0) -> Optional[dict]:
    """Fetch a batch of PMIDs with Entrez. Returns parsed dict or None on failure after retries."""
    attempt = 0
    while attempt <= retries:
        try:
            with Entrez.efetch(db="pubmed", id=",".join(pmid_batch), retmode="xml") as handle:
                records = Entrez.read(handle)
            return records
        except Exception as e:
            wait = backoff_base * (2 ** attempt)
            logger.warning(f"Entrez fetch failed (attempt {attempt + 1}/{retries + 1}): {e}; retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1
    logger.error(f"Failed to fetch batch after {retries + 1} attempts: first PMID in batch {pmid_batch[0] if pmid_batch else 'N/A'}")
    return None


# -----------------------------
# Field extraction helpers
# -----------------------------

def extract_text_fields(article: dict) -> Tuple[str, str, str, str]:
    """Return (title, abstract, authors, journal). Title/abstract empty strings if missing."""
    def _get(path: List[str], node):
        cur = node
        for key in path:
            if isinstance(cur, list):
                cur = cur[0] if cur else None
            if cur is None:
                return None
            cur = cur.get(key) if isinstance(cur, dict) else None
        return cur

    title = _get(["MedlineCitation", "Article", "ArticleTitle"], article) or ""

    # Abstract: join parts if list of AbstractText
    abs_node = _get(["MedlineCitation", "Article", "Abstract", "AbstractText"], article)
    abstract = ""
    if isinstance(abs_node, list):
        # Each item may be a str or dict with attributes
        parts = []
        for part in abs_node:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                txt = part.get("_", "")
                parts.append(txt)
        abstract = "\n".join(p.strip() for p in parts if p and p.strip())
    elif isinstance(abs_node, str):
        abstract = abs_node

    # Authors
    auths = _get(["MedlineCitation", "Article", "AuthorList"], article)
    authors: List[str] = []
    if isinstance(auths, list):
        for a in auths:
            if not isinstance(a, dict):
                continue
            if a.get("CollectiveName"):
                authors.append(str(a.get("CollectiveName")))
                continue
            last = a.get("LastName")
            initials = a.get("Initials")
            if last and initials:
                authors.append(f"{last} {initials}")
            elif last:
                authors.append(str(last))
    authors_str = "; ".join(authors)

    # Journal
    journal = _get(["MedlineCitation", "Article", "Journal", "Title"], article) or ""

    return str(title), str(abstract), authors_str, str(journal)


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Fetch, validate, and sort corpus metadata from PMIDs")
    # Fix input to the pmids.txt in repo root by default (absolute path recommended)
    default_pmids = str(Path.cwd() / "pmids.txt")
    parser.add_argument("--input", default=default_pmids, help="Path to pmids.txt (one PMID per line)")
    parser.add_argument("--email", default=None, help="Entrez email (overrides env/.env.local)")
    parser.add_argument("--api-key", default=None, help="Entrez API key (overrides env/.env.local)")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for Entrez efetch")
    parser.add_argument("--output-dir", default=str(Path.cwd() / "outputs"), help="Directory to place outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reconfigure logging to write inside output_dir
    for h in list(logging.root.handlers):
        logging.root.removeHandler(h)
    logging.basicConfig(
        filename=str(output_dir / "phase1_log.txt"),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    email, api_key = resolve_entrez_credentials(args.email, args.api_key)
    if not email:
        msg = (
            "Entrez email is required. Provide via --email, NCBI_EMAIL/ENTREZ_EMAIL env, "
            "or set it in .env.local."
        )
        logging.error(msg)
        print(msg, file=sys.stderr)
        sys.exit(2)

    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    logging.info("=== Phase 1 run started ===")
    logging.info(f"Using Entrez email={email}; api_key={'set' if api_key else 'not set'}")
    logging.info(f"Input file: {input_path}")
    logging.info(f"Output dir: {output_dir}")

    pmids = read_pmids(input_path)
    if not pmids:
        logging.error("No valid PMIDs found. Exiting.")
        print("No valid PMIDs found in input.", file=sys.stderr)
        sys.exit(1)

    records: List[Dict] = []

    # Fetch in batches
    for i in tqdm(range(0, len(pmids), args.batch_size), desc="Fetching", unit="batch"):
        batch = pmids[i : i + args.batch_size]
        data = fetch_pubmed_batch(batch)
        if data is None:
            # record failures; skip this batch
            logging.error(f"Skipping batch starting at index {i} due to repeated failures.")
            continue
        # data is a dict with key 'PubmedArticle'
        arts = data.get("PubmedArticle", [])
        # Map back by PMID for robustness
        for art in arts:
            try:
                # PMID
                pm = None
                # Path: MedlineCitation -> PMID
                pm_node = art.get("MedlineCitation", {}).get("PMID")
                if isinstance(pm_node, list) and pm_node:
                    pm = str(pm_node[0])
                elif isinstance(pm_node, dict) and pm_node.get("_id"):
                    pm = str(pm_node.get("_id"))
                elif isinstance(pm_node, str):
                    pm = pm_node
                else:
                    # fallback via ArticleIdList
                    ids = art.get("PubmedData", {}).get("ArticleIdList", [])
                    for idn in ids:
                        if isinstance(idn, dict) and idn.get("IdType") == "pubmed":
                            pm = str(idn.get("_"))
                            break
                pmid_val = pm or ""

                # Extract fields
                title, abstract, authors, journal = extract_text_fields(art)
                pub_dt = parse_pubdate_fields(art, pmid=pmid_val)

                records.append(
                    {
                        "PMID": pmid_val,
                        "Title": title,
                        "Abstract": abstract,
                        "Authors": authors,
                        "Journal": journal,
                        "PubDate": pub_dt.isoformat(),
                        "_PubDate_obj": pub_dt,  # for sorting/validation
                    }
                )
            except Exception as e:
                logging.exception(f"Failed to extract record in batch around PMID={batch[0] if batch else 'N/A'}: {e}")

    if not records:
        logging.error("No records fetched successfully. Exiting.")
        print("No records fetched successfully.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)

    # Validation: duplicates by PMID (keep earliest date)
    if df["PMID"].duplicated().any():
        logging.info("Duplicate PMIDs in fetched data detected; keeping earliest PubDate per PMID")
        df.sort_values(["PMID", "_PubDate_obj"], ascending=[True, True], inplace=True)
        df = df.drop_duplicates(subset=["PMID"], keep="first")

    # Missing abstract flag
    df["Missing_Abstract"] = df["Abstract"].fillna("").str.strip().eq("")

    # Sort chronologically by standardized date
    df.sort_values("_PubDate_obj", inplace=True)

    # Finalize columns and write outputs
    out_df = df[["PMID", "Title", "Abstract", "Authors", "Journal", "PubDate", "Missing_Abstract"]].copy()
    out_df.to_csv(output_dir / "corpus_sorted.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    # Summary (printed to terminal)
    total = len(out_df)
    missing_abs = int(out_df["Missing_Abstract"].sum())
    print("Phase 1 summary:")
    print(f"  total_papers: {total}")
    print(f"  missing_abstracts: {missing_abs}")

    # Write PMIDs with missing abstracts (one PMID per line)
    missing_pmids_path = output_dir / "missing_abstract_pmids.txt"
    out_df.loc[out_df["Missing_Abstract"], "PMID"].astype(str).to_csv(missing_pmids_path, index=False, header=False)

    logging.info(f"Wrote {output_dir / 'corpus_sorted.csv'} with {total} rows")
    logging.info(f"Missing abstracts: {missing_abs}")
    logging.info(f"Wrote {missing_pmids_path} with {missing_abs} PMIDs")
    logging.info("=== Phase 1 run completed ===")


if __name__ == "__main__":
    main()