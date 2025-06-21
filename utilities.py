
import fitz
import pytesseract
from rapidfuzz import fuzz, process
import json
import re
import sys
from datetime import datetime
import os
import hashlib
from datetime import date
import calendar

USE_OCR = True
USE_REGEX_EXTRACTION = True
USE_FIELD_WEIGHTS = True
USE_NORMALIZATION = True

FIELD_WEIGHTS = {
    "MBLNR": 3,
    "MJAHR": 2,
    "Purchase Order Number": 2,
    "Delivery Note Number": 3,
    "Delivery Note Date": 2,
    "Vendor - Name 1": 3,
    "Vendor - Name 2": 1,
    "Vendor - Address - Street": 1,
    "Vendor - Address - Number": 1,
    "Vendor - Address - ZIP Code": 1,
    "Vendor - Address - City": 1,
    "Vendor - Address - Country": 1,
    "Vendor - Address - Region": 1
}

def is_probable_agb_page(text):
    """
    Returns True if a page likely contains AGBs or similar annexes.

    Uses word count, keyword presence, and lack of metadata fields.
    """
    text_lower = text.lower()
    word_count = len(text_lower.split())

    agb_keywords = [
        "agb", "allgemeine geschäftsbedingungen", "bedingungen", "haftung", "zahlung",
        "lieferung", "datenschutz", "rückgabe", "vertrag", "recht", "widerruf"
    ]

    doc_header_keywords = ["lieferschein", "purchase order", "mblnr", "vendor"]

    has_agb_terms = any(kw in text_lower for kw in agb_keywords)
    has_header_terms = any(h in text_lower for h in doc_header_keywords)

    return word_count > 500 and has_agb_terms and not has_header_terms

def normalize(text):
    """
    Normalizes a text string for fuzzy matching.

    This function:
    - Converts the text to lowercase.
    - Applies replacement rules from NORMALIZATION_MAP to reduce formatting variations.
    - Strips leading/trailing whitespace.

    Args:
        text (str): The input text to normalize.

    Returns:
        str: A cleaned and normalized version of the text.
    """
    NORMALIZATION_MAP = {
        "str.": "straße",
        "str": "straße",
        "gmbh": "",
        "ug": "",
        "co.": "",
        "e.k.": "",
        "&": "und"
    }
    text = text.lower()
    for k, v in NORMALIZATION_MAP.items():
        text = text.replace(k, v)
    return text.strip()

def extract_text_from_doc(path):
    """
    Extracts text from each page of a PDF document.

    - Tries to extract machine-readable text using PyMuPDF.
    - Falls back to OCR using pytesseract if text is empty and USE_OCR is True.
    - Converts all text to lowercase.
    - Optionally prints page text if SHOW_PAGE_TEXT is True.

    Args:
        path (str): The file path to the input PDF.

    Returns:
        List[str]: A list of page-level text strings.
    """
    doc = fitz.open(path)
    text_pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if not text.strip() and USE_OCR:
            img = page.get_pixmap().pil_tobytes(format="PNG")
            text = pytesseract.image_to_string(img)
        text = text.lower()
        text_pages.append(text)
    return text_pages

KNOWN_REGEX_PATTERNS = {
    "Delivery Note Number": re.compile(r"(lieferschein[- ]?nr\.?|delivery note)[^\w]*([\w\-\/]{4,})", re.IGNORECASE),
    "Purchase Order Number": re.compile(r"(bestell[- ]?nr\.?|purchase order)[^\w]*([\w\-\/]{4,})", re.IGNORECASE),
    "MBLNR": re.compile(r"(mblnr)[^\d]*?(\d{5,})", re.IGNORECASE)
}

STOP_WORDS = {
    "und", "co", "kg", "gmbh", "ug", "e.k.", "str", "str.", 
}
MIN_SEGMENT_LENGTH = 3

def extract_known_fields(text):
    """
    Extracts key metadata fields from the given text using predefined regex patterns.

    Uses the KNOWN_REGEX_PATTERNS dictionary to search for:
    - Delivery Note Number
    - Purchase Order Number
    - MBLNR (Material Document Number)

    Args:
        text (str): The input text to search.

    Returns:
        Dict[str, str]: A dictionary of extracted fields and their matched values.
    """
    extracted = {}
    for label, pattern in KNOWN_REGEX_PATTERNS.items():
        match = pattern.search(text)
        if match:
            extracted[label] = match.group(2)
    return extracted

def date_variants(iso_datetime_str):
    """
    Given e.g. "2017-06-29T00:00:00.000",
    return a list of common human‐readable formats.
    """
    dt = datetime.fromisoformat(iso_datetime_str).date()
    day, mon, year = dt.day, dt.month, dt.year
    # numeric
    yield f"{day:02d}.{mon:02d}.{year}"
    yield f"{day:02d}-{mon:02d}-{year}"
    yield f"{day:02d}/{mon:02d}/{str(year)[2:]}"  # two-digit year
    # German month‐name with dot
    yield f"{day}. {calendar.month_name[mon]} {year}"
    yield f"{day}. {calendar.month_name[mon][:3]} {year}"
    # English month‐day
    yield f"{calendar.month_name[mon]} {day}, {year}"
    yield f"{calendar.month_name[mon][:3]} {day}, {year}"
    # Plain month‐year combos
    yield f"{calendar.month_name[mon]} {year}"
    yield f"{calendar.month_name[mon][:3]} {year}"


def generate_signature(entry):
    """
    Generates a normalized text signature for a given reference entry.

    This function creates n-gram text segments (up to 4 words) for each non-empty field
    in the entry. These segments are later used for fuzzy matching against document text.

    Args:
        entry (dict): A dictionary representing a reference record (e.g., from reference.json).

    Returns:
        List[Tuple[str, str]]: A list of (field_name, text_segment) tuples representing
                               normalized n-grams from the reference fields.
    """
    signature = []
    for key, val in entry.items():
        if val:
            if key == "Delivery Note Date":
                for seg in date_variants(val):
                    seg_norm = seg.lower().strip()
                    if len(seg_norm) < MIN_SEGMENT_LENGTH:
                        continue
                    signature.append((key, seg_norm))
                continue
            val = normalize(str(val)) if USE_NORMALIZATION else str(val)
            words = val.split()
            for n in range(1, min(4, len(words)) + 1):
                for i in range(len(words) - n + 1):
                    segment = " ".join(words[i:i+n])
                    if any(tok in STOP_WORDS for tok in segment.split()) or len(segment) < MIN_SEGMENT_LENGTH:
                        continue
                    signature.append((key, segment))
    return signature



def match_with_weights(signature, text):
    """
    Matches a reference signature against input text using partial fuzzy matching,
    applying weights to more important fields.

    Args:
        signature (List[Tuple[str, str]]): The reference signature generated via `generate_signature`.
        text (str): The full document or page text to compare against.

    Returns:
        Tuple[List[Tuple[str, str, int]], int, Dict[str, int]]:
            - List of matched (field, segment, score) triples.
            - Total weighted score.
            - Dictionary of max individual field scores.
    """
    raw_matches = []
    # 1) collect *all* passes >= threshold
    for field, seg in signature:
        seg_score = fuzz.partial_ratio(seg, text)
        if seg_score >= 85:
            raw_matches.append((field, seg, seg_score))

    # 2) pick the single best (seg,score) for each field
    best_per_field = {}
    for field, seg, score in raw_matches:
        prev = best_per_field.get(field)
        if prev is None or score > prev[1]:
            best_per_field[field] = (seg, score)

    # 3) build the final match_list and compute the weighted total
    match_list = []
    total_score = 0
    field_scores = {}
    for field, (seg, score) in best_per_field.items():
        weight = FIELD_WEIGHTS.get(field, 1) if USE_FIELD_WEIGHTS else 1
        total_score += score * weight
        match_list.append((field, seg, score))
        field_scores[field] = score

    return match_list, total_score, field_scores