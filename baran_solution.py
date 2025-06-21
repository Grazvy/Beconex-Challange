
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
from collections import Counter, defaultdict
import math
from PyPDF2 import PdfReader
import io
from typing import List
import fitz  # PyMuPDF
from PIL import Image
import pytesseract


# ================================
# CONFIGURATION FLAGS
# ================================

# PDF_PATH = "data/batch_3_2021.pdf"  # Default PDF for single document processing
# PDF_TEST = "data/batch_2_2019_2020/20190917_Konica.pdf"
# PDF_DIR = "data/batch_1_2017_2018"  # Directory for batch processing
# PDF_PATH_Batch = "batch.pdf"  # Default batch PDF for processing multiple documents


PDF_PATH = "solution/superbatch_merged.pdf"  # Default PDF for single document processing
REFERENCE_JSON_PATH = "data/SAP_data.json"
LOG_DIR = "debug_logs"
OUTPUT_JSON = "output.json"

# Behavior toggles
USE_OCR = True
USE_REGEX_EXTRACTION = True
USE_FIELD_WEIGHTS = True
USE_NORMALIZATION = True
RETURN_TOP_K = 3
MIN_MATCH_FIELDS = 2
CONFIDENCE_THRESHOLD = 800

SHOW_PAGE_TEXT = False
SHOW_REGEX_HINTS = True
SHOW_RANKING = True
SHOW_RANKING_FULL = True

MAX_WINDOW_SIZE = 3
MIN_WINDOW_SIZE = 2

# Weighting of fields
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

NORMALIZATION_MAP = {
    "str.": "straße",
    "str": "straße",
    "gmbh": "",
    "ug": "",
    "co.": "",
    "e.k.": "",
    "&": "und"
}

KNOWN_REGEX_PATTERNS = {
    "Delivery Note Number": re.compile(r"(lieferschein[- ]?nr\.?|delivery note)[^\w]*([\w\-\/]{4,})", re.IGNORECASE),
    "Purchase Order Number": re.compile(r"(bestell[- ]?nr\.?|purchase order)[^\w]*([\w\-\/]{4,})", re.IGNORECASE),
    "MBLNR": re.compile(r"(mblnr)[^\d]*?(\d{5,})", re.IGNORECASE)
}

STOP_WORDS = {
    "und", "co", "kg", "gmbh", "ug", "e.k.", "str", "str.", 
}
MIN_SEGMENT_LENGTH = 3


# ================================
# UTILITIES
# ================================

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
    text = text.lower()
    for k, v in NORMALIZATION_MAP.items():
        text = text.replace(k, v)
    return text.strip()


def extract_text_from_doc(
    path: str,
    use_ocr: bool = True,
    show_page_text: bool = False,
    ocr_dpi: int = 300
) -> List[str]:
    """
    Extracts text from each page of a PDF document.

    - Tries to extract machine-readable text using PyMuPDF.
    - Falls back to OCR using pytesseract if text is empty and use_ocr is True.
    - Converts all text to lowercase.
    - Optionally prints page text if show_page_text is True.

    Args:
        path (str): The file path to the input PDF.
        use_ocr (bool): Whether to run OCR on pages with no text.
        show_page_text (bool): Whether to print each page’s text.
        ocr_dpi (int): DPI to render pages for OCR (higher = better quality).

    Returns:
        List[str]: A list of page-level text strings.
    """
    text_pages: List[str] = []

    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            try:
                # 1) Native text extraction
                raw = page.get_text().strip()

                # 2) Fallback to OCR if needed
                if not raw and use_ocr:
                    pix = page.get_pixmap(dpi=ocr_dpi)
                    img = Image.open(io.BytesIO(pix.tobytes()))
                    raw = pytesseract.image_to_string(img).strip()

                # Normalize
                text = raw.lower()
            except Exception as e:
                # Log & continue
                print(f"[Warning] page {i} extraction failed: {e}")
                text = ""
            # print(f"{text}")
            text_pages.append(text)

            if show_page_text:
                print(f"--- Page {i} ---\n{text}\n--------------------")

    return text_pages


def detect_page_count(pages):
    """
    Attempts to detect the total number of pages in the document based on page footer patterns.

    Looks for common patterns such as:
    - "Page X of Y"
    - "Seite X von Y" (German)

    Args:
        pages (List[str]): List of text strings for each page.

    Returns:
        int or None: The detected total page count (Y), or None if not found.
    """
    patterns = [
        re.compile(r"(page|seite)?\s*(\d{1,2})\s*(of|von)\s*(\d{1,2})", re.IGNORECASE),
        re.compile(r"(\d{1,2})\s*(von|of)\s*(\d{1,2})", re.IGNORECASE)
    ]
    for text in pages:
        for pattern in patterns:
            m = pattern.findall(text)
            for match in m:
                nums = [int(s) for s in match if s.isdigit()]
                if len(nums) == 2:
                    return nums[1]
    return None


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

# 2023-06-26T00:00:00.000
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
    # For this format 2017-06-28
    yield f"{year}-{mon:02d}-{day:02d}"  # ISO format
    
# Tino-Schwierzina-StraBe 86
def address_variants(address):
    """
    Generates common variants of an address string for fuzzy matching.

    Args:
        address (str): The input address string.

    Returns:
        List[str]: A list of normalized address variants.
    """
    address = normalize(address)
    # Replace common abbreviations
    address = address.replace("straße", "str.")
    address = address.replace("strasse", "str.")
    # Add common variations
    yield address
    yield address.replace("str.", "straße")
    yield address.replace("str.", "strasse")
    yield address.replace("straße", "strasse")
    

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
                    # print(f"  • Date segment: {seg_norm}")
                    if len(seg_norm) < MIN_SEGMENT_LENGTH:
                        continue
                    signature.append((key, seg_norm))
                continue
            if key == "Vendor - Address - Street":
                for seg in address_variants(val):
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
        base_w = FIELD_WEIGHTS.get(field, 1) if USE_FIELD_WEIGHTS else 1
        # boost/attenuate by IDF
        # idf = IDF_WEIGHTS.get((field, seg), 1.0)
        w = base_w * 1
        total_score += score * w
        match_list.append((field, seg, score))
        field_scores[field] = score

    return match_list, total_score, field_scores
    # score = 0
    # match_list = []
    # field_scores = {}
    # for field, seg in signature:
    #     seg_score = fuzz.partial_ratio(seg, text)
    #     if seg_score >= 85:
    #         weight = FIELD_WEIGHTS.get(field, 1) if USE_FIELD_WEIGHTS else 1
    #         score += seg_score * weight
    #         match_list.append((field, seg, seg_score))
    #         field_scores[field] = max(field_scores.get(field, 0), seg_score)
    # return match_list, score, field_scores


def match_with_token_sort(signature, text):
    """
    Matches a reference signature against input text using token sort ratio,
    without applying field weighting.

    Args:
        signature (List[Tuple[str, str]]): The reference signature.
        text (str): The full document or page text to compare against.

    Returns:
        Tuple[List[Tuple[str, str, int]], int]:
            - List of matched (field, segment, score) triples.
            - Cumulative match score (sum of individual match scores).
    """
    score = 0
    match_list = []
    for field, seg in signature:
        seg_score = fuzz.token_sort_ratio(seg, text)
        if seg_score >= 85:
            match_list.append((field, seg, seg_score))
            score += seg_score
    return match_list, score


def log_debug_output(filename, content):
    """
    Saves debug information to a log file in the designated LOG_DIR.

    Automatically creates the directory if it doesn't exist.

    Args:
        filename (str): The name of the file to write.
        content (str): The textual content to save.
    """
    pass
    # os.makedirs(LOG_DIR, exist_ok=True)
    # with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
    #     f.write(content)


def top_k_matches(entries, text, k=3, confidence_threshold=CONFIDENCE_THRESHOLD):
    """
    Finds the top-k best-matching reference entries for a given document text.

    It uses field-weighted fuzzy matching and filters out weak matches based on:
    - Minimum number of matched fields (MIN_MATCH_FIELDS)
    - Minimum total score (CONFIDENCE_THRESHOLD)

    Args:
        entries (List[dict]): List of reference entries to match against.
        text (str): The document text to match.
        k (int): The number of top results to return.

    Returns:
        List[Tuple[int, List[Tuple[str, str, int]], dict]]:
            A sorted list of top-k matches, each consisting of:
            - The total weighted score.
            - List of matched segments.
            - The matched reference entry.
    """
    scored = []
    for i, entry in enumerate(entries):
        # if i % 50 == 0:
        #     print(f"🔍 Matching against reference {i+1}/{len(entries)}...")
        sig = generate_signature(entry)
        matches, score, fields = match_with_weights(sig, text)
        if len(matches) >= MIN_MATCH_FIELDS and score >= confidence_threshold:
            scored.append((score, matches, entry))
    return sorted(scored, key=lambda x: -x[0])[:k]

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

PROCESS_PAGEWISE = False

def _print_matches(matches, indent=""):
    for rank, (score, mlist, entry) in enumerate(matches, start=1):
        print(f"{indent}Rank {rank} | Score: {score} | MBLNR={entry['MBLNR']}")
        if SHOW_RANKING_FULL:
            for f, val, s in mlist:
                w = FIELD_WEIGHTS.get(f, 1)
                print(f"{indent}  - {f} | '{val}' | Score: {s} | Weight: {w}")

# ================================
# MAIN LOGIC
# ================================

def process_pdf_pagewise(pdf_path, references):
    pages = extract_text_from_doc(pdf_path)
    total_pages = len(pages)
    page_info = []
    last_mblnr = None
    last_mjahr = None

    # Precompute full-doc regex
    full_text = "\n".join(pages)
    regex_hits = extract_known_fields(full_text) if USE_REGEX_EXTRACTION else {}
    
    texts = []

    # Pagewise matching with sliding-window fallback
    for idx, text in enumerate(pages, start=1):
        texts.append(text)
        if is_probable_agb_page(text):
            # assign to previous
            mblnr, mjahr = last_mblnr, last_mjahr
        else:
            # try single-page match
            matches = top_k_matches(references, text, RETURN_TOP_K, CONFIDENCE_THRESHOLD)
            if matches:
                score, mlist, entry = matches[0]
                print(f"📄 Page {idx}: MATCH FOUND with score {score}")
                mblnr, mjahr = entry['MBLNR'], entry['MJAHR']
            else:
                # sliding-window fallback
                found = False
                window_candidates = []
                
                def to_score_map(ms):
                    return { e["MBLNR"]: sc for sc, _, e in ms }
            
                    
                if idx - 1 > 0:
                    prev_text = texts[idx-2]  # previous page text
                    current_text = text  # current page text
                    
                    prev_map = to_score_map(top_k_matches(references, prev_text, RETURN_TOP_K))
                    current_map = to_score_map(top_k_matches(references, current_text, 2, 300))
                    
                    summed = []
                    for mblnr in set(prev_map) | set(current_map):
                        if current_map.get(mblnr, 0) < 300:
                            continue
                        s = prev_map.get(mblnr, 0) + current_map.get(mblnr, 0)
                        summed.append((s, mblnr))
                        if SHOW_RANKING_FULL:
                            print(f"  • {mblnr}: prev={prev_map.get(mblnr,0)}, curr={current_map.get(mblnr,0)}, sum={s}")
                    if summed:
                        best_sum, best_mblnr = max(summed, key=lambda x: x[0])
                        if SHOW_RANKING_FULL:
                            print(f"  • Best sum: {best_sum} for MBLNR={best_mblnr}")
                        if best_sum >= CONFIDENCE_THRESHOLD + 400:
                            mblnr, mjahr = best_mblnr, last_mjahr
                            found = True
                            window_candidates.append((best_sum, best_mblnr, mjahr))
                                        
                
                if idx + 1 <= total_pages:
                    # check next pages in the window
                    current_text = texts[idx-1]  # current page text
                    next_text = pages[idx]  # next page text
                    next_map = to_score_map(top_k_matches(references, next_text, RETURN_TOP_K))
                    current_map = to_score_map(top_k_matches(references, current_text, 2, 300))
                    summed = []
                    for mblnr in set(next_map) | set(current_map):
                        if current_map.get(mblnr, 0) < 300:
                            continue  # skip low-confidence matches
                        s = next_map.get(mblnr, 0) + current_map.get(mblnr, 0)
                        summed.append((s, mblnr))
                        if SHOW_RANKING_FULL:
                            print(f"  • {mblnr}: next={next_map.get(mblnr,0)}, curr={current_map.get(mblnr,0)}, sum={s}")
                    if summed:
                        best_sum, best_mblnr = max(summed, key=lambda x: x[0])
                        if SHOW_RANKING_FULL:
                            print(f"  • Best sum: {best_sum} for MBLNR={best_mblnr}")
                        if best_sum >= CONFIDENCE_THRESHOLD:
                            mblnr, mjahr = best_mblnr, last_mjahr
                            found = True
                            window_candidates.append((best_sum, best_mblnr, mjahr))
                
                if window_candidates:
                    # use the best candidate from the window
                    window_candidates.sort(reverse=True, key=lambda x: x[0])
                    best_sum, mblnr_b, mjahr_b = window_candidates[0]
                    if SHOW_RANKING_FULL:
                        print(f"  • Best window candidate: {best_sum} for MBLNR={mblnr}")
                    found = True
                    mblnr = mblnr_b
                    mjahr = mjahr_b
                    
                else:
                    print(f"⚠️ No high-confidence matches on page {idx}. Falling back to regex or previous match…")
                    # fallback to regex or previous
                    mblnr = regex_hits.get('MBLNR', -1)
                    mjahr = regex_hits.get('MJAHR', -1)

        page_info.append({'page': idx, 'MBLNR': mblnr, 'MJAHR': mjahr})
        last_mblnr, last_mjahr = mblnr, mjahr

    # Group consecutive by same MBLNR
    blocks = []
    if page_info:
        current = page_info[0]
        start = current['page']
        cur_mblnr = current['MBLNR']
        cur_mjahr = current['MJAHR']
        for info in page_info[1:]:
            if info['MBLNR'] != cur_mblnr:
                blocks.append({'Page of batch where document starts': start,
                               'MBLNR': cur_mblnr, 'MJAHR': cur_mjahr})
                start = info['page']
                cur_mblnr = info['MBLNR']
                cur_mjahr = info['MJAHR']
        # last block
        blocks.append({'Page of batch where document starts': start,
                       'MBLNR': cur_mblnr, 'MJAHR': cur_mjahr})

    log_debug_output(os.path.basename(pdf_path)+".log.txt",
                     f"Pages: {[p['page'] for p in page_info]}\nBlocks: {blocks}\n")
    return blocks

# with open(REFERENCE_JSON_PATH, 'r', encoding='utf-8') as f:
#         references = json.load(f)
# IDF_WEIGHTS = compute_idf_weights(references)

def main():
    with open(REFERENCE_JSON_PATH, 'r', encoding='utf-8') as f:
        references = json.load(f)
    all_blocks = []
    # for fname in sorted(os.listdir(PDF_DIR)):
    #     if not fname.lower().endswith('.pdf'):
    #         continue
    #     pdf_path = os.path.join(PDF_DIR, fname)
    #     print(f"Processing {fname}...")
    all_blocks = process_pdf_pagewise(PDF_PATH, references)
    # for blk in blocks:
    #     blk['file'] = fname
    #     all_blocks.append(blk)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as out:
        json.dump(all_blocks, out, indent=2)
    print(f"Output.json written with {len(all_blocks)} entries.")


def main_test():
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = PDF_PATH
    
    result = []
    current = None
    with open(REFERENCE_JSON_PATH, "r", encoding="utf-8") as f:
        references = json.load(f)
        
    for pdf_file in os.listdir(pdf_path):
        if not pdf_file.lower().endswith('.pdf'):
            continue
        pdf_path_join = os.path.join(pdf_path, pdf_file)
        print(f"Processing {pdf_file}...")

        # Extract text per page
        pages = extract_text_from_doc(pdf_path_join)
        print("📊 Pages:", len(pages))
        
        texts = []

        # Detect and report total page count if possible
        detected_count = detect_page_count(pages)
        if detected_count:
            print("📌 Detected Page Count from text:", detected_count)
            
        blacklist = []

        # Filter out any pages likely to be AGB/annex
        filtered_pages = []
        for i, p in enumerate(pages, start=1):
            if is_probable_agb_page(p):
                print(f"📝 Skipping page {i}: probable AGB/annex")
                blacklist.append(i)
            else:
                filtered_pages.append((i, p))

        # Optional regex-based “quick hits” on entire doc
        if USE_REGEX_EXTRACTION:
            full_text = "\n".join(p for _, p in filtered_pages)
            print("🔎 Extracted Fields (regex on full doc):")
            for k, v in extract_known_fields(full_text).items():
                print(f"  - {k}: {v}")
                
        # ── Matching logic ──
        if PROCESS_PAGEWISE and len(filtered_pages) > 1:
            # Page-level matching: report true/false per page
            for page_num, text in filtered_pages:
                texts.append(text)
                matches = top_k_matches(references, text, RETURN_TOP_K)
                hit = len(matches) > 0
                print(f"📄 Page {page_num}: {'MATCH FOUND' if hit else 'no match'}")
                if hit and SHOW_RANKING:
                    for rank, (score, mlist, entry) in enumerate(matches, start=1):
                        print(f"  Rank {rank} | Score {score} | MBLNR={entry['MBLNR']}")
                elif hit == False:
                    print("⚠️ No high-confidence matches on full document. Falling back to page windows…")

                    # Gather all candidate windows that produced at least one match
                    window_candidates = []

                    def to_score_map(ms):
                        return { e["MBLNR"]: sc for sc, _, e in ms }

                    # --- prev + curr ---
                    if page_num > 0:
                        print(f"🔍 Scoring page {page_num-1} (prev) + {page_num} (curr)")
                        #print(texts[-2])
                        prev_map = to_score_map(top_k_matches(references, texts[-2], RETURN_TOP_K))
                        print(f"  • Previous page {page_num-1} matches: {prev_map}")
                        curr_map = to_score_map(top_k_matches(references, text, RETURN_TOP_K, 300))
                        print(f"  • Current page {page_num} matches: {curr_map}")
                        summed = []
                        for mblnr in set(prev_map) | set(curr_map):
                            s = prev_map.get(mblnr, 0) + curr_map.get(mblnr, 0)
                            summed.append((s, mblnr))
                            print(f"  • {mblnr}: prev={prev_map.get(mblnr,0)}, curr={curr_map.get(mblnr,0)}, sum={s}")
                        if summed:
                            best_sum, best_mblnr = max(summed, key=lambda x: x[0])
                            print(f"  • Best sum: {best_sum} for MBLNR={best_mblnr}")
                            window_candidates.append(((page_num-1, page_num), best_mblnr, best_sum))

                    # --- curr + next ---
                    if page_num < len(pages):
                        print(f"🔍 Scoring page {page_num} (curr) + {page_num+1} (next)")
                        correct_page_number = page_num
                        for i, p in enumerate(blacklist):
                            if p < correct_page_number:
                                correct_page_number -= 1
                        print(f"  • Corrected page number: {correct_page_number}")
                        next_map = to_score_map(top_k_matches(references, filtered_pages[correct_page_number], RETURN_TOP_K))
                        print(f"  • Next page {correct_page_number} matches: {next_map}")
                        curr_map = to_score_map(top_k_matches(references, text, RETURN_TOP_K, 300))
                        print(f"  • Current page {page_num} matches: {curr_map}")
                        summed = []
                        for mblnr in set(curr_map) | set(next_map):
                            s = curr_map.get(mblnr,0) + next_map.get(mblnr,0)
                            summed.append((s, mblnr))
                            print(f"  • {mblnr}: curr={curr_map.get(mblnr,0)}, next={next_map.get(mblnr,0)}, sum={s}")
                        if summed:
                            best_sum, best_mblnr = max(summed, key=lambda x: x[0])
                            print(f"  • Best sum: {best_sum} for MBLNR={best_mblnr}")
                            window_candidates.append(((page_num, page_num+1), best_mblnr, best_sum))

                    # pick the best neighbor-sum
                    if window_candidates:
                        print("\n🔍 Best neighbor matches:")
                        print(f"{window_candidates}")
                        pages_used, best_matches, best_score = max(window_candidates, key=lambda x: x[2])
                        print(f"Best match: {best_score}")
                        # print(f"  Pages {pages_used[0]}–{pages_used[1]}: combined score={best_score}")
                        if best_score >= CONFIDENCE_THRESHOLD:
                            print(f"\n✅ Best neighbor match pages {pages_used[0]}–{pages_used[1]}: combined score={best_score}")
                            continue
                
        else:
            # Full-document matching (original behavior)
            full_text = "\n".join(p for _, p in filtered_pages)
            top_matches = top_k_matches(references, full_text, RETURN_TOP_K, confidence_threshold=300)

            print("\n🏁 Final Top Matches:")
            if not top_matches:
                print("⚠️ No high-confidence matches on full document. Falling back to page windows…")
                hit = False

                # Sliding window over pages
                for window_size in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
                    print(f"\n🔍 Trying window size = {window_size} pages:")
                    # slide window over filtered_pages
                    for idx in range(len(filtered_pages) - window_size + 1):
                        page_nums, texts = zip(*filtered_pages[idx : idx + window_size])
                        combined = "\n".join(texts)
                        matches = top_k_matches(references, combined, RETURN_TOP_K)
                        if matches:
                            hit = True
                            window_str = f"pages {page_nums[0]}–{page_nums[-1]}"
                            print(f"✅ Match found on {window_str}:")
                            _print_matches(matches, indent="  ")
                            break
                    if hit:
                        break
            else:
                for i, (score, matches, entry) in enumerate(top_matches, start=1):
                    print(f"\nRank {i} | Score: {score}")
                    print(json.dumps({k: entry[k] for k in ('MBLNR', 'MJAHR')}, indent=2))
                    print("Matched Fields:")
                    for f, val, s in matches:
                        weight = FIELD_WEIGHTS.get(f, 1)
                        print(f"  - {f} | '{val}' | Score: {s} | Weight: {weight}")

    # Log debug output for the first few pages
    log_content = f"PDF: {pdf_path}\nFiltered Pages: {[n for n,_ in filtered_pages]}\n"
    log_content += "\n".join(p for _,p in filtered_pages[:3])
    log_debug_output(f"{os.path.basename(pdf_path)}.log.txt", log_content)



if __name__ == '__main__':
    main()









# def main():
#     if len(sys.argv) > 1:
#         pdf_path = sys.argv[1]
#     else:
#         pdf_path = PDF_PATH
    
#     result = []
#     current = None

#     # Load reference data
#     with open(REFERENCE_JSON_PATH, "r", encoding="utf-8") as f:
#         references = json.load(f)

#     # Extract text per page
#     pages = extract_text_from_doc(pdf_path)
#     print("📊 Pages:", len(pages))
    
#     texts = []

#     # Detect and report total page count if possible
#     detected_count = detect_page_count(pages)
#     if detected_count:
#         print("📌 Detected Page Count from text:", detected_count)
        
#     blacklist = []

#     # Filter out any pages likely to be AGB/annex
#     filtered_pages = []
#     for i, p in enumerate(pages, start=1):
#         if is_probable_agb_page(p):
#             print(f"📝 Skipping page {i}: probable AGB/annex")
#             blacklist.append(i)
#         else:
#             filtered_pages.append((i, p))

#     # Optional regex-based “quick hits” on entire doc
#     if USE_REGEX_EXTRACTION:
#         full_text = "\n".join(p for _, p in filtered_pages)
#         print("🔎 Extracted Fields (regex on full doc):")
#         for k, v in extract_known_fields(full_text).items():
#             print(f"  - {k}: {v}")
            
#     # ── Matching logic ──
#     if PROCESS_PAGEWISE and len(filtered_pages) > 1:
#         # Page-level matching: report true/false per page
#         for page_num, text in filtered_pages:
#             texts.append(text)
#             matches = top_k_matches(references, text, RETURN_TOP_K)
#             hit = len(matches) > 0
#             print(f"📄 Page {page_num}: {'MATCH FOUND' if hit else 'no match'}")
#             if hit and SHOW_RANKING:
#                 for rank, (score, mlist, entry) in enumerate(matches, start=1):
#                     print(f"  Rank {rank} | Score {score} | MBLNR={entry['MBLNR']}")
#             elif hit == False:
#                 print("⚠️ No high-confidence matches on full document. Falling back to page windows…")

#                 # Gather all candidate windows that produced at least one match
#                 window_candidates = []

#                 def to_score_map(ms):
#                     return { e["MBLNR"]: sc for sc, _, e in ms }

#                 # --- prev + curr ---
#                 if page_num > 0:
#                     print(f"🔍 Scoring page {page_num-1} (prev) + {page_num} (curr)")
#                     #print(texts[-2])
#                     prev_map = to_score_map(top_k_matches(references, texts[-2], RETURN_TOP_K))
#                     print(f"  • Previous page {page_num-1} matches: {prev_map}")
#                     curr_map = to_score_map(top_k_matches(references, text, RETURN_TOP_K, 300))
#                     print(f"  • Current page {page_num} matches: {curr_map}")
#                     summed = []
#                     for mblnr in set(prev_map) | set(curr_map):
#                         s = prev_map.get(mblnr, 0) + curr_map.get(mblnr, 0)
#                         summed.append((s, mblnr))
#                         print(f"  • {mblnr}: prev={prev_map.get(mblnr,0)}, curr={curr_map.get(mblnr,0)}, sum={s}")
#                     if summed:
#                         best_sum, best_mblnr = max(summed, key=lambda x: x[0])
#                         print(f"  • Best sum: {best_sum} for MBLNR={best_mblnr}")
#                         window_candidates.append(((page_num-1, page_num), best_mblnr, best_sum))

#                 # --- curr + next ---
#                 if page_num < len(pages):
#                     print(f"🔍 Scoring page {page_num} (curr) + {page_num+1} (next)")
#                     correct_page_number = page_num
#                     for i, p in enumerate(blacklist):
#                         if p < correct_page_number:
#                             correct_page_number -= 1
#                     print(f"  • Corrected page number: {correct_page_number}")
#                     next_map = to_score_map(top_k_matches(references, filtered_pages[correct_page_number], RETURN_TOP_K))
#                     print(f"  • Next page {correct_page_number} matches: {next_map}")
#                     curr_map = to_score_map(top_k_matches(references, text, RETURN_TOP_K, 300))
#                     print(f"  • Current page {page_num} matches: {curr_map}")
#                     summed = []
#                     for mblnr in set(curr_map) | set(next_map):
#                         s = curr_map.get(mblnr,0) + next_map.get(mblnr,0)
#                         summed.append((s, mblnr))
#                         print(f"  • {mblnr}: curr={curr_map.get(mblnr,0)}, next={next_map.get(mblnr,0)}, sum={s}")
#                     if summed:
#                         best_sum, best_mblnr = max(summed, key=lambda x: x[0])
#                         print(f"  • Best sum: {best_sum} for MBLNR={best_mblnr}")
#                         window_candidates.append(((page_num, page_num+1), best_mblnr, best_sum))

#                 # pick the best neighbor-sum
#                 if window_candidates:
#                     print("\n🔍 Best neighbor matches:")
#                     print(f"{window_candidates}")
#                     pages_used, best_matches, best_score = max(window_candidates, key=lambda x: x[2])
#                     print(f"Best match: {best_score}")
#                     # print(f"  Pages {pages_used[0]}–{pages_used[1]}: combined score={best_score}")
#                     if best_score >= CONFIDENCE_THRESHOLD:
#                         print(f"\n✅ Best neighbor match pages {pages_used[0]}–{pages_used[1]}: combined score={best_score}")
#                         continue
            
#     else:
#         # Full-document matching (original behavior)
#         full_text = "\n".join(p for _, p in filtered_pages)
#         top_matches = top_k_matches(references, full_text, RETURN_TOP_K, confidence_threshold=300)

#         print("\n🏁 Final Top Matches:")
#         if not top_matches:
#             print("⚠️ No high-confidence matches on full document. Falling back to page windows…")
#             hit = False

#             # Sliding window over pages
#             for window_size in range(MIN_WINDOW_SIZE, MAX_WINDOW_SIZE + 1):
#                 print(f"\n🔍 Trying window size = {window_size} pages:")
#                 # slide window over filtered_pages
#                 for idx in range(len(filtered_pages) - window_size + 1):
#                     page_nums, texts = zip(*filtered_pages[idx : idx + window_size])
#                     combined = "\n".join(texts)
#                     matches = top_k_matches(references, combined, RETURN_TOP_K)
#                     if matches:
#                         hit = True
#                         window_str = f"pages {page_nums[0]}–{page_nums[-1]}"
#                         print(f"✅ Match found on {window_str}:")
#                         _print_matches(matches, indent="  ")
#                         break
#                 if hit:
#                     break
#         else:
#             for i, (score, matches, entry) in enumerate(top_matches, start=1):
#                 print(f"\nRank {i} | Score: {score}")
#                 print(json.dumps({k: entry[k] for k in ('MBLNR', 'MJAHR')}, indent=2))
#                 print("Matched Fields:")
#                 for f, val, s in matches:
#                     weight = FIELD_WEIGHTS.get(f, 1)
#                     print(f"  - {f} | '{val}' | Score: {s} | Weight: {weight}")

#     # Log debug output for the first few pages
#     log_content = f"PDF: {pdf_path}\nFiltered Pages: {[n for n,_ in filtered_pages]}\n"
#     log_content += "\n".join(p for _,p in filtered_pages[:3])
#     log_debug_output(f"{os.path.basename(pdf_path)}.log.txt", log_content)

# if __name__ == "__main__":
#     main()
