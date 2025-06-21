import fitz
import pytesseract
from PIL import Image
from fuzzywuzzy import fuzz
import json
import warnings

# Suppress warnings from fuzzywuzzy about missing Levenshtein (using difflib fallback)
warnings.filterwarnings("ignore", category=UserWarning, module='fuzzywuzzy')

# Load SAP reference data
with open("data/SAP_data.json", "r") as f:
    reference_data = json.load(f)

# Preprocess reference entries for matching
entries = []
for entry in reference_data:
    # Combine vendor name fields
    vendor_name = (entry.get("Vendor - Name 1") or "")
    if entry.get("Vendor - Name 2"):
        vendor_name += " " + entry["Vendor - Name 2"]
    vendor_name = vendor_name.strip().lower()
    entries.append({
        "MBLNR": entry["MBLNR"],
        "MJAHR": entry["MJAHR"],
        "vendor": vendor_name,
        "dn": str(entry["Delivery Note Number"]).lower() if entry.get("Delivery Note Number") else "",
        "po": str(entry["Purchase Order Number"]).lower() if entry.get("Purchase Order Number") else ""
    })

# Open the merged PDF batch
doc = fitz.open("solution/superbatch_merged.pdf")
total_pages = doc.page_count

output = []       # list of output records
log_lines = []    # accumulated logs for debugging

page_index = 0
while page_index < total_pages:
    page_number = page_index + 1  # 1-based page number in output
    # Extract text from page (use embedded text if available, OCR if not)
    page = doc.load_page(page_index)
    text = page.get_text("text")
    if not text:  # If no text extracted, perform OCR
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        text = pytesseract.image_to_string(img, lang="eng")
    text_lower = text.lower()

    # Fuzzy match this page text against all reference entries
    best_entry = None
    best_score_sum = 0
    best_vendor_score = best_dn_score = best_po_score = 0
    second_best_sum = 0

    for entry in entries:
        # Compute partial fuzzy ratios for vendor, delivery note, and PO
        vs = fuzz.partial_ratio(entry["vendor"], text_lower) if entry["vendor"] else 0
        ds = fuzz.partial_ratio(entry["dn"], text_lower) if entry["dn"] else 0
        ps = fuzz.partial_ratio(entry["po"], text_lower) if entry["po"] else 0
        # Use the higher of delivery note vs. purchase order in combination with vendor
        combined_score = vs + (ps if ps > ds else ds)
        if combined_score > best_score_sum:
            second_best_sum = best_score_sum
            best_score_sum = combined_score
            best_entry = entry
            best_vendor_score, best_dn_score, best_po_score = vs, ds, ps
        elif combined_score > second_best_sum:
            second_best_sum = combined_score

    # Decide if this page confidently matches a reference entry
    is_confident_match = False
    if best_entry:
        # Primary confidence conditions: high vendor & DN match, or one of them nearly exact
        if best_vendor_score >= 70 and best_dn_score >= 70:
            is_confident_match = True
        if best_vendor_score >= 70 and best_dn_score == 0 and best_po_score >= 70:
            # Vendor and PO match (no DN found, possibly on packing slips)
            is_confident_match = True
        if best_dn_score >= 90:
            # Delivery note appears almost exactly (even if vendor is low due to OCR issues)
            is_confident_match = True

    if best_entry and is_confident_match:
        # Log the match decision
        log_lines.append(f"Page {page_number}: Matched MBLNR {best_entry['MBLNR']} "
                         f"(vendor_score={best_vendor_score}, dn_score={best_dn_score}, po_score={best_po_score}).")
        output.append({
            "Page of batch where document starts": page_number,
            "MBLNR": best_entry["MBLNR"],
            "MJAHR": best_entry["MJAHR"]
        })

        # Determine document continuation: assume current entry continues until a new doc is detected
        current_doc_id = best_entry["MBLNR"]
        page_index += 1
        while page_index < total_pages:
            next_page = doc.load_page(page_index)
            next_text = next_page.get_text("text")
            if not next_text:
                # OCR for next page if needed
                pix = next_page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                next_text = pytesseract.image_to_string(img, lang="eng")
            next_text_lower = next_text.lower()

            # Fuzzy match next page to identify if a new doc starts here
            next_best_entry = None
            next_best_score = 0
            next_vs = next_ds = next_ps = 0
            for entry in entries:
                vs2 = fuzz.partial_ratio(entry["vendor"], next_text_lower) if entry["vendor"] else 0
                ds2 = fuzz.partial_ratio(entry["dn"], next_text_lower) if entry["dn"] else 0
                ps2 = fuzz.partial_ratio(entry["po"], next_text_lower) if entry["po"] else 0
                score2 = vs2 + (ps2 if ps2 > ds2 else ds2)
                if score2 > next_best_score:
                    next_best_score = score2
                    next_best_entry = entry
                    next_vs, next_ds, next_ps = vs2, ds2, ps2

            # Check if this next page is the start of a different document
            if next_best_entry:
                if next_best_entry["MBLNR"] != current_doc_id:
                    # Different entry content detected
                    if next_vs >= 85:  # a strong new vendor header
                        break
                    if next_vs >= 60 and next_ds >= 60:  # moderate new vendor and DN
                        break
                    if next_vs >= 60 and next_ds == 0 and next_ps >= 70:  # moderate vendor and strong PO (no DN)
                        break
                    if next_ds >= 90:  # a clear new delivery note number
                        break
            # If no break conditions met, treat as continuation of current document
            log_lines.append(f"Page {page_index+1}: Continuation of page {page_number} (same document).")
            page_index += 1

        # continue to next iteration without incrementing page_index here (it's already moved)
        continue

    else:
        # No confident match found – mark this page as an unknown document start
        if best_entry:
            log_lines.append(f"Page {page_number}: No confident match (best guess MBLNR {best_entry['MBLNR']} "
                             f"with vendor_score={best_vendor_score}, dn_score={best_dn_score}). Marking as unknown.")
        else:
            log_lines.append(f"Page {page_number}: No match found. Marking as unknown.")
        output.append({
            "Page of batch where document starts": page_number,
            "MBLNR": -1,
            "MJAHR": -1
        })
        page_index += 1

# Save output JSON
with open("output.json", "w") as out_f:
    json.dump(output, out_f, indent=2)

# (Optional) print log lines for debugging
for line in log_lines:
    print(line)
