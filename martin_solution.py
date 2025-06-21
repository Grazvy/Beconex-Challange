import fitz
import pytesseract
from PIL import Image
from rapidfuzz import fuzz
import json
import os

# Configuration
PDF_PATH = "solution/superbatch_merged.pdf"
REFERENCE_JSON = "data/SAP_data.json"
OUTPUT_JSON = "output.json"
# Minimum scores for confident matches
VENDOR_THRESH = 70
DN_PO_THRESH = 70
DN_EXACT_THRESH = 90

def extract_text(page):
    """Extract embedded text, fallback to OCR if empty."""
    txt = page.get_text().strip()
    if not txt:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        txt = pytesseract.image_to_string(img)
    return txt.lower()

def load_references():
    refs = []
    with open(REFERENCE_JSON, encoding="utf-8") as f:
        for e in json.load(f):
            vendor = (e.get("Vendor - Name 1") or "").strip()
            if e.get("Vendor - Name 2"):
                vendor += " " + e["Vendor - Name 2"]
            refs.append({
                "MBLNR": e["MBLNR"],
                "MJAHR": e["MJAHR"],
                "vendor": vendor.lower(),
                "dn": str(e.get("Delivery Note Number", "")).lower(),
                "po": str(e.get("Purchase Order Number", "")).lower()
            })
    return refs

def find_candidate(page_text, refs):
    """Return the best matching ref entry or None."""
    best = None
    best_score = 0
    best_vs = best_ds = best_ps = 0

    for r in refs:
        vs = fuzz.partial_ratio(r["vendor"], page_text) if r["vendor"] else 0
        ds = fuzz.partial_ratio(r["dn"], page_text) if r["dn"] else 0
        ps = fuzz.partial_ratio(r["po"], page_text) if r["po"] else 0
        # use the stronger of DN vs PO
        primary = ds if ds > ps else ps
        score = vs + primary
        if score > best_score:
            best_score, best, best_vs, best_ds, best_ps = score, r, vs, ds, ps

    if not best:
        return None

    # Confidence rules
    if (best_vs >= VENDOR_THRESH and best_ds >= DN_PO_THRESH) \
       or (best_vs >= VENDOR_THRESH and best_ps >= DN_PO_THRESH) \
       or (best_ds >= DN_EXACT_THRESH) \
       or (best_ps >= DN_EXACT_THRESH):
        return {
            "MBLNR": best["MBLNR"],
            "MJAHR": best["MJAHR"],
            "vendor_score": best_vs,
            "dn_score": best_ds,
            "po_score": best_ps
        }
    return None

def main():
    refs = load_references()
    doc = fitz.open(PDF_PATH)
    total_pages = doc.page_count

    # Step 1: find all pages that start a (known) document
    matches = []
    for i in range(total_pages):
        text = extract_text(doc.load_page(i))
        cand = find_candidate(text, refs)
        if cand:
            matches.append((i+1, cand["MBLNR"], cand["MJAHR"]))

    # Step 2: remove duplicates (keep first occurrence of each MBLNR)
    seen = set()
    filtered = []
    for page, mblnr, mjahr in matches:
        if mblnr not in seen:
            seen.add(mblnr)
            filtered.append((page, mblnr, mjahr))

    # Step 3: build the final list with placeholders for missing starts
    #    insert any pages between starts as unknown docs
    filtered.sort(key=lambda x: x[0])
    final = []
    for idx, (pg, mblnr, mjahr) in enumerate(filtered):
        final.append({"Page of batch where document starts": pg,
                      "MBLNR": mblnr, "MJAHR": mjahr})
        # compute gap to next known start
        if idx + 1 < len(filtered):
            next_pg = filtered[idx+1][0]
            # any pages in between are unknown docs
            for gap in range(pg+1, next_pg):
                final.append({"Page of batch where document starts": gap,
                              "MBLNR": -1, "MJAHR": -1})

    # Step 4: ensure pages 1 and last page are covered
    #    (in case the very first doc isn't on page 1, or last doc < last page)
    pages_covered = {e["Page of batch where document starts"] for e in final}
    # page 1
    if 1 not in pages_covered:
        final.insert(0, {"Page of batch where document starts": 1, "MBLNR": -1, "MJAHR": -1})
    # last page
    if filtered:
        last_defined = filtered[-1][0]
        for gap in range(last_defined+1, total_pages+1):
            final.append({"Page of batch where document starts": gap,
                          "MBLNR": -1, "MJAHR": -1})

    # Step 5: sort final by page and dedupe again just in case
    final.sort(key=lambda x: x["Page of batch where document starts"])

    # Finally, write to output.json
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    # Sanity check: compare to provided solution
    if os.path.exists("superbatch_output.json"):
        sol = json.load(open("superbatch_output.json", encoding="utf-8"))
        print("Test matches solution:",
              "True" if sol == final else "False")

if __name__ == "__main__":
    main()
