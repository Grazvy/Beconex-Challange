import os
import json
from PyPDF2 import PdfReader
from datetime import datetime

def get_pdf_page_count(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return len(reader.pages)
    except Exception as e:
        print(f"Fehler beim Lesen von {pdf_path}: {e}")
        return 0

def load_sap_data(sap_json_path):
    with open(sap_json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_solution(batch_folder_path, sap_json_path="data/SAP_data.json"):
    sap_data = load_sap_data(sap_json_path)
    used_entries = set()
    output_json = []
    current_page = 1

    pdf_files = sorted([
        f for f in os.listdir(batch_folder_path) 
        if f.lower().endswith(".pdf")
    ])

    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(batch_folder_path, pdf_file)
        num_pages = get_pdf_page_count(full_pdf_path)

        filename = os.path.basename(full_pdf_path).lower()
        try:
            date_from_filename = filename.split("_")[0]
            date_obj = datetime.strptime(date_from_filename, "%Y%m%d")
            date_key = date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"⚠️ Fehler beim Extrahieren des Datums aus {filename}: {e}")
            continue

        sap_entry = None
        for idx, entry in enumerate(sap_data):
            if idx in used_entries:
                continue
            delivery_date_raw = entry.get("Delivery Note Date", "")
            try:
                delivery_date = datetime.fromisoformat(delivery_date_raw.replace("Z", "")).strftime("%Y-%m-%d")
            except Exception:
                continue

            if delivery_date == date_key:
                used_entries.add(idx)
                sap_entry = {
                    "MBLNR": entry.get("MBLNR"),
                    "MJAHR": entry.get("MJAHR")
                }
                break

        if sap_entry:
            output_json.append({
                "Page of batch where document starts": current_page,
                "MBLNR": sap_entry["MBLNR"],
                "MJAHR": sap_entry["MJAHR"]
            })
        else:
            print(f"⚠️ Keine Zuordnung für: {pdf_file}")
            output_json.append({
                "Page of batch where document starts": current_page,
                "MBLNR": -1, 
                "MJAHR": -1 
            })

        current_page += num_pages

    return output_json
