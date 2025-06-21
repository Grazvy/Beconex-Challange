import os
import json
from PyPDF2 import PdfMerger, PdfReader
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



used_entries = set()  # global oder übergebbar

def find_sap_entry(pdf_path, sap_data):
    filename = os.path.basename(pdf_path).lower()

    try:
        date_from_filename = filename.split("_")[0]
        date_obj = datetime.strptime(date_from_filename, "%Y%m%d")
        date_key = date_obj.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"⚠️ Fehler beim Extrahieren des Datums aus {filename}: {e}")
        return None

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
            return {
                "MBLNR": entry.get("MBLNR"),
                "MJAHR": entry.get("MJAHR")
            }

    print(f"⚠️ Keine freie SAP-Zeile mit Datum {date_key} für Datei: {filename}")
    return None



def process_batch_folder(batch_folder_path, sap_json_path, output_pdf_path, output_json_path):
    sap_data = load_sap_data(sap_json_path)
    merger = PdfMerger()
    output_json = []

    current_page = 1
    pdf_files = sorted([
        f for f in os.listdir(batch_folder_path) 
        if f.lower().endswith(".pdf")
    ])

    for pdf_file in pdf_files:
        full_pdf_path = os.path.join(batch_folder_path, pdf_file)
        num_pages = get_pdf_page_count(full_pdf_path)

        sap_entry = find_sap_entry(full_pdf_path, sap_data)
        if sap_entry:
            output_json.append({
                "Page of batch where document starts": current_page,
                "MBLNR": sap_entry["MBLNR"],
                "MJAHR": sap_entry["MJAHR"]
            })
        else:
            print(f"⚠️ Keine Zuordnung für: {pdf_file}")

        merger.append(full_pdf_path)
        current_page += num_pages

    # Gesamt-PDF speichern
    merger.write(output_pdf_path)
    merger.close()

    # Output JSON speichern
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    print(f"✅ Zusammengefügt: {output_pdf_path}")
    print(f"✅ Output JSON gespeichert: {output_json_path}")


if __name__ == "__main__":
    # Beispielkonfiguration
    batch_dir = "data/batch_7_2023_2024"
    sap_json = "data/SAP_data.json"
    
    # Extract batch_id from batch_dir
    batch_id = os.path.basename(batch_dir)

    output_dir = "solution"
    os.makedirs(output_dir, exist_ok=True)

    merged_pdf = os.path.join(output_dir, f"{batch_id}_merged.pdf")
    output_json = os.path.join(output_dir, f"{batch_id}_output.json")

    process_batch_folder(batch_dir, sap_json, merged_pdf, output_json)
