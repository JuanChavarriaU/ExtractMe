from flask import Flask, request, send_file, jsonify
import zipfile
import os
import tempfile
import secrets
import shutil
import io
import pypdfium2 as pdfium
import tableExtraction as te
import re
ALLOWED_EXTENSIONS = ("pdf")

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

def allowed_file(filename):
    """Verifies if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return "Welcome to the Flask API. Use /upload to upload files."

#note: debe estar dentro de un ciclo para iterar el numero de pagina
def process_pdf_page(pdf: pdfium.PdfDocument, page_num: int, temp_dir: str):
    """Processes a single PDF page: converts it to an image, detects tables, and saves extracted CSVs."""
    page = pdf[page_num]
    image = page.render(scale=4).to_pil()  # Convert to high-resolution image

    # Detect tables
    table = te.detect_and_crop_table(image)

    if not table:
            print(f"No tables found on page {page_num}, skipping.")
            return None  # ✅ Correctly return None instead of an empty list


    if table is None:
        return None
    
    _, df, _ = te.process_pdf(table)
    if df is None:
        return None
    # Save each extracted table as CSV
    csv_path = os.path.join(temp_dir, f"table_{page_num}.csv")
    df.to_csv(csv_path, index=False)

    return csv_path

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    temp_dir = tempfile.mkdtemp()
    status_messages = []
    extracted_tables = []

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(file.read()))

        for page_num in range(len(pdf)):
            csv_path = process_pdf_page(pdf, page_num, temp_dir)

            if csv_path is None:
                status_messages.append(f"⚠️ No tables found on page {page_num}, skipping.")
                continue
            
            extracted_tables.append(csv_path)
            status_messages.append(f"✅ Tables found on page {page_num} ({len(extracted_tables)} tables extracted).")

        if not extracted_tables:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({"error": "No tables found in the document."}), 200

        zip_path = os.path.join(temp_dir, "extracted_data.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for csv_file in extracted_tables:
                zf.write(csv_file, os.path.basename(csv_file))

        response = send_file(zip_path, as_attachment=True, mimetype="application/zip")

        return response
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True) 


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
