import re
import os
import tempfile
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from rapidocr_onnxruntime import RapidOCR

# 1. Initialize FastAPI app
app = FastAPI(
    title="Baris KK API",
    description="API Untuk Mengekstrak Jumlah Anggota Keluarga dari Kartu Keluarga",
    version="1.2.0"
)

# 2. Initialize engine globally
engine = RapidOCR()

# 3. Define the expected request payload
class DocumentRequest(BaseModel):
    file_url: HttpUrl

# 4. Core Image Processing Pipeline
def extract_nik_pipeline(img_path):
    """
    Takes an image path as input, runs RapidOCR, and extracts valid 16-digit NIKs.
    Returns a tuple containing the count of matches and the matched strings.
    """
    result, _ = engine(img_path)
    
    valid_niks = []
    nik_pattern = r'\b\d{16}\b'
    
    # RapidOCR returns a list of tuples/lists depending on the version, 
    # typically: [[box, text, confidence], ...]
    if result:
        # Extract just the text from the result
        txts = [res[1] for res in result if len(res) > 1]
        
        for i, text_string in enumerate(txts):
            cleaned_string = text_string.replace(" ", "")
            matches = re.findall(nik_pattern, cleaned_string)
            
            if matches:
                start_index = max(0, i - 2)
                context_window = " ".join(txts[start_index : i+1]).upper()
                
                if "KK" in context_window or "KELUARGA" in context_window:
                    print(f"Skipping detected Family ID (No. KK): {matches[0]}")
                    continue
                
                valid_niks.extend(matches)
                
    match_count = len(valid_niks)
    return match_count, valid_niks

# 5. PDF Processing Pipeline
def process_pdf(pdf_path):
    """
    Opens a PDF, converts each page to an image, and runs the OCR pipeline.
    """
    doc = fitz.open(pdf_path)
    all_valid_niks = []
    
    # Iterate through every page in the PDF
    for page in doc:
        # Render page to an image (dpi=200 is generally good for OCR)
        pix = page.get_pixmap(dpi=200) 
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_img:
            pix.save(temp_img.name)
            
            # Reuse your existing pipeline on the temporary image
            _, niks = extract_nik_pipeline(temp_img.name)
            all_valid_niks.extend(niks)
            
        # Clean up the temporary image for this page
        os.remove(temp_img.name)
        
    return len(all_valid_niks), all_valid_niks

# 6. Define the API endpoint
@app.post("/extract-family-members")
async def extract_family_endpoint(request: DocumentRequest):
    url = str(request.file_url)
    
    # Fetch the file from the URL
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
        # Determine file type based on headers
        content_type = response.headers.get('Content-Type', '')
        is_pdf = content_type == 'application/pdf'
        is_image = content_type.startswith("image/")
        
        if not (is_pdf or is_image):
            raise HTTPException(
                status_code=400, 
                detail=f"URL must point to a valid image or PDF. Got: {content_type}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch file from URL: {str(e)}")

    temp_file_path = None
    try:
        # Save the downloaded file to a temporary file
        suffix = ".pdf" if is_pdf else ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        # Route to the appropriate processing function
        if is_pdf:
            count, nik_strings = process_pdf(temp_file_path)
        else:
            count, nik_strings = extract_nik_pipeline(temp_file_path)

        # Deduplicate NIKs in case the PDF had overlapping scans or multiple identical pages
        unique_niks = list(set(nik_strings))

        return {
            "match_count": len(unique_niks),
            "nik_strings": unique_niks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up the downloaded document
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)