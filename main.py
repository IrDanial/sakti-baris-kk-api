import re
import os
import tempfile
import shutil
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File
from rapidocr_onnxruntime import RapidOCR

# 1. Initialize FastAPI app
app = FastAPI(
    title="Baris KK API",
    description="API Untuk Mengekstrak Jumlah Anggota Keluarga dari Kartu Keluarga",
    version="1.3.0"
)

# 2. Initialize engine globally
engine = RapidOCR()

# 3. Core Image Processing Pipeline
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
                    print(f"Skipping detected Family ID")
                    continue
                
                valid_niks.extend(matches)
                
    match_count = len(valid_niks)
    return match_count, valid_niks

# 4. PDF Processing Pipeline
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
        
        # Create temp file, get the name, and close it immediately
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_img_name = temp_img.name
        temp_img.close() # <-- Releases the Windows lock!
        
        try:
            # PyMuPDF writes to the file
            pix.save(temp_img_name)
            
            # RapidOCR reads the file
            _, niks = extract_nik_pipeline(temp_img_name)
            all_valid_niks.extend(niks)
            
        finally:
            # Clean up the temporary image for this page
            if os.path.exists(temp_img_name):
                os.remove(temp_img_name)
        
    # Close the PDF document
    doc.close()
        
    return len(all_valid_niks), all_valid_niks

# 5. Define the API endpoint for file upload
@app.post("/extract-family-members")
async def extract_family_endpoint(file: UploadFile = File(...)):
    
    # Determine file type based on the uploaded file's content type
    content_type = file.content_type
    is_pdf = content_type == 'application/pdf'
    is_image = content_type.startswith("image/")
    
    if not (is_pdf or is_image):
        raise HTTPException(
            status_code=400, 
            detail=f"Uploaded file must be a valid image or PDF. Got: {content_type}"
        )

    temp_file_path = None
    try:
        # Create temp file, get the name, and close it immediately
        suffix = ".pdf" if is_pdf else ".png"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file_path = temp_file.name
        temp_file.close() # <-- Releases the Windows lock!

        # Write the uploaded file using a standard 'open' block
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Route to the appropriate processing function
        if is_pdf:
            count, nik_strings = process_pdf(temp_file_path)
        else:
            count, nik_strings = extract_nik_pipeline(temp_file_path)

        # Deduplicate NIKs
        unique_niks = list(set(nik_strings))

        return {
            "match_count": len(unique_niks),
            "nik_strings": unique_niks
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Close the uploaded file stream
        file.file.close()
        # Clean up the temporary document on the disk
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)