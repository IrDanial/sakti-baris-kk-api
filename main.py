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
    description="API Untuk Mengekstrak Nama dan NIK Anggota Keluarga dari Kartu Keluarga",
    version="1.4.0"
)

# 2. Initialize engine globally
engine = RapidOCR()

# 3. Define the expected request payload (Supabase URL)
class DocumentRequest(BaseModel):
    file_url: HttpUrl

# 4. Core Image Processing Pipeline with Geometry Matching
def extract_kk_pipeline(img_path):
    """
    Takes an image path as input, runs RapidOCR, extracts NIKs, 
    and uses Bounding Box Geometry to pair NIKs with Names.
    """
    result, _ = engine(img_path)
    
    fam_id = None
    nik_data = []
    people = []
    
    nik_pattern = r'\b\d{16}\b'
    
    # If no text is found, return empty structure
    if not result:
        return [], []

    # ==========================================
    # PASS 1: Find Family ID and NIK Geometries
    # ==========================================
    # result format: [[ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], "Text", confidence ], ...]
    for i, res in enumerate(result):
        box = res[0]
        text_string = res[1]
        
        cleaned_string = text_string.replace(" ", "")
        matches = re.findall(nik_pattern, cleaned_string)
        
        if matches:
            # Context Window to check for Family ID (No. KK)
            start_index = max(0, i - 2)
            context_window = " ".join([r[1] for r in result[start_index : i+1]]).upper()
            
            if "KK" in context_window or "KELUARGA" in context_window:
                if not fam_id:
                    fam_id = matches[0]
                continue
            
            # Calculate the vertical center (Y) and left edge (X) of the NIK
            # box format: [top-left, top-right, bottom-right, bottom-left]
            y_center = sum(pt[1] for pt in box) / 4
            x_left = box[0][0]
            height = box[3][1] - box[0][1] # height of the text box
            
            nik_data.append({
                "nik": matches[0],
                "y_center": y_center,
                "x_left": x_left,
                "tolerance": height * 0.6  # 60% of text height tolerance
            })

    # ==========================================
    # PASS 2: Match Names to NIK Rows
    # ==========================================
    for person in nik_data:
        name_fragments = []
        
        for j, res in enumerate(result):
            box = res[0]
            text_string = res[1]
            
            cand_y_center = sum(pt[1] for pt in box) / 4
            cand_x_right = box[1][0] # The right edge of the candidate text
            
            # 1. Is this text on the same horizontal row as the NIK?
            if abs(cand_y_center - person["y_center"]) < person["tolerance"]:
                # 2. Is this text to the LEFT of the NIK?
                if cand_x_right < person["x_left"]:
                    # 3. Does it contain letters? (Ignores numbers like "1", "2")
                    if re.search(r'[A-Za-z]', text_string):
                        name_fragments.append((box[0][0], text_string))
        
        # Sort words left-to-right
        name_fragments.sort(key=lambda x: x[0])
        
        # Join the words together to form the full name
        full_name = " ".join(" ".join([frag[1] for frag in name_fragments]).split())    
        
        people.append({
            "name": full_name.upper() if full_name else "",
            "nik": person["nik"]
        })
        
    # Extract just the lists to return
    nik_strings = [p["nik"] for p in people]
    name_strings = [p["name"] for p in people]
    
    return nik_strings, name_strings

# 5. PDF Processing Pipeline
def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    all_niks = []
    all_names = []
    
    for page in doc:
        pix = page.get_pixmap(dpi=200) 
        
        # Windows-safe temporary file handling
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_img_name = temp_img.name
        temp_img.close() # Release lock immediately
        
        try:
            pix.save(temp_img_name)
            niks, names = extract_kk_pipeline(temp_img_name)
            all_niks.extend(niks)
            all_names.extend(names)
        finally:
            if os.path.exists(temp_img_name):
                os.remove(temp_img_name)
                
    doc.close()
    return all_niks, all_names

# 6. Define the API endpoint
@app.post("/extract-family-members")
async def extract_family_endpoint(request: DocumentRequest):
    url = str(request.file_url)
    
    # Fetch file from Supabase URL
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        
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
        # Windows-safe temporary file handling for the downloaded document
        suffix = ".pdf" if is_pdf else ".png"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file_path = temp_file.name
        temp_file.close() # Release the lock!

        # Save downloaded chunks to the released temp file
        with open(temp_file_path, "wb") as buffer:
            for chunk in response.iter_content(chunk_size=8192):
                buffer.write(chunk)

        # Route to appropriate processing function
        if is_pdf:
            nik_strings, name_strings = process_pdf(temp_file_path)
        else:
            nik_strings, name_strings = extract_kk_pipeline(temp_file_path)

        # Return the strictly formatted JSON
        return {
            "match_count": len(nik_strings),
            "nik_strings": nik_strings,
            "name_count": len(name_strings),
            "name_strings": name_strings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up the main downloaded document
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)