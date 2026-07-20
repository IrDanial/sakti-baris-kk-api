import re
import os
import tempfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from rapidocr import RapidOCR

# 1. Initialize FastAPI app
app = FastAPI(title="Baris KK API")

# 2. Initialize engine globally
engine = RapidOCR()

# 3. Define the expected request payload
class ImageRequest(BaseModel):
    image_url: HttpUrl

def extract_nik_pipeline(img_path):
    """
    Takes an image path as input, runs RapidOCR, and extracts valid 16-digit NIKs.
    Returns a tuple containing the count of matches and the matched strings.
    """
    result = engine(img_path)
    
    valid_niks = []
    nik_pattern = r'\b\d{16}\b'
    
    if hasattr(result, 'txts') and result.txts:
        for i, text_string in enumerate(result.txts):
            cleaned_string = text_string.replace(" ", "")
            matches = re.findall(nik_pattern, cleaned_string)
            
            if matches:
                start_index = max(0, i - 2)
                context_window = " ".join(result.txts[start_index : i+1]).upper()
                
                if "KK" in context_window or "KELUARGA" in context_window:
                    print(f"Skipping detected Family ID (No. KK): {matches[0]}")
                    continue
                
                valid_niks.extend(matches)
                
    match_count = len(valid_niks)
    return match_count, valid_niks

# 4. Define the API endpoint
@app.post("/extract-family-members")
async def extract_family_endpoint(request: ImageRequest):
    url = str(request.image_url)
    
    # Fetch the image from the Supabase signed URL
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        # Basic validation to ensure the URL points to an image
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="URL does not point to a valid image.")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {str(e)}")

    temp_image_path = None
    try:
        # Save the downloaded image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_image_path = temp_file.name

        # Process the temporary file
        count, nik_strings = extract_nik_pipeline(temp_image_path)

        return {
            "match_count": count,
            "nik_strings": nik_strings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up the temporary file
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)