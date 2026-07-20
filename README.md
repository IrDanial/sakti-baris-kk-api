# NIK Extractor API

A FastAPI service that accepts a Supabase signed image URL, downloads the image, and uses RapidOCR to extract 16-digit Indonesian NIKs (filtering out Kartu Keluarga numbers).

## Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the server: `uvicorn main:app --reload`

## Usage
Send a POST request to `/extract-nik` with the image URL:
```json
{
  "image_url": "[https://your-supabase-url.com/.../image.png?token=](https://your-supabase-url.com/.../image.png?token=)..."
}