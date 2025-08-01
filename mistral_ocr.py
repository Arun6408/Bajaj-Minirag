from dotenv import load_dotenv
import base64
import tempfile
import requests
from PyPDF2 import PdfReader, PdfWriter
from mistralai import Mistral
import os

load_dotenv()
api_key = os.getenv('MISTRAL_OCR_API_KEY')
client = Mistral(api_key=api_key)

def download_pdf(doc_url):
    response = requests.get(doc_url)
    if response.status_code == 200:
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(response.content)
        temp_pdf.close()
        # print('download complete')
        return temp_pdf.name
    else:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

def pdf_to_data_url(pdf_path):
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
    return f"data:application/pdf;base64,{pdf_base64}"

def split_pdf(pdf_path, chunk_size=50):
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    chunks = []

    for start in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        for i in range(start, min(start + chunk_size, total_pages)):
            writer.add_page(reader.pages[i])

        temp_chunk = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        with open(temp_chunk.name, "wb") as f:
            writer.write(f)
        chunks.append(temp_chunk.name)

    return chunks

def get_ocr_response(doc_url):
    local_pdf_path = download_pdf(doc_url)
    reader = PdfReader(local_pdf_path)

    if len(reader.pages) <= 50:
        data_url = pdf_to_data_url(local_pdf_path)
        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": data_url},
            include_image_base64=True
        )
        return "\n".join([page.markdown for page in response.pages])
    else:
        print('file is more than 50 pages')
        chunks = split_pdf(local_pdf_path, chunk_size=50)
        full_markdown = ""

        for chunk_path in chunks:
            data_url = pdf_to_data_url(chunk_path)
            response = client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": data_url},
                include_image_base64=True
            )
            full_markdown += "\n".join([page.markdown for page in response.pages]) + "\n"
        return full_markdown
