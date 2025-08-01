from dotenv import load_dotenv

load_dotenv()

from mistralai import Mistral
import os

api_key = os.getenv('MISTRAL_OCR_API_KEY')
client = Mistral(api_key=api_key)

def get_ocr_response(doc_url):

    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": doc_url
        },
        include_image_base64=True
    )
    markdown_text = ocr_response.pages[0].markdown
    return markdown_text
