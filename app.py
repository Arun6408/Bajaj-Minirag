from flask import Flask, request, jsonify
from mistral_ocr import get_ocr_response

from rag import rag_v1
import os


app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "bajaj-rag"}), 200

@app.route('/hackrx/run', methods=['POST'])
def hackrx_run():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()

    documents = data.get("documents")
    questions = data.get("questions")

    if not documents or not questions:
        return jsonify({"error": "Missing documents or questions"}), 400
    doc_text = get_ocr_response(documents)

    # Example processing (replace with real logic)
    answers = rag_v1(doc_text,questions)

    return jsonify({
        "answers": answers
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
