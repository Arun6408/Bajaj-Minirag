import sys
import os

from dotenv import load_dotenv
import tempfile


load_dotenv()

from huggingface_hub import login
login(os.getenv("hugging_face_token"))



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from minirag import MiniRAG, QueryParam
from minirag.llm import (
    gpt_4o_mini_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    temp_dir = tempfile.gettempdir()

    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="PHI")
    parser.add_argument("--workingdir", type=str, default=temp_dir)
    parser.add_argument("--datapath", type=str, default="./dataset/")
    args = parser.parse_args()
    return args


args = get_args()

def rag_v1(doc_text,questions):

    if args.model == "PHI":
        LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
    elif args.model == "GLM":
        LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
    elif args.model == "MiniCPM":
        LLM_MODEL = "openbmb/MiniCPM3-4B"
    elif args.model == "qwen":
        LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
    else:
        print("Invalid model name")
        exit(1)

    WORKING_DIR = args.workingdir
    DATA_PATH = args.datapath
    print("USING LLM:", LLM_MODEL)
    print("USING WORKING DIR:", WORKING_DIR)


    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    rag = MiniRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        llm_model_max_token_size=200,
        llm_model_name=LLM_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=1000,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
                embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
            ),
        ),
    )



    rag.insert(doc_text)
    answers =[]
    for q in questions:
        answers.append(rag.query(q, param=QueryParam(mode="mini")).replace("\n", "").replace("\r", ""))
    print(answers)
    return answers
