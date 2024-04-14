from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
import nltk
import json
import os

qwen_model = None
qwen_tokenizer = None

def load_qwen_model_and_tokenizer():
    global qwen_model, qwen_tokenizer
    if qwen_model is None or qwen_tokenizer is None:
        device = "cuda"  # the device to load the model onto
        model_name = "Qwen/Qwen1.5-1.8B-Chat"
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
        )
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)

    return qwen_model, qwen_tokenizer



