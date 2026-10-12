import requests
import os

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from mlx_lm import load
from dotenv import load_dotenv


load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

@st.cache_resource()
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource()
def load_text_llm():
    model, tokenizer = load("mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    return model, tokenizer

embedder = load_embedder()
llm_model, llm_tokenizer = load_text_llm()

API_URL = "http://localhost:8000/ask"

st.title("Multimodal Video Search RAG Engine")
search_query = st.text_input("Ask a question about the video")

if st.button("Search") and search_query:
    with st.spinner("Agent is analyzing the video database..."):
        try:
            response = requests.post(API_URL, json={"question": search_query})
            if response.status_code == 200:
                st.success(response.json()["answer"])
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
