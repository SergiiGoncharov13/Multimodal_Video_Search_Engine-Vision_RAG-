import os

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from mlx_lm import load, generate
from dotenv import load_dotenv

from ingest import VideoFrame

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"

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

st.title("Multimodal Video Search RAG Engine")

search_query = st.text_input("Ask a question about the video")

if st.button("Search") and search_query:
    query_vector = embedder.encode(search_query)
    with SessionLocal() as session:
        result = (
            session.query(VideoFrame)
            .order_by(VideoFrame.text_embedding.cosine_distance(query_vector)).first()
        )
    if result:
        st.success(f"Best match found at {result.timestamp_sec}s in {result.video_name}")

        with st.spinner("Reading the frame and writing answer..."):
            messages = [
                {"role": "system", "content": "Answer the user's question using ONLY the provided video frame description"},
                {"role": "user", "content": f"Video description: {result.caption}\n\nQuestion: {search_query}"}
            ]
            prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            answer = generate(llm_model, llm_tokenizer, prompt, max_tokens=150, verbose=False)
        st.success(answer.strip())
        st.caption(f"**AI Vision Log:** {result.caption}")

        with st.expander("See Raw Database Match"):
            st.write(f"**Matched Timestamp** {result.timestamp_sec}s")
            st.info(f"**Vision Model's Original Log** {result.caption}")
    else:
        st.warning("No results found.")
