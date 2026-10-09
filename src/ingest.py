import os
import argparse
from datetime import timedelta

import cv2
import mlx.core as mlx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from sqlalchemy import Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
POSTGRES_DB = os.getenv("POSTGRES_DB", "video_rag")
DATABASE_URL = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"

Base = declarative_base()


class VideoFrame(Base):
    __tablename__ = "video_frames"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_name: Mapped[str] = mapped_column(String(255), nullable=False)
    timestamp_sec: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp_formatted: Mapped[str] = mapped_column(String(50), nullable=False)
    filepath: Mapped[str] = mapped_column(String(255), nullable=False)
    caption: Mapped[str] = mapped_column(Text, nullable=True)
    text_embedding: Mapped[Vector] = mapped_column(Vector(384))


def setup_database():
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def extract_frames(video_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    metadata = []
    target_sec = 0

    while True:
        if not cap.grab(): break
        if cap.get(cv2.CAP_PROP_POS_MSEC) >= target_sec * 1000:
            succes, frame = cap.retrieve()
            if succes:
                filepath = os.path.join(output_dir, f"frame_{target_sec:04d}s.jpg")
                cv2.imwrite(filepath, frame)
                metadata.append({
                    "timestamp_sec": target_sec,
                    "timestamp_formatted": str(timedelta(seconds=target_sec)),
                    "filepath": filepath
                })
                target_sec += 1
    cap.release()
    return metadata
    
def caption_frames(metadata: list):
    model, processor = load("mlx-community/Qwen2-VL-2B-Instruct-4bit")
    prompt = apply_chat_template(processor, model.config, "Describe the scene in detail", num_images=1)

    for item in metadata:
        caption = generate(model, processor, prompt, [item["filepath"]], max_tokens=150, temperature=0.0)
        item["caption"] = caption.text.strip()
        print(f"[{item['timestamp_sec']}s]: {item['caption']}")

    del model
    del processor
    mlx.metal.clear_cache()
    return metadata

def store_data(metadata: list, video_name: str, Session):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with Session() as session:
        for item in metadata:
            vector = embedder.encode(item['caption'])
            record = VideoFrame(
                video_name=video_name,
                timestamp_sec=item["timestamp_sec"],
                timestamp_formatted=item.get("timestamp_formatted", str(timedelta(seconds=item["timestamp_sec"]))),
                filepath=item["filepath"],
                caption=item["caption"],
                text_embedding=vector
            )
            session.add(record)
        session.commit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/extracted_frames", help="Directory to save extracted frames.")
    args = parser.parse_args()

    Session = setup_database()
    print("Extracting frames..")
    frames = extract_frames(args.video_path, args.output_dir)
    print(f"Extracted {len(frames)} frames.")

    print("Captioning frames..")
    captioned_frames = caption_frames(frames)

    print("Storing data in database..")
    video_name = os.path.basename(args.video_path)
    store_data(captioned_frames, video_name, Session)
