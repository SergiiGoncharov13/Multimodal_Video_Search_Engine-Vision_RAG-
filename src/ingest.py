import os
import argparse
import warnings
import gc
import subprocess
from datetime import timedelta

import cv2
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from models import VideoFrame, Base

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")
try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except ImportError:
    pass

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")


def setup_database():
    """Sets up the database connection and creates the necessary tables."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE video_frames ADD COLUMN IF NOT EXISTS modality VARCHAR(50);"))
        conn.execute(text("ALTER TABLE video_frames ALTER COLUMN filepath DROP NOT NULL;"))
        conn.commit()
    return sessionmaker(bind=engine)


def extract_key_frames(video_path: str, output_dir: str, threshold: float = 30.0) -> list[dict]:
    """
    Extracts frames from a video at regular intervals or when significant scene changes occur.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video loaded: {fps:.2f} FPS")

    metadata = []
    frame_index = 0
    prev_gray_frame = None

    min_frames_between_cuts = int(fps * 2)
    last_cut_frame_index = -min_frames_between_cuts

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_sec = frame_index / fps
        time_str = str(timedelta(seconds=int(current_sec)))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray_frame is None:
            prev_gray_frame = gray_frame
            filepath = os.path.join(output_dir, f"frame_{int(current_sec):04d}s.jpg")
            cv2.imwrite(filepath, frame)
            metadata.append({
                "timestamp_sec": int(current_sec),
                "timestamp_formatted": time_str,
                "filepath": filepath,
                "change_score": 0.0,
            })
            last_cut_frame_index = frame_index
            print(f"[{time_str}] Extracted initial frame -> {filepath}")
            frame_index += 1
            continue

        if (frame_index - last_cut_frame_index) < min_frames_between_cuts:
            frame_index += 1
            continue

        diff = cv2.absdiff(prev_gray_frame, gray_frame)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        non_zero_count = np.count_nonzero(thresh)
        total_pixels = thresh.size
        change_percentage = (non_zero_count / total_pixels) * 100

        if change_percentage > threshold:
            filepath = os.path.join(output_dir, f"frame_{int(current_sec):04d}s.jpg")
            cv2.imwrite(filepath, frame)
            metadata.append({
                "timestamp_sec": int(current_sec),
                "timestamp_formatted": time_str,
                "filepath": filepath,
                "change_score": round(change_percentage, 2)
            })
            print(f"[{time_str}] Scene Cut Detected ({change_percentage:.1f}% change) -> {filepath}")

            prev_gray_frame = gray_frame
            last_cut_frame_index = frame_index

        frame_index += 1

    cap.release()
    print(f"Done. Extracted {len(metadata)} unique scene keyframes to {output_dir}")
    return metadata


def caption_frames(metadata: list):
    import mlx.core as mx
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    try:
        from transformers import logging as transformers_logging
        transformers_logging.set_verbosity_error()
    except ImportError:
        pass

    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    print(f"Loading local Vision-Language Model: {model_path} into Unified Memory...")
    model, processor = load(model_path)
    
    prompt = (
        "Describe this scene in high detail. Mention specific objects, "
        "colors, text, lighting, actions, and the general environment."
    )

    total_frames = len(metadata)
    print(f"Captioning {total_frames} frames locally using MLX...")

    for i, item in enumerate(metadata):
        filepath = item["filepath"]
        ts_str = item.get("timestamp_formatted", f"{item['timestamp_sec']}s")
        print(f"[{i+1}/{total_frames}] Captioning {ts_str}...", end=" ", flush=True)

        try:
            image_list = [filepath]
            formatted_prompt = apply_chat_template(
                processor, 
                model.config, 
                prompt, 
                num_images=1
            )
            
            caption = generate(
                model, 
                processor, 
                formatted_prompt, 
                image_list, 
                max_tokens=150, 
                temperature=0.0,
                verbose=False
            )
            
            item["caption"] = caption.text.strip()
            print(f"Done!\n    Preview: {item['caption'][:75]}...")
            
        except Exception as e:
            print(f"Failed! Error: {e}")
            item["caption"] = "Failed to generate caption locally."

        mx.clear_cache()
        gc.collect()

    print("Unloading Vision Model from memory...")
    del model
    del processor
    mx.clear_cache()
    gc.collect()

    return metadata

def transcribe_audio(video_path: str) -> list[dict]:
    """
    Extracts audio from video, transcribes it using Whisper.
    Args:
        video_path: Path to the video file
    Returns:
        List of dictionaries containing audio metadata
    """
    print("Extracting and Transcribing Audio")
    audio_path = video_path.replace(".mp4", ".wav")
    command = [
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
        audio_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError:
        print("Could not extract audio (video might not have an audio track)")
        return []
    
    print("Loading MLX Whisper")
    import mlx_whisper
    result = mlx_whisper.transcribe(
        audio=audio_path,
        path_or_hf_repo="mlx-community/whisper-tiny",
        word_timestamps=True
    )
    audio_metadata = []
    for segment in result.get("segments", []):
        start_sec = int(segment["start"])
        text = segment["text"].strip()
        if not text:
            continue
        time_str = str(timedelta(seconds=start_sec))
        audio_metadata.append({
            "timestamp_sec": start_sec,
            "timestamp_formatted": time_str,
            "filepath": None,
            "caption": f"[Spoken Audio]: {text}",
            "modality": "audio"
        })
        print(f"[{time_str}] Transcribed: {text[:50]}... Done!")
    
    if os.path.exists(audio_path):
        os.remove(audio_path)
    print(f"Extracted {len(audio_metadata)} audio text segments")
    return audio_metadata


def store_data(metadata: list, video_name: str, Session):
    """
    Embeds the captions and stores the metadata and embeddings in the database.
    """
    if not metadata:
        print(f"No metadata to store for '{video_name}'.")
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...", flush=True)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    total = len(metadata)
    print(f"Generating embeddings and storing {total} records in database...", flush=True)

    captions = [item["caption"] for item in metadata]
    vectors = embedder.encode(captions, show_progress_bar=True)

    with Session() as session:
        for item, vector in zip(metadata, vectors):
            record = VideoFrame(
                video_name=video_name,
                timestamp_sec=int(item["timestamp_sec"]),
                timestamp_formatted=item.get("timestamp_formatted", str(timedelta(seconds=int(item["timestamp_sec"])))),
                filepath=item.get("filepath"),
                caption=item["caption"],
                modality=item.get("modality", "visual"),
                text_embedding=vector
            )
            session.add(record)
        session.commit()
    print(f"Successfully stored {total} items for '{video_name}' in database.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract keyframes from a video, caption them with Gemini, and store in pgvector.")
    parser.add_argument("video_path", help="Path to the video file.")
    parser.add_argument("--output_dir", default="data/extracted_frames", help="Directory to save extracted frames.")
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        exit(1)

    print("=" * 60)
    print("Video Ingestion Pipeline")
    print(f"Target video: {args.video_path}")
    print("=" * 60)

    Session = setup_database()

    print("\n[Step 1/4] Extracting scene keyframes...")
    frames = extract_key_frames(args.video_path, args.output_dir)
    print(f"Extracted {len(frames)} frames.")

    print("\n[Step 2/4] Captioning frames and generating vector embeddings...")
    saved_frames = caption_frames(frames)

    for frame in saved_frames:
        frame["modality"] = "visual"

    print("\n[Step 3/4] Processing audio track...")
    audio_segments = transcribe_audio(args.video_path)

    print("\n[Step 4/4] Generating embeddings and storing to database...")
    all_data = saved_frames + audio_segments
    video_name = os.path.basename(args.video_path)
    store_data(all_data, video_name, Session)

    print("\n" + "=" * 60)
    print("Pipeline finished successfully!")
    print("=" * 60)
