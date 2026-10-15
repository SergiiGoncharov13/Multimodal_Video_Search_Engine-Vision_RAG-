import os
import sys
import warnings
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from dotenv import load_dotenv, find_dotenv

from agent import VideoAnalysisAgent
from ingest import setup_database, extract_key_frames, caption_frames, transcribe_audio, store_data

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Kwargs passed to.*")
try:
    from transformers import logging as transformers_logging
    transformers_logging.set_verbosity_error()
except ImportError:
    pass

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
load_dotenv(find_dotenv(), override=True)

video_agent = None

BASE_DIR = Path(__file__).resolve().parent.parent

candidate_video_dirs = [
    BASE_DIR / "data" / "videos",
    Path(__file__).resolve().parent / "data" / "videos"
]
VIDEO_DIR = next((d for d in candidate_video_dirs if d.exists()), candidate_video_dirs[0])
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

candidate_frame_dirs = [
    BASE_DIR / "data" / "extracted_frames",
    Path(__file__).resolve().parent / "data" / "extracted_frames"
]
FRAMES_DIR = next((d for d in candidate_frame_dirs if d.exists()), candidate_frame_dirs[0])
FRAMES_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_agent
    load_dotenv(find_dotenv(), override=True)
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Warning: DATABASE_URL is not set.")
    print("Initializing VideoAnalysisAgent...")
    video_agent = VideoAnalysisAgent(db_url, video_dir=str(VIDEO_DIR))
    print("VideoAnalysisAgent ready for queries!")
    yield


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Multimodal Video Search Engine API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")

ingestion_tasks = {}


def run_ingestion_pipeline(video_path: str, video_name: str):
    try:
        ingestion_tasks[video_name] = {"status": "processing", "step": "Connecting to database..."}
        Session = setup_database()
        print(f"{video_name} ingestion started")

        ingestion_tasks[video_name] = {"status": "processing", "step": "Extracting scene keyframes..."}
        frames = extract_key_frames(video_path, str(FRAMES_DIR))

        ingestion_tasks[video_name] = {"status": "processing", "step": f"Captioning {len(frames)} frames with MLX vision model..."}
        saved_frames = caption_frames(frames)
        for frame in saved_frames:
            frame["modality"] = "visual"
        
        ingestion_tasks[video_name] = {"status": "processing", "step": "Transcribing audio track with Whisper..."}
        audio_segments = transcribe_audio(video_path)

        ingestion_tasks[video_name] = {"status": "processing", "step": "Generating embeddings and indexing in database..."}
        all_data = saved_frames + audio_segments
        store_data(all_data, video_name, Session)

        ingestion_tasks[video_name] = {"status": "completed", "step": "Ingestion completed successfully"}
        print(f"{video_name} ingestion completed")
    except Exception as e:
        print(f"{video_name} ingestion failed: {e}")
        ingestion_tasks[video_name] = {"status": "failed", "step": "Failed", "error": str(e)}


@app.get("/")
async def root():
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        return {"Error": "index.html not found"}
    return FileResponse(html_path)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent_ready": video_agent is not None,
        "video_dir": str(VIDEO_DIR),
        "video_files": [f.name for f in VIDEO_DIR.glob("*") if f.is_file() and not f.name.startswith(".")]
    }


@app.get("/api/videos")
async def get_videos():
    disk_videos = [f.name for f in VIDEO_DIR.glob("*") if f.is_file() and not f.name.startswith(".")]
    db_videos = []
    if video_agent and video_agent.SessionLocal:
        try:
            with video_agent.SessionLocal() as session:
                rows = session.execute(text("SELECT DISTINCT video_name FROM video_frames ORDER BY video_name")).fetchall()
                db_videos = [r[0] for r in rows if r[0]]
        except Exception as e:
            print(f"Error reading video names from database: {e}")
    return {
        "disk_videos": sorted(disk_videos),
        "db_videos": sorted(db_videos)
    }


@app.get("/api/ingest/status")
async def get_ingest_status():
    return ingestion_tasks


@app.post("/upload")
async def upload_video(background_task: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")
    file_location = VIDEO_DIR / file.filename
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    
    ingestion_tasks[file.filename] = {"status": "processing", "step": "Upload complete. Starting background ingestion..."}
    background_task.add_task(
        run_ingestion_pipeline,
        video_path=str(file_location),
        video_name=file.filename
    )
    return {"message": f"{file.filename} uploaded successfully. Ingestion started in background"}
    

@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    if video_agent is None:
        await websocket.close(code=1011, reason="Agent not initialized")
        return
    await websocket.accept()
    try:
        while True:
            question = await websocket.receive_text()
            async for chunk in video_agent.astream_ask(question):
                await websocket.send_text(chunk)
            await websocket.send_text("[DONE]")

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
