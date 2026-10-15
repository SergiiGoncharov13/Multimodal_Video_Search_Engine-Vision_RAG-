import os
import sys
import warnings
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from dotenv import load_dotenv, find_dotenv

from agent import VideoAnalysisAgent

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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


app = FastAPI(title="Multimodal Video Search Engine API", lifespan=lifespan)
app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
app.mount("/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")


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
