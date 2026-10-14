import os
import sys
import warnings
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from dotenv import load_dotenv, find_dotenv

from agent import VideoAnalysisAgent

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

src_dir = str(Path(__file__).resolve().parent)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
load_dotenv(find_dotenv(), override=True)

video_agent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_agent
    load_dotenv(find_dotenv(), override=True)
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("Warning: DATABASE_URL is not set.")
    print("Initializing VideoAnalysisAgent...")
    video_agent = VideoAnalysisAgent(db_url)
    print("VideoAnalysisAgent ready for queries!")
    yield


app = FastAPI(title="Multimodal Video Search Engine API", lifespan=lifespan)


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
        "agent_ready": video_agent is not None
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
