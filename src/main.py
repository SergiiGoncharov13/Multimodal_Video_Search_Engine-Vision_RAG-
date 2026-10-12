import os
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv, find_dotenv

from agent import VideoAnalysisAgent
from schemas import QueryRequest, QueryResponse

load_dotenv(find_dotenv(), override=True)
video_agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_agent
    load_dotenv(find_dotenv(), override=True)
    db_url = os.getenv('DATABASE_URL')
    video_agent = VideoAnalysisAgent(db_url)
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/ask", response_model=QueryResponse)
async def ask_agent(request: QueryRequest):
    if video_agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    try:
        answer = await asyncio.to_thread(video_agent.ask, request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
