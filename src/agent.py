import os
import sys
import warnings
import json
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from typing import Optional
from models import VideoFrame

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()


def _extract_text_content(content) -> str:
    """Safely extracts text string from LangChain / Google GenAI message content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif hasattr(part, "text"):
                parts.append(getattr(part, "text"))
        return "".join(parts)
    return str(content) if content is not None else ""


class VideoAnalysisAgent:
    def __init__(self, db_url: str, video_dir: Optional[str] = None):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.video_dir = Path(video_dir).resolve() if video_dir else None
        self.last_matched_moments = []
        
        print("Loading local embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        search_tool = StructuredTool.from_function(
            func=self._search_video_timestamp,
            name="search_video_timestamp",
            description="""
            Searches the video database for a specific event, object, or action.
            Returns matching timestamps, filenames, and visual captions.
            """
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL", "gemini-3.5-flash"),
            temperature=0.0,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        system_prompt = (
            "You are an expert video analysis agent. You answer questions about video files. "
            "Call the search_video_timestamp tool once or twice with a relevant query to retrieve evidence. "
            "Then synthesize a clear and helpful answer based on the retrieved scene captions."
        )

        self.agent_executer = create_react_agent(
            self.llm,
            [search_tool],
            prompt=system_prompt
        )
        print("React Agent created successfully")

    def _search_video_timestamp(self, query: str, limit: int = 6) -> str:
        """
        Searches the video database for visual events, objects, actions, OR spoken dialogue.
        Call this tool if the user asks what happens visually, or what was said/spoken.
        """
        print(f"\n[Agent called tool] Searching for '{query}'")
        query_vector = self.embedder.encode(query)
        with self.SessionLocal() as session:
            results = session.query(VideoFrame).order_by(VideoFrame.text_embedding.cosine_distance(query_vector)).limit(limit).all()
            if not results:
                return "No matching scenes found in the database."

            seen = set()
            unique_moments = []
            for res in results:
                cue_key = (res.video_name, int(res.timestamp_sec), res.modality)
                if cue_key not in seen:
                    seen.add(cue_key)
                    file_exists = False
                    if self.video_dir:
                        file_exists = (self.video_dir / res.video_name).exists()
                    else:
                        file_exists = (Path("data/videos") / res.video_name).exists()

                    frame_url = None
                    if res.filepath:
                        frame_url = f"/frames/{Path(res.filepath).name}"

                    unique_moments.append({
                        "video_name": res.video_name,
                        "timestamp_sec": int(res.timestamp_sec),
                        "formatted": res.timestamp_formatted,
                        "modality": res.modality or "visual",
                        "file_exists": file_exists,
                        "frame_url": frame_url,
                        "caption": (res.caption or "").strip()
                    })
            self.last_matched_moments = unique_moments

        formatted_results = []
        for res in results:
            formatted_results.append(f"[{res.modality.upper()}] {res.timestamp_sec}s in {res.video_name}: {res.caption}")
        return "\n".join(formatted_results)

    async def astream_ask(self, question: str):
        self.last_matched_moments = []
        inputs = {"messages": [HumanMessage(content=question)]}
        
        async for chunk in self.agent_executer.astream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    yield f"🔄 Calling tool: {tool_call['name']} with {tool_call['args']}"
            
            elif message.type == "tool":
                yield "✅ Database returned results"

            elif message.type == "ai" and message.content:
                text_content = _extract_text_content(message.content)
                if text_content.strip():
                    yield f"Final answer:\n{text_content}"

        if self.last_matched_moments:
            payload = json.dumps({"type": "PLAYBACK_CUES", "cues": self.last_matched_moments})
            yield f"__METADATA__{payload}"
