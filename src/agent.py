import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from models import VideoFrame

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
sys.path.insert(0, str(Path(__file__).resolve().parent))
load_dotenv()


class VideoAnalysisAgent:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
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

    def _search_video_timestamp(self, query: str, limit: int = 5) -> str:
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

        formatted_results = []
        for res in results:
            formatted_results.append(f"- Timestamp {res.timestamp_sec}s, {res.modality} in {res.video_name}: {res.caption}")
        return "\n".join(formatted_results)

    async def astream_ask(self, question: str):
        inputs = {"messages": [HumanMessage(content=question)]}
        
        async for chunk in self.agent_executer.astream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    yield f"🔄 Calling tool: {tool_call['name']} with {tool_call['args']}"
            
            elif message.type == "tool":
                yield "✅ Database returned results"

            elif message.type == "ai" and message.content:
                yield f"Final answer:\n{message.content}"
