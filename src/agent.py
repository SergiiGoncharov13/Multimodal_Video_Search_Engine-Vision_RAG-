import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sentence_transformers import SentenceTransformer
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from ingest import VideoFrame, DATABASE_URL

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
            Call this tool whenever the user asks what happens in a video, or asks 
            to find a specific moment.
            """
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("MODEL"),
            temperature=0.0,
            max_retries=2,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        system_prompt = (
            "You are an expert video analysis agent. You answer questions about video files. "
            "Always use the search_video_timestamps tool to gather evidence before answering. "
            "If a user asks a complex question, call the tool multiple times with different search phrasing."
        )

        self.agent_executer = create_react_agent(
            self.llm,
            [search_tool],
            prompt=system_prompt
        )
        print("React Agent created successfully")

    def _search_video_timestamp(self, query: str, limit: int=3) -> str:
        print(f"\n[Agent called tool] Searching for '{query}'")
        query_vector = self.embedder.encode(query)
        with self.SessionLocal() as session:
            results = session.query(VideoFrame).order_by(VideoFrame.text_embedding.cosine_distance(query_vector)).limit(limit).all()
            if not results:
                return "No matching scenes found in the database."

        formatted_results = []
        for res in results:
            formatted_results.append(f"- Timestamp {res.timestamp_sec}s in {res.video_name}: {res.caption}")
        return "\n".join(formatted_results)
        
    def ask(self, question: str) -> str:
        inputs = {"messages": [HumanMessage(content=question)]}
        final_answer = ""

        for chunk in self.agent_executer.stream(inputs, stream_mode="values"):
            message = chunk["messages"][-1]
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"-> Agent routing to {tool_call['name']} with {tool_call['args']}")
            elif message.type == "ai" and message.content:
                final_answer = message.content

        return final_answer


if __name__ == "__main__":
    agent = VideoAnalysisAgent(DATABASE_URL)
    response = agent.ask("What happens in the video?")
    print(response)
