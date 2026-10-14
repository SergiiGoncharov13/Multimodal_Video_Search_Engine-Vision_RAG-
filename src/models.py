from sqlalchemy import Integer, String, Text
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from pgvector.sqlalchemy import Vector


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
