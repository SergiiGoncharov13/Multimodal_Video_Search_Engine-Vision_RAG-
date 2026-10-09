# Multimodal Video Search Engine (Vision RAG)

An on-device, multimodal Video Retrieval-Augmented Generation (Vision RAG) engine designed for Apple Silicon. 

The system extracts keyframes from video files, generates rich visual descriptions using an on-device Vision-Language Model (VLM), indexes semantic text embeddings into PostgreSQL (`pgvector`), and enables natural language question-answering over video moments with an on-device Large Language Model (LLM) via a Streamlit web interface.

---

## Architecture Overview

1. **Frame Extraction (`OpenCV`)**: Samples video keyframes at 1 frame per second (1 FPS).
2. **Visual Captioning (`MLX-VLM`)**: Employs `mlx-community/Qwen2-VL-2B-Instruct-4bit` to generate detailed textual descriptions of each frame directly in unified memory.
3. **Semantic Embedding (`Sentence-Transformers`)**: Encodes frame descriptions into 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
4. **Vector Storage (`pgvector`)**: Stores frame metadata, captions, and vector embeddings in a PostgreSQL container running `pgvector`.
5. **RAG & Generation (`MLX-LM` + `Streamlit`)**: Retrieves the best matching video frame via cosine similarity and synthesizes a grounded answer using `mlx-community/Qwen2.5-1.5B-Instruct-4bit`.

---

## System Requirements

- **Hardware**: Mac with Apple Silicon (**M1 / M2 / M3 / M4**).
  - *MLX relies on Apple Silicon's unified memory architecture and Metal GPU acceleration.*
- **Operating System**: macOS 13.0+ (Ventura, Sonoma, Sequoia).
- **Python**: Python **3.12** recommended.
- **Docker**: Docker Desktop for macOS (with Docker Compose support).

---

## Installation

### 1. Clone & Navigate to the Repository

```bash
git clone <repository-url>
cd Multimodal_Video_Search_Engine\(Vision_RAG\)
```

### 2. Set Up Virtual Environment

Create and activate a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy `.env.example` to `.env` and verify your database credentials:

```bash
cp .env.example .env
```

Default `.env` configuration:
```env
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=video_rag
```

---

## How to Run

### Step 1: Start PostgreSQL with pgvector

Launch the database container in the background:

```bash
docker compose up -d
```

### Step 2: Ingest Video

Place your video file in `data/videos/` (e.g. `data/videos/video_file.mp4`) and run the ingestion pipeline:

```bash
python src/ingest.py data/videos/video_file.mp4
```

This will:
- Extract frames at 1 FPS into `data/extracted_frames/`.
- Generate detailed scene captions using `Qwen2-VL-2B-Instruct-4bit`.
- Embed the captions and store vectors in the PostgreSQL database.

### Step 3: Launch the Streamlit Web Application

Start the interactive search UI:

```bash
streamlit run src/app.py
```

Open your browser at `http://localhost:8501`. Ask any question about the video content to retrieve the matched frame timestamp, AI vision logs, and model-generated answer.

---

## Project Structure

```text
├── data/
│   ├── extracted_frames/   # Extracted 1 FPS image frames (.jpg)
│   └── videos/             # Source input videos (.mp4)
├── src/
│   ├── app.py              # Streamlit frontend & RAG Q&A engine
│   └── ingest.py           # Video frame extraction, VLM captioning & vector DB indexing
├── .env                    # Local environment variables
├── .env.example            # Example environment template
├── docker-compose.yml      # PostgreSQL + pgvector service definition
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation
```