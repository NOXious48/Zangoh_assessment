# Audio Customer Support Agent

A modular audio-based customer support agent that seamlessly integrates Speech-to-Text (STT), a Large Language Model (LLM) powered by Retrieval-Augmented Generation (RAG), and Text-to-Speech (TTS) technologies.

## Project Overview

This repository contains the complete implementation of the Audio Customer Support Agent. The agent is capable of taking audio input, transcribing it, querying a domain-specific knowledge base (customer support documents) to generate accurate responses, and synthesizing the response back into audio.

### Pipeline Flow
```text
Audio Input → STT (Speech-to-Text) → LLM Agent (with RAG/ChromaDB) → TTS (Text-to-Speech) → Audio Output
```

## Architecture

The system is built with modularity and asynchronous processing in mind, utilizing the following core components:

1. **STT (Speech-to-Text)**: `src/stt/base_stt.py`
   - Handles accurate transcription of incoming customer audio queries into text.
2. **LLM Agent with RAG**: `src/llm/agent.py`
   - Built using the LangChain ReAct framework.
   - Integrates ChromaDB as a vector store populated with comprehensive customer support documents (returns, shipping, warranties, etc.).
   - Retrieves relevant context before generating an LLM response.
3. **TTS (Text-to-Speech)**: `src/tts/base_tts.py`
   - Synthesizes the generated LLM text responses into natural-sounding speech audio.
4. **Pipeline Orchestrator**: `src/pipeline.py`
   - Orchestrates the complete STT → LLM → TTS flow.
   - Handles errors gracefully across the audio processing pipeline.
5. **API Server**: `src/api/server.py`
   - FastAPI server with REST endpoints for testing text and audio processing, as well as a `/health` endpoint for component monitoring.
6. **Testing UI**: `streamlit_app.py`
   - A Streamlit-based web interface for testing text/audio chats and visualizing system health.

### Mid-Session Requirement
*(Note: If you have a specific mid-session requirement implementation to detail for your submission, please describe it here.)*

---

## Quick Start

### 1. Installation

```bash
# Navigate to the project root
# Create virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install core dependencies
pip install fastapi uvicorn streamlit requests numpy
pip install langchain chromadb sentence-transformers

# Optional: For audio recording in UI
pip install sounddevice
```

### 2. Environment Setup

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` with your active API keys based on the services you configured:
```env
STT_API_KEY=your_stt_api_key_here
LLM_API_KEY=your_llm_api_key_here
TTS_API_KEY=your_tts_api_key_here
```

### 3. Running the Application

To run the full stack, use a two-terminal workflow:

**Terminal 1: Start the API Server**
```bash
python -m src.api.server
```
*The server will start on `http://localhost:8000`.*

**Terminal 2: Launch the Testing UI**
```bash
streamlit run streamlit_app.py
```
*The Streamlit dashboard will open in your browser at `http://localhost:8501`.*

---

## API Usage Examples

Once the FastAPI server is running, you can manually interact with the API endpoints.

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Text Chat Endpoint:**
```bash
curl -X POST http://localhost:8000/chat/text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is your return policy?"}'
```

**Audio Chat Endpoint (Full Pipeline):**
```bash
curl -X POST http://localhost:8000/chat/audio \
  -F "audio=@test_audio.wav" --output response.mp3
```

---

## Development & Testing

You can use the built-in testing utilities to verify your Knowledge Base (RAG) implementation independently of the UI:

```bash
python src/utils/kb_test.py
```

To run unit tests (if configured):
```bash
pytest tests/
```