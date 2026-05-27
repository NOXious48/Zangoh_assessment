"""
FastAPI Server for Audio Customer Support Agent

This module provides REST API endpoints for testing the audio support pipeline.
All services (STT, LLM, TTS) are powered by OpenAI APIs using a single API key.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import logging
import os
import time

from dotenv import load_dotenv

from src.pipeline import AudioSupportPipeline, create_pipeline, PipelineConfig

# Load environment variables from .env file
load_dotenv()


class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str
    parameters: Optional[Dict[str, Any]] = {}


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """
    Initialize the pipeline on server startup.
    
    Uses OpenAI API for all three services (STT, LLM, TTS) with a single API key.
    """
    global pipeline
    
    try:
        logger.info("Starting Audio Support Agent API server...")
        
        # Get OpenAI API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            logger.warning(
                "OPENAI_API_KEY not set! Please set your API key in the .env file. "
                "Server will start but pipeline will not be functional."
            )
            return
        
        # Configure all services using OpenAI
        stt_config = {
            "api_key": api_key,
            "model": "whisper-1",
        }
        
        llm_config = {
            "api_key": api_key,
            "model": os.getenv("LLM_MODEL", "gpt-3.5-turbo"),
            "temperature": 0.7,
        }
        
        tts_config = {
            "api_key": api_key,
            "voice": os.getenv("TTS_VOICE", "alloy"),
            "model": "tts-1",
        }
        
        # Create and initialize the pipeline
        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline initialized successfully! All components ready.")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        # Don't raise here to allow server to start for debugging


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all pipeline components.
    """
    global pipeline
    
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False
            },
            message="Pipeline not initialized. Check OPENAI_API_KEY in .env file."
        )
    
    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            components=components,
            message="All components ready" if all_healthy else "Some components not ready"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check failed: {str(e)}"
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process text query through the LLM agent.
    
    This endpoint allows testing the LLM component without audio processing.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Set OPENAI_API_KEY in .env file.")
    
    try:
        start_time = time.time()
        
        # Process text through pipeline (LLM + TTS)
        response_text, response_audio = await pipeline.process_text(
            request.text, 
            **request.parameters
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return TextResponse(
            response_text=response_text,
            audio_available=len(response_audio) > 0,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """
    Process audio query through the complete pipeline.
    
    This endpoint handles the full STT -> LLM -> TTS pipeline.
    
    Args:
        audio: Audio file upload (WAV, MP3, etc.)
        
    Returns:
        Audio response as bytes
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Set OPENAI_API_KEY in .env file.")
    
    try:
        # Read audio file
        audio_bytes = await audio.read()
        
        # Validate audio format/size
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Process through the complete pipeline (STT -> LLM -> TTS)
        response_audio = await pipeline.process_audio(audio_bytes)
        
        # Return audio response
        return Response(
            content=response_audio,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    Convert text to audio using TTS.
    
    Useful for testing TTS component independently.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio file as bytes
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Set OPENAI_API_KEY in .env file.")
    
    try:
        if not pipeline.tts:
            raise HTTPException(status_code=503, detail="TTS not available")
        
        audio_bytes = await pipeline.tts.synthesize(text)
        
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    Debug endpoint for testing STT component independently.
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        Transcription result
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Set OPENAI_API_KEY in .env file.")
    
    try:
        audio_bytes = await audio.read()
        
        if not pipeline.stt:
            raise HTTPException(status_code=503, detail="STT not available")
        
        transcription = await pipeline.stt.transcribe(audio_bytes)
        
        return {"transcription": transcription}
        
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )