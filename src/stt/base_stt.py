"""
Base Speech-to-Text (STT) Interface

This module defines the abstract base class for Speech-to-Text implementations
and the concrete STTService using OpenAI's Whisper API.

Implementation: OpenAI Whisper API (whisper-1)
- High accuracy, supports multiple audio formats (wav, mp3, ogg, flac, etc.)
- Fast cloud-based transcription
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations.
    
    This class defines the interface that all STT implementations must follow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT service.
        
        Args:
            config: Configuration dictionary containing API keys, model settings, etc.
                   Example: {"api_key": "your_api_key", "model": "whisper-1"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the STT service (setup API clients, load models, etc.).
        This method should be called before using the STT service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data as bytes
            **kwargs: Additional parameters specific to the STT implementation
                     (e.g., language, model, formatting options)
        
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the STT service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the STT service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class STTService(BaseSTT):
    """
    OpenAI Whisper API STT implementation.
    
    Uses OpenAI's Whisper model (whisper-1) for high-accuracy cloud-based
    speech-to-text transcription. Supports WAV, MP3, OGG, FLAC formats.
    
    Input: audio_bytes (bytes) - Raw audio data
    Output: transcribed_text (str) - The text transcription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
    
    async def initialize(self) -> None:
        """
        Initialize the OpenAI Whisper STT service.
        
        Steps:
        1. Get API key from config or environment
        2. Create AsyncOpenAI client instance
        3. Set is_initialized to True
        """
        import os
        from openai import AsyncOpenAI
        
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key in config.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.config.get("model", "whisper-1")
        self.is_initialized = True
        logger.info(f"STT service initialized with model: {self.model}")
    
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text using OpenAI Whisper API.
        
        Input: audio_bytes (bytes) - Raw audio data in any supported format
        Output: str - Transcribed text
        
        Steps:
        1. Check if service is initialized
        2. Wrap audio bytes in a file-like object with filename
        3. Call OpenAI transcription API
        4. Return transcribed text
        """
        if not self.is_ready():
            raise RuntimeError("STT service not initialized")
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Empty audio data provided")
        
        try:
            # Wrap bytes in a file-like buffer with a filename for the API
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = "audio.wav"
            
            # Call OpenAI Whisper transcription API
            language = kwargs.get("language", None)
            
            transcription_params = {
                "model": self.model,
                "file": audio_file,
            }
            if language:
                transcription_params["language"] = language
            
            transcript = await self.client.audio.transcriptions.create(
                **transcription_params
            )
            
            transcribed_text = transcript.text.strip()
            logger.info(f"Transcription result: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """
        Cleanup STT resources.
        
        Steps:
        1. Close the OpenAI client
        2. Set is_initialized to False
        """
        if self.client:
            await self.client.close()
        self.client = None
        self.is_initialized = False
        logger.info("STT service cleaned up")