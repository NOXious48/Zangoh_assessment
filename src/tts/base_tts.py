"""
Base Text-to-Speech (TTS) Interface

This module defines the abstract base class for Text-to-Speech implementations
and the concrete TTSService using OpenAI's TTS API.

Implementation: OpenAI TTS API (tts-1)
- High quality, natural-sounding voices
- Multiple voice options: alloy, echo, fable, onyx, nova, shimmer
- Fast cloud-based synthesis
"""

import io
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class BaseTTS(ABC):
    """
    Abstract base class for Text-to-Speech implementations.
    
    This class defines the interface that all TTS implementations must follow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TTS service.
        
        Args:
            config: Configuration dictionary containing API keys, voice settings, etc.
                   Example: {"api_key": "your_api_key", "voice": "alloy", "model": "tts-1"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the TTS service (setup API clients, load models, etc.).
        This method should be called before using the TTS service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Convert text to speech audio bytes.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters specific to the TTS implementation
                     (e.g., voice_id, speed, pitch, format)
        
        Returns:
            bytes: Audio data as bytes (typically MP3 or WAV format)
            
        Raises:
            Exception: If synthesis fails
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        Convert text to speech with streaming support.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional parameters for streaming
        
        Returns:
            io.BytesIO: Streaming audio data
            
        Raises:
            Exception: If streaming synthesis fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the TTS service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the TTS service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class TTSService(BaseTTS):
    """
    OpenAI TTS API implementation.
    
    Uses OpenAI's text-to-speech models for natural-sounding speech synthesis.
    Supports multiple voices and output formats.
    
    Input: text (str) - Text to convert to speech
    Output: audio_bytes (bytes) - Audio data (MP3 format)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.voice = None
        self.model = None
    
    async def initialize(self) -> None:
        """
        Initialize the OpenAI TTS service.
        
        Steps:
        1. Get API key from config or environment
        2. Create AsyncOpenAI client instance
        3. Set voice and model parameters
        4. Set is_initialized to True
        """
        import os
        from openai import AsyncOpenAI
        
        api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY or pass api_key in config.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.voice = self.config.get("voice", "alloy")
        self.model = self.config.get("model", "tts-1")
        self.is_initialized = True
        logger.info(f"TTS service initialized with model: {self.model}, voice: {self.voice}")
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Convert text to speech using OpenAI TTS API.
        
        Input: text (str) - Text to convert to speech
        Output: bytes - Audio data in MP3 format
        
        Steps:
        1. Check if service is initialized
        2. Validate input text
        3. Call OpenAI TTS API
        4. Return audio bytes
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Get voice override from kwargs or use default
            voice = kwargs.get("voice", self.voice)
            model = kwargs.get("model", self.model)
            speed = kwargs.get("speed", 1.0)
            
            # Call OpenAI TTS API
            response = await self.client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                speed=speed,
                response_format="mp3",
            )
            
            # Read the response content as bytes
            audio_bytes = response.content
            
            logger.info(f"TTS synthesis complete: {len(audio_bytes)} bytes generated for text: '{text[:50]}...'")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            raise
    
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        Convert text to speech with streaming support.
        
        Falls back to full synthesis and wraps the result in a BytesIO buffer
        for compatibility with streaming interfaces.
        
        Input: text (str) - Text to convert to speech
        Output: io.BytesIO - Streaming audio data buffer
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        
        # Perform full synthesis and wrap in buffer
        audio_data = await self.synthesize(text, **kwargs)
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        return audio_buffer
    
    async def cleanup(self) -> None:
        """
        Cleanup TTS resources.
        
        Steps:
        1. Close the OpenAI client
        2. Clear voice and model references
        3. Set is_initialized to False
        """
        if self.client:
            await self.client.close()
        self.client = None
        self.voice = None
        self.model = None
        self.is_initialized = False
        logger.info("TTS service cleaned up")
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get list of available OpenAI TTS voices.
        
        Returns:
            List of available voices with their metadata
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        
        # OpenAI's available TTS voices
        return [
            {"voice_id": "alloy", "name": "Alloy", "description": "Neutral and balanced"},
            {"voice_id": "echo", "name": "Echo", "description": "Warm and engaging"},
            {"voice_id": "fable", "name": "Fable", "description": "Expressive and dynamic"},
            {"voice_id": "onyx", "name": "Onyx", "description": "Deep and authoritative"},
            {"voice_id": "nova", "name": "Nova", "description": "Friendly and upbeat"},
            {"voice_id": "shimmer", "name": "Shimmer", "description": "Clear and gentle"},
        ]