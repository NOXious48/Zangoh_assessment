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
    Edge-TTS implementation.
    
    Uses Microsoft Edge's free TTS service for speech synthesis.
    
    Input: text (str) - Text to convert to speech
    Output: audio_bytes (bytes) - Audio data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.voice = None
        self._rate = None
        self._volume = None
    
    async def initialize(self) -> None:
        """
        Initialize the Edge-TTS service.
        """
        self.voice = self.config.get("voice", "en-US-AriaNeural") 
        self._rate = self.config.get("rate", "+0%")
        self._volume = self.config.get("volume", "+0%")
        self.is_initialized = True 
        logger.info(f"TTS service initialized with voice: {self.voice}")
    
    async def synthesize(self, text: str, **kwargs) -> bytes:
        """
        Convert text to speech using Edge-TTS.
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
            
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        import edge_tts 
        
        try:
            voice = kwargs.get("voice", self.voice)
            rate = kwargs.get("rate", self._rate)
            volume = kwargs.get("volume", self._volume)
            
            communicate = edge_tts.Communicate(text, voice, rate=rate, volume=volume) 
            audio_bytes = b"" 
            
            async for chunk in communicate.stream(): 
                if chunk["type"] == "audio": 
                    audio_bytes += chunk["data"] 
                    
            logger.info(f"TTS synthesis complete: {len(audio_bytes)} bytes generated for text: '{text[:50]}...'")
            return audio_bytes
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {str(e)}")
            raise
    
    async def synthesize_stream(self, text: str, **kwargs) -> io.BytesIO:
        """
        Convert text to speech with streaming support.
        """
        if not self.is_ready():
            raise RuntimeError("TTS service not initialized")
        
        audio_data = await self.synthesize(text, **kwargs)
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)
        return audio_buffer
    
    async def cleanup(self) -> None:
        """
        Cleanup TTS resources.
        """
        self._voice = None
        self.is_initialized = False
        logger.info("TTS service cleaned up")