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
    Local Whisper STT implementation.
    
    Uses OpenAI's Whisper model locally for high-accuracy
    speech-to-text transcription.
    
    Input: audio_bytes (bytes) - Raw audio data
    Output: transcribed_text (str) - The text transcription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
    
    async def initialize(self) -> None:
        """
        Initialize the local Whisper STT service.
        """
        import whisper 
        model_name = self.config.get("model", "base") 
        self.client = whisper.load_model(model_name) 
        self.is_initialized = True 
        logger.info(f"STT service initialized with model: {model_name}")
    
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text using local Whisper.
        """
        if not self.is_ready():
            raise RuntimeError("STT service not initialized")
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise ValueError("Empty audio data provided")
            
        import tempfile 
        import os
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file: 
                temp_file.write(audio_bytes) 
                temp_path = temp_file.name
                
            result = self.client.transcribe(temp_path) 
            
            os.unlink(temp_path)
            
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """
        Cleanup STT resources.
        """
        self.client = None
        self.is_initialized = False
        logger.info("STT service cleaned up")