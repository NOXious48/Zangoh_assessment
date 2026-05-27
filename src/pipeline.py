"""
Audio Customer Support Agent Pipeline

This module orchestrates the complete STT -> LLM -> TTS pipeline.
All three components are wired together to process audio input end-to-end.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    
    This class manages the entire audio processing pipeline for customer support.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the audio support pipeline.
        
        Args:
            config: Pipeline configuration containing settings for all components
        """
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False
        
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    async def initialize(self) -> None:
        """
        Initialize all pipeline components.
        
        Steps:
        1. Initialize STT service (OpenAI Whisper)
        2. Initialize LLM agent (ChatGPT with RAG)
        3. Initialize TTS service (OpenAI TTS)
        4. Verify all components are ready
        
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            self.logger.info("Initializing Audio Support Pipeline...")
            
            # Initialize STT
            self.logger.info("Initializing STT service...")
            self.stt = STTService(self.config.stt_config)
            await self.stt.initialize()
            self.logger.info("STT service initialized successfully")
            
            # Initialize LLM Agent
            self.logger.info("Initializing LLM agent...")
            self.llm_agent = CustomerSupportAgent(self.config.llm_config)
            await self.llm_agent.initialize()
            self.logger.info("LLM agent initialized successfully")
            
            # Initialize TTS
            self.logger.info("Initializing TTS service...")
            self.tts = TTSService(self.config.tts_config)
            await self.tts.initialize()
            self.logger.info("TTS service initialized successfully")
            
            # Verify all components are ready
            if not all([self.stt.is_ready(), self.llm_agent.is_initialized, self.tts.is_ready()]):
                raise RuntimeError("Some pipeline components failed to initialize")
            
            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise
    
    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        Process audio input through the complete pipeline.
        
        This is the main method that handles the STT -> LLM -> TTS flow.
        
        Args:
            audio_bytes: Input audio data
            **kwargs: Additional parameters for processing
            
        Returns:
            bytes: Response audio data (MP3 format)
            
        Raises:
            RuntimeError: If pipeline is not initialized
            Exception: If processing fails at any stage
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Step 1: Speech to Text
            self.logger.info("Step 1/3: Converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            self.logger.info(f"Transcribed text: {text_input}")
            
            # Step 2: Process with LLM Agent
            self.logger.info("Step 2/3: Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            self.logger.info(f"Agent response: {agent_response[:200]}...")
            
            # Step 3: Text to Speech
            self.logger.info("Step 3/3: Converting response to speech...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info(f"Audio response generated: {len(response_audio)} bytes")
            
            return response_audio
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        Process text input (useful for testing without STT).
        
        Args:
            text_input: Text query from user
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, bytes]: (agent_response_text, response_audio)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        try:
            # Process with LLM Agent
            self.logger.info(f"Processing text query: {text_input}")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            self.logger.info(f"Agent response: {agent_response[:200]}...")
            
            # Convert to speech
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info(f"Audio response generated: {len(response_audio)} bytes")
            
            return agent_response, response_audio
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check the health status of all pipeline components.
        
        Returns:
            Dict[str, bool]: Status of each component
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }
    
    async def cleanup(self) -> None:
        """
        Cleanup all pipeline resources.
        
        This method should be called when the pipeline is no longer needed.
        """
        self.logger.info("Cleaning up pipeline resources...")
        
        try:
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()
                
            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise

    def _create_transcript_data(self, user_input: str, agent_response: str) -> Dict[str, Any]:
        """Create structured transcript data."""
        return {
            "user_input": user_input,
            "agent_response": agent_response
        }

    async def process_audio_with_transcript(self, audio_bytes: bytes, **kwargs) -> Tuple[bytes, Dict[str, Any], int]:
        """Process audio and capture transcript data and processing time."""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        start_time = time.time()
        try:
            # Step 1: STT
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            
            # Step 2: LLM
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            
            # Step 3: TTS
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            
            # Metadata & Transcript
            processing_time_ms = int((time.time() - start_time) * 1000)
            transcript_data = self._create_transcript_data(text_input, agent_response)
            
            return response_audio, transcript_data, processing_time_ms
            
        except Exception as e:
            self.logger.error(f"Pipeline processing with transcript failed: {str(e)}")
            raise

    async def process_text_with_timing(self, text: str, **kwargs) -> Tuple[str, int]:
        """Process text and capture processing time"""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
            
        start_time = time.time()
        try:
            agent_response = await self.llm_agent.process_query(text, **kwargs)
            processing_time_ms = int((time.time() - start_time) * 1000)
            return agent_response, processing_time_ms
        except Exception as e:
            self.logger.error(f"Text processing with timing failed: {str(e)}")
            raise


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True
) -> AudioSupportPipeline:
    """
    Factory function to create and initialize a pipeline.
    
    Args:
        stt_config: STT configuration
        llm_config: LLM configuration  
        tts_config: TTS configuration
        enable_logging: Whether to enable logging
        
    Returns:
        AudioSupportPipeline: Initialized pipeline instance
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging
    )
    
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    
    return pipeline


if __name__ == "__main__":
    """
    Example usage of the pipeline.
    """
    async def main():
        import os
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return
        
        stt_config = {"api_key": api_key, "model": "whisper-1"}
        llm_config = {"api_key": api_key, "model": "gpt-3.5-turbo", "temperature": 0.7}
        tts_config = {"api_key": api_key, "voice": "alloy", "model": "tts-1"}
        
        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        
        # Test with text input
        response_text, response_audio = await pipeline.process_text(
            "What is your return policy?"
        )
        print(f"Response: {response_text}")
        print(f"Audio size: {len(response_audio)} bytes")
        
        # Cleanup
        await pipeline.cleanup()
        print("Pipeline example completed successfully!")
    
    asyncio.run(main())