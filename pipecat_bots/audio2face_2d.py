"""Audio2Face-2D avatar service for Pipecat.

Integrates NVIDIA Audio2Face-2D NIM to animate an avatar based on TTS audio.
Takes audio frames from TTS and generates synchronized video frames of the avatar.

The service uses gRPC to communicate with the Audio2Face-2D NIM container which
animates a portrait image based on audio input.

Environment variables:
    NVIDIA_A2F_URL      Audio2Face-2D gRPC URL (default: localhost:8001)
    NVIDIA_A2F_AVATAR   Path to avatar portrait image (default: assets/avatar.png)

Usage:
    avatar = Audio2Face2DService(
        server_url="localhost:8001",
        avatar_path="assets/avatar.png"
    )
    # In pipeline: ... -> tts -> avatar -> transport.output() -> ...
"""

import asyncio
import io
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    OutputImageRawFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

try:
    import grpc
    from grpc import aio as grpc_aio
except ImportError:
    logger.error("grpcio not installed. Install with: pip install grpcio grpcio-tools")
    raise

# Default configuration
DEFAULT_A2F_URL = "localhost:8001"
DEFAULT_AVATAR_PATH = Path(__file__).parent / "assets" / "avatar.png"

# Video output settings
OUTPUT_FPS = 30
OUTPUT_WIDTH = 512
OUTPUT_HEIGHT = 512


class Audio2Face2DConfig(BaseModel):
    """Configuration for Audio2Face-2D service."""

    model_selection: str = "QUALITY"  # PERF or QUALITY
    animation_crop_mode: str = "INSET_BLENDING"  # FACEBOX, REGISTRATION_BLENDING, INSET_BLENDING
    head_pose_mode: str = "PRE_DEFINED_ANIMATION"  # RETAIN_FROM_PORTRAIT_IMAGE, PRE_DEFINED_ANIMATION, USER_DEFINED_ANIMATION
    enable_lookaway: bool = True
    lookaway_max_offset: int = 10
    lookaway_interval_range: int = 90
    lookaway_interval_min: int = 30
    blink_frequency: int = 15  # blinks per minute
    blink_duration: int = 3  # frames
    mouth_expression_multiplier: float = 1.0
    head_pose_multiplier: float = 0.5


@dataclass
class Audio2Face2DParams:
    """Runtime parameters for Audio2Face-2D."""

    config: Audio2Face2DConfig = field(default_factory=Audio2Face2DConfig)


class Audio2Face2DService(FrameProcessor):
    """Avatar animation service using NVIDIA Audio2Face-2D NIM.

    This service listens for TTSAudioRawFrame, buffers audio, and generates
    synchronized video frames of an animated avatar.

    The service operates in two modes:
    1. Streaming mode: Audio chunks are sent as they arrive, video frames stream back
    2. Batch mode: Audio is buffered until TTSStoppedFrame, then processed

    Currently implements batch mode for simplicity and stability.
    """

    def __init__(
        self,
        *,
        server_url: str = DEFAULT_A2F_URL,
        avatar_path: Optional[str] = None,
        params: Optional[Audio2Face2DParams] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._server_url = server_url
        self._avatar_path = Path(avatar_path) if avatar_path else DEFAULT_AVATAR_PATH
        self._params = params or Audio2Face2DParams()

        # gRPC channel and stub
        self._channel: Optional[grpc_aio.Channel] = None
        self._stub = None

        # Audio buffer for batch processing
        self._audio_buffer = bytearray()
        self._audio_sample_rate = 22000  # Magpie TTS sample rate

        # Portrait image data (loaded once)
        self._portrait_data: Optional[bytes] = None

        # State tracking
        self._is_speaking = False
        self._started = False

        logger.info(f"Audio2Face2D service initialized")
        logger.info(f"  Server URL: {self._server_url}")
        logger.info(f"  Avatar path: {self._avatar_path}")

    async def _load_portrait(self) -> bytes:
        """Load and cache the portrait image."""
        if self._portrait_data is not None:
            return self._portrait_data

        if not self._avatar_path.exists():
            raise FileNotFoundError(f"Avatar image not found: {self._avatar_path}")

        self._portrait_data = self._avatar_path.read_bytes()
        logger.info(f"Loaded portrait image: {len(self._portrait_data)} bytes")
        return self._portrait_data

    async def _connect(self):
        """Establish gRPC connection to Audio2Face-2D server."""
        if self._channel is not None:
            return

        try:
            # Use insecure channel for local development
            self._channel = grpc_aio.insecure_channel(self._server_url)

            # Import the generated protobuf modules
            # These need to be compiled from the Audio2Face-2D protos
            try:
                from . import audio2face2d_pb2
                from . import audio2face2d_pb2_grpc

                self._stub = audio2face2d_pb2_grpc.Audio2Face2DServiceStub(self._channel)
                self._pb2 = audio2face2d_pb2
                logger.info(f"Connected to Audio2Face-2D server at {self._server_url}")
            except ImportError:
                logger.warning(
                    "Audio2Face-2D protobuf modules not found. "
                    "Running in simulation mode (static avatar)."
                )
                self._stub = None

        except Exception as e:
            logger.error(f"Failed to connect to Audio2Face-2D: {e}")
            self._channel = None
            raise

    async def _disconnect(self):
        """Close gRPC connection."""
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None
            logger.info("Disconnected from Audio2Face-2D server")

    async def _animate_avatar(self, audio_data: bytes) -> AsyncGenerator[bytes, None]:
        """Send audio to Audio2Face-2D and yield video frame data.

        Args:
            audio_data: Raw PCM audio data (16-bit, mono)

        Yields:
            PNG image bytes for each video frame
        """
        if self._stub is None:
            # Simulation mode: yield static portrait frames
            portrait = await self._load_portrait()
            # Calculate frame count based on audio duration
            audio_duration = len(audio_data) / (self._audio_sample_rate * 2)  # 16-bit = 2 bytes
            frame_count = int(audio_duration * OUTPUT_FPS)

            for _ in range(max(1, frame_count)):
                yield portrait
                await asyncio.sleep(1.0 / OUTPUT_FPS)
            return

        try:
            portrait = await self._load_portrait()

            # Build configuration
            config = self._pb2.AnimateConfig(
                portrait_image=portrait,
                model_selection=getattr(
                    self._pb2.ModelSelection, self._params.config.model_selection
                ),
                animation_crop_mode=getattr(
                    self._pb2.AnimationCroppingMode, self._params.config.animation_crop_mode
                ),
                head_pose_mode=getattr(
                    self._pb2.HeadPoseMode, self._params.config.head_pose_mode
                ),
                enable_lookaway=self._params.config.enable_lookaway,
                lookaway_max_offset=self._params.config.lookaway_max_offset,
                lookaway_interval_range=self._params.config.lookaway_interval_range,
                lookaway_interval_min=self._params.config.lookaway_interval_min,
                blink_frequency=self._params.config.blink_frequency,
                blink_duration=self._params.config.blink_duration,
                mouth_expression_multiplier=self._params.config.mouth_expression_multiplier,
                head_pose_multiplier=self._params.config.head_pose_multiplier,
            )

            # Create request generator
            async def request_generator():
                # First send config
                yield self._pb2.AnimateRequest(config=config)

                # Then send audio in chunks (1MB each as per NVIDIA recommendation)
                chunk_size = 1024 * 1024
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i : i + chunk_size]
                    yield self._pb2.AnimateRequest(audio_file_data=chunk)

            # Stream responses
            responses = self._stub.Animate(request_generator())
            async for response in responses:
                if response.HasField("video_file_data"):
                    # Extract frames from video data
                    # For real implementation, would decode video to frames
                    yield response.video_file_data

        except grpc.RpcError as e:
            logger.error(f"gRPC error during animation: {e}")
            # Fall back to static portrait
            portrait = await self._load_portrait()
            yield portrait

    async def _output_avatar_frames(self, audio_data: bytes):
        """Process audio and output avatar video frames."""
        frame_count = 0
        start_time = time.time()

        async for frame_data in self._animate_avatar(audio_data):
            # Create output image frame
            # Note: Audio2Face-2D outputs video, we extract frames
            frame = OutputImageRawFrame(
                image=frame_data,
                size=(OUTPUT_WIDTH, OUTPUT_HEIGHT),
                format="png",
            )
            await self.push_frame(frame)
            frame_count += 1

        duration = time.time() - start_time
        logger.debug(f"Output {frame_count} avatar frames in {duration:.2f}s")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and generate avatar output."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            self._started = True
            try:
                await self._connect()
            except Exception as e:
                logger.error(f"Failed to initialize Audio2Face-2D: {e}")
            await self.push_frame(frame, direction)

        elif isinstance(frame, EndFrame):
            await self._disconnect()
            self._started = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, CancelFrame):
            self._audio_buffer.clear()
            self._is_speaking = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, InterruptionFrame):
            self._audio_buffer.clear()
            self._is_speaking = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStartedFrame):
            self._is_speaking = True
            self._audio_buffer.clear()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSAudioRawFrame):
            # Buffer audio for batch processing
            self._audio_buffer.extend(frame.audio)
            # Pass through audio unchanged
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            self._is_speaking = False
            # Process buffered audio and output avatar frames
            if len(self._audio_buffer) > 0:
                audio_data = bytes(self._audio_buffer)
                self._audio_buffer.clear()
                # Run avatar animation in background to not block audio
                asyncio.create_task(self._output_avatar_frames(audio_data))
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


class StaticAvatarService(FrameProcessor):
    """Simple static avatar display service.

    Displays a static avatar image when the bot is speaking.
    This is a simpler alternative to Audio2Face-2D when animation
    is not required or the NIM is not available.
    """

    def __init__(
        self,
        *,
        avatar_path: Optional[str] = None,
        output_fps: int = 1,  # Low FPS for static image
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._avatar_path = Path(avatar_path) if avatar_path else DEFAULT_AVATAR_PATH
        self._output_fps = output_fps
        self._portrait_data: Optional[bytes] = None
        self._portrait_size: tuple[int, int] = (512, 512)
        self._is_speaking = False
        self._display_task: Optional[asyncio.Task] = None

        logger.info(f"Static avatar service initialized: {self._avatar_path}")

    async def _load_portrait(self):
        """Load portrait image and determine size."""
        if self._portrait_data is not None:
            return

        if not self._avatar_path.exists():
            logger.warning(f"Avatar image not found: {self._avatar_path}")
            return

        self._portrait_data = self._avatar_path.read_bytes()

        # Try to get image dimensions using PIL if available
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(self._portrait_data))
            self._portrait_size = img.size
        except ImportError:
            pass  # Use default size

        logger.info(f"Loaded avatar: {self._avatar_path} ({self._portrait_size})")

    async def _display_loop(self):
        """Continuously display avatar while speaking."""
        while self._is_speaking:
            if self._portrait_data:
                frame = OutputImageRawFrame(
                    image=self._portrait_data,
                    size=self._portrait_size,
                    format="png",
                )
                await self.push_frame(frame)
            await asyncio.sleep(1.0 / self._output_fps)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames and display avatar when speaking."""
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self._load_portrait()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStartedFrame):
            self._is_speaking = True
            if self._display_task is None or self._display_task.done():
                self._display_task = asyncio.create_task(self._display_loop())
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            self._is_speaking = False
            if self._display_task and not self._display_task.done():
                self._display_task.cancel()
                try:
                    await self._display_task
                except asyncio.CancelledError:
                    pass
            await self.push_frame(frame, direction)

        elif isinstance(frame, (CancelFrame, InterruptionFrame)):
            self._is_speaking = False
            if self._display_task and not self._display_task.done():
                self._display_task.cancel()
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
