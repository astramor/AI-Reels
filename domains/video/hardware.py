#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re
from typing import List, Optional
from core.commands import run_command, CommandError

# Standard Logger
logger = logging.getLogger(__name__)

def detect_available_nvenc_codecs() -> List[str]:
    """
    Scans FFmpeg for available NVIDIA NVENC encoders.
    Returns a list of found encoders (e.g. ['h264_nvenc', 'hevc_nvenc']).
    
    Raises:
        RuntimeError: If FFmpeg is not installed or the command fails critically.
    """
    try:
        # We use -hide_banner to reduce noise
        result = run_command(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True)
        
        # Log complete output only on DEBUG level
        logger.debug(f"FFmpeg encoders output:\n{result.stdout}")
        
        # Regex to find video encoders (starting with V) that contain 'nvenc'
        # Format: V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
        nvenc_pattern = re.compile(r"^\s*V\.\.\.\.\.\s+(\S*nvenc\S*)", re.MULTILINE)
        codecs = nvenc_pattern.findall(result.stdout)
        
        logger.info(f"Detected NVENC codecs: {', '.join(codecs) if codecs else 'None'}")
        return codecs

    except FileNotFoundError:
        logger.error("FFmpeg not found in PATH. Please install FFmpeg.")
        raise RuntimeError("FFmpeg is not installed or not in PATH.")
    except Exception as e:
        logger.error(f"Failed to detect FFmpeg encoders: {e}")
        raise RuntimeError(f"FFmpeg encoder detection failed: {e}")

def assert_nvenc_available(codec: str) -> None:
    """
    Verifies that a specific NVENC codec is available.
    
    Args:
        codec: The name of the encoder (e.g. 'h264_nvenc')
        
    Raises:
        RuntimeError: If the codec is missing or FFmpeg fails.
    """
    available = detect_available_nvenc_codecs()
    if codec not in available:
        error_msg = (
            f"Requested NVENC encoder '{codec}' is not available in your FFmpeg installation. "
            "Ensure NVIDIA drivers and the correct FFmpeg build are installed."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info(f"Verified NVENC encoder '{codec}' is available.")

def get_best_video_codec(prefer_nvenc: bool = True) -> str:
    """
    Helper to choose the best available video codec.
    Prefers h264_nvenc if available and requested, otherwise falls back to libx265.
    
    Args:
        prefer_nvenc: Whether to attempt using NVENC.
        
    Returns:
        The name of the best available codec.
    """
    if prefer_nvenc:
        try:
            available = detect_available_nvenc_codecs()
            if "h264_nvenc" in available:
                logger.info("Using 'h264_nvenc' for hardware acceleration.")
                return "h264_nvenc"
            logger.warning("NVENC requested but 'h264_nvenc' not found. Falling back to 'libx265'.")
        except Exception as e:
            logger.warning(f"Error checking NVENC, falling back to CPU: {e}")
            
    return "libx265"
