import pytest
from unittest.mock import patch, MagicMock
from domains.video.hardware import detect_available_nvenc_codecs, assert_nvenc_available, get_best_video_codec
from core.commands import CommandResult

# Mock output for ffmpeg -encoders
MOCK_ENCODERS_NVENC = """
Encoders:
 V..... = Video
 ------
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder (codec hevc)
 A..... aac                  AAC (Advanced Audio Coding)
"""

MOCK_ENCODERS_NO_NVENC = """
Encoders:
 V..... = Video
 ------
 V..... libx265              libx265 H.265 / HEVC / MPEG-H Part 2 / ICTCP
 A..... aac                  AAC (Advanced Audio Coding)
"""

@patch("domains.video.hardware.run_command")
def test_detect_available_nvenc_codecs_success(mock_run):
    mock_run.return_value = CommandResult(
        command=["ffmpeg", "-encoders"],
        returncode=0,
        stdout=MOCK_ENCODERS_NVENC,
        stderr=""
    )
    
    codecs = detect_available_nvenc_codecs()
    assert "h264_nvenc" in codecs
    assert "hevc_nvenc" in codecs
    assert len(codecs) == 2

@patch("domains.video.hardware.run_command")
def test_detect_available_nvenc_codecs_none(mock_run):
    mock_run.return_value = CommandResult(
        command=["ffmpeg", "-encoders"],
        returncode=0,
        stdout=MOCK_ENCODERS_NO_NVENC,
        stderr=""
    )
    
    codecs = detect_available_nvenc_codecs()
    assert len(codecs) == 0

@patch("domains.video.hardware.run_command")
def test_detect_available_nvenc_codecs_ffmpeg_missing(mock_run):
    mock_run.side_effect = FileNotFoundError()
    
    with pytest.raises(RuntimeError, match="FFmpeg is not installed"):
        detect_available_nvenc_codecs()

@patch("domains.video.hardware.detect_available_nvenc_codecs")
def test_assert_nvenc_available_ok(mock_detect):
    mock_detect.return_value = ["h264_nvenc"]
    # Should not raise
    assert_nvenc_available("h264_nvenc")

@patch("domains.video.hardware.detect_available_nvenc_codecs")
def test_assert_nvenc_available_fail(mock_detect):
    mock_detect.return_value = []
    with pytest.raises(RuntimeError, match="h264_nvenc' is not available"):
        assert_nvenc_available("h264_nvenc")

@patch("domains.video.hardware.detect_available_nvenc_codecs")
def test_get_best_video_codec_nvenc(mock_detect):
    mock_detect.return_value = ["h264_nvenc"]
    assert get_best_video_codec(prefer_nvenc=True) == "h264_nvenc"

@patch("domains.video.hardware.detect_available_nvenc_codecs")
def test_get_best_video_codec_fallback(mock_detect):
    mock_detect.return_value = []
    assert get_best_video_codec(prefer_nvenc=True) == "libx265"

def test_get_best_video_codec_no_prefer():
    assert get_best_video_codec(prefer_nvenc=False) == "libx265"
