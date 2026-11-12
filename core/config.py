from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import os
import shutil
import yaml

SETTINGS_ALIASES = {
    "manifest": "manifest_path",
    "crf": "ffmpeg_crf",
    "preset": "ffmpeg_preset",
    "audio_bitrate": "ffmpeg_audio_bitrate",
    "ducking_threshold": "ffmpeg_ducking_threshold",
    "ducking_ratio": "ffmpeg_ducking_ratio",
    "ducking_attack": "ffmpeg_ducking_attack",
    "ducking_release": "ffmpeg_ducking_release",
    "openai_base_url": "openai_base_url",
    "openai_api_key": "openai_api_key",
}


def load_yaml_defaults() -> dict:
    """Load config.yaml and map its keys to Settings field names."""
    yaml_path = Path("config.yaml")
    if not yaml_path.exists():
        return {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    mapped = {SETTINGS_ALIASES.get(key, key): value for key, value in data.items()}
    # API keys are environment-first: empty/null YAML values and YAML values
    # that would shadow a set environment variable are omitted from Settings.
    api_key_env = {
        "openai_api_key": "OPENAI_API_KEY",
        "gemini_api_key": "GEMINI_API_KEY",
    }
    for key, env_name in api_key_env.items():
        if mapped.get(key) in ("", None) or os.getenv(env_name, "").strip():
            mapped.pop(key, None)
    return mapped

def get_default_device() -> str:
    return "cuda" if shutil.which("nvidia-smi") else "cpu"

def get_default_compute_type() -> str:
    return "float16" if shutil.which("nvidia-smi") else "int8"

class Settings(BaseSettings):
    """
    Zentrale Konfigurationsklasse für die gesamte Anwendung.
    Lädt Einstellungen aus der Umgebungsvariable oder der .env Datei.
    """

    # --- API Keys & Provider ---
    gemini_api_key: Optional[SecretStr] = Field(default=None, validation_alias="GEMINI_API_KEY")
    llm_provider: str = Field(default="ollama", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:14b-instruct-q6_K", validation_alias="LLM_MODEL")
    gemini_model: str = Field(default="gemini-2.0-flash", validation_alias="GEMINI_MODEL")
    openrouter_model: Optional[str] = Field(default=None, validation_alias="OPENROUTER_MODEL")
    openai_model: Optional[str] = Field(default=None, validation_alias="OPENAI_MODEL")
    ollama_model: Optional[str] = Field(default=None, validation_alias="OLLAMA_MODEL")
    llama_server_model: Optional[str] = Field(default=None, validation_alias="LLAMA_SERVER_MODEL")

    # --- OpenAI-compatible providers (OpenRouter, llama-server, etc.) ---
    openai_base_url: str = Field(default="https://openrouter.ai/api/v1", validation_alias="OPENAI_BASE_URL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # --- WhisperX / Transcription ---
    whisper_model: str = Field(default="large-v2", validation_alias="WHISPER_MODEL")
    whisper_language: Optional[str] = Field(default="de", validation_alias="WHISPER_LANGUAGE")
    
    # --- CUDA / Device ---
    whisper_device: str = Field(default_factory=get_default_device, validation_alias="WHISPER_DEVICE")
    whisper_device_index: int = Field(default=0, validation_alias="WHISPER_DEVICE_INDEX")
    
    # --- Compute ---
    whisper_compute_type: str = Field(default_factory=get_default_compute_type, validation_alias="WHISPER_COMPUTE_TYPE")
    
    # --- Performance ---
    whisper_batch_size: int = Field(default=16, validation_alias="WHISPER_BATCH_SIZE")
    whisper_threads: int = Field(default=4, validation_alias="WHISPER_THREADS")
    whisper_chunk_size: int = Field(default=30, validation_alias="WHISPER_CHUNK_SIZE")
    max_parallel_renders: int = Field(default=2, validation_alias="MAX_PARALLEL_RENDERS", gt=0)
    
    # --- Alignment ---
    align_model: Optional[str] = Field(default=None, validation_alias="ALIGN_MODEL")
    enable_alignment: bool = Field(default=True, validation_alias="ENABLE_ALIGNMENT")
    return_char_alignments: bool = Field(default=False, validation_alias="RETURN_CHAR_ALIGNMENTS")
    
    # --- VAD ---
    vad_method: str = Field(default="pyannote", validation_alias="VAD_METHOD")
    vad_onset: float = Field(default=0.500, validation_alias="VAD_ONSET")
    vad_offset: float = Field(default=0.363, validation_alias="VAD_OFFSET")
    
    # --- Optional: Diarization ---
    diarization_enabled: bool = Field(default=False, validation_alias="DIARIZATION_ENABLED")
    min_speakers: Optional[int] = Field(default=None, validation_alias="MIN_SPEAKERS")
    max_speakers: Optional[int] = Field(default=None, validation_alias="MAX_SPEAKERS")

    # --- Pfade ---
    raw_dir: Path = Field(default="raw_videos", validation_alias="RAW_DIR")
    out_dir: Path = Field(default="out_clips", validation_alias="OUT_DIR")

    # --- FFmpeg & Video Encoding ---
    ffmpeg_crf: int = Field(default=20, validation_alias="FFMPEG_CRF", ge=0, le=51)
    ffmpeg_preset: str = Field(default="veryfast", validation_alias="FFMPEG_PRESET")
    ffmpeg_audio_bitrate: str = Field(default="192k", validation_alias="FFMPEG_AUDIO_BITRATE")

    # --- Audio Ducking (Musik) ---
    ffmpeg_ducking_threshold: float = Field(
        default=0.08, validation_alias="FFMPEG_DUCKING_THRESHOLD", ge=0.0
    )
    ffmpeg_ducking_ratio: float = Field(default=5.0, validation_alias="FFMPEG_DUCKING_RATIO", gt=0.0)
    ffmpeg_ducking_attack: float = Field(default=50.0, validation_alias="FFMPEG_DUCKING_ATTACK", ge=0.0)
    ffmpeg_ducking_release: float = Field(default=1200.0, validation_alias="FFMPEG_DUCKING_RELEASE", ge=0.0)

    # --- Allgemeine Pfad-Einstellungen ---
    manifest_path: Path = Field(default="sermon_windows.yaml", validation_alias="MANIFEST_PATH")
    input_dir: Optional[Path] = Field(default=None, validation_alias="INPUT_DIR")
    out_root: Optional[Path] = Field(default=None, validation_alias="OUT_ROOT")
    music_dir: Optional[Path] = Field(default=None, validation_alias="MUSIC_DIR")

    # --- Batch / Workflow Flags ---
    whisper: bool = Field(default=True, validation_alias="WHISPER")
    whisper_vad: bool = Field(default=False, validation_alias="WHISPER_VAD")
    no_cudnn: bool = Field(default=False, validation_alias="NO_CUDNN")
    summarizer_script: str = Field(default="srt_summarizer_local.py", validation_alias="SUMMARIZER_SCRIPT")
    summarizer_mode: str = Field(default="fast", validation_alias="SUMMARIZER_MODE")
    summarizer_preset: str = Field(default="reel", validation_alias="SUMMARIZER_PRESET")
    summarizer_timeout: int = Field(default=900, validation_alias="SUMMARIZER_TIMEOUT")
    strict_snap_tol: int = Field(default=8, validation_alias="STRICT_SNAP_TOL")
    min_gap_sec: int = Field(default=12, validation_alias="MIN_GAP_SEC")
    target_count: int = Field(default=6, validation_alias="TARGET_COUNT")
    max_highlights: int = Field(default=8, validation_alias="MAX_HIGHLIGHTS")
    fix_spelling: bool = Field(default=False, validation_alias="FIX_SPELLING")
    llm_titles: bool = Field(default=True, validation_alias="LLM_TITLES")
    build_spans: bool = Field(default=True, validation_alias="BUILD_SPANS")
    spans_script: str = Field(default="build_highlight_spans.py", validation_alias="SPANS_SCRIPT")
    span_min: int = Field(default=15, validation_alias="SPAN_MIN")
    span_max: int = Field(default=20, validation_alias="SPAN_MAX")
    title_min: int = Field(default=3, validation_alias="TITLE_MIN")
    title_max: int = Field(default=6, validation_alias="TITLE_MAX")
    end_boundary: str = Field(default="end", validation_alias="END_BOUNDARY")
    end_forward_tol: int = Field(default=2, validation_alias="END_FORWARD_TOL")
    render: bool = Field(default=True, validation_alias="RENDER")
    confirm_spans: bool = Field(default=True, validation_alias="CONFIRM_SPANS")
    confirm_subtitles: bool = Field(default=True, validation_alias="CONFIRM_SUBTITLES")

    # --- Pipeline Verhalten & Design ---
    subtitle_fontsize: int = Field(default=32, validation_alias="SUBTITLE_FONTSIZE", gt=0)
    overlay_title: bool = Field(default=False, validation_alias="OVERLAY_TITLE")
    overlay_fontsize: int = Field(default=40, validation_alias="OVERLAY_FONTSIZE", gt=0)
    overlay_margin_top: int = Field(default=140, validation_alias="OVERLAY_MARGIN_TOP")
    
    loudnorm: bool = Field(default=False, validation_alias="LOUDNORM")
    fade_in: bool = Field(default=False, validation_alias="FADE_IN")
    fade_in_sec: float = Field(default=0.5, validation_alias="FADE_IN_SEC", ge=0.0)
    fade_out: bool = Field(default=False, validation_alias="FADE_OUT")
    fade_sec: float = Field(default=1.5, validation_alias="FADE_SEC", ge=0.0)
    
    reencode: bool = Field(default=False, validation_alias="REENCODE")
    nvenc: bool = Field(default=False, validation_alias="NVENC")
    blur_bg: bool = Field(default=False, validation_alias="BLUR_BG")
    
    prefix: str = Field(default="highlight_", validation_alias="PREFIX")
    preroll: float = Field(default=0.0, validation_alias="PREROLL", ge=0.0)
    postroll: float = Field(default=0.0, validation_alias="POSTROLL", ge=0.0)
    max_window: float = Field(default=60.0, validation_alias="MAX_WINDOW", gt=0.0)
    srt_min_first: float = Field(default=0.6, validation_alias="SRT_MIN_FIRST")
    
    color_grade: bool = Field(default=False, validation_alias="COLOR_GRADE")
    music_volume: float = Field(default=0.15, validation_alias="MUSIC_VOLUME", ge=0.0, le=1.0)
    
    # --- Face Tracking ---
    face_detect_on_start: bool = Field(default=True, validation_alias="FACE_DETECT_ON_START")
    face_first_robust: bool = Field(default=True, validation_alias="FACE_FIRST_ROBUST")
    face_first_range: float = Field(default=1.0, validation_alias="FACE_FIRST_RANGE")
    face_first_grid: float = Field(default=0.4, validation_alias="FACE_FIRST_GRID")
    face_first_steps: int = Field(default=3, validation_alias="FACE_FIRST_STEPS")
    face_first_sample_rate: float = Field(default=8.0, validation_alias="FACE_FIRST_SAMPLE_RATE", gt=0.0)
    face_use_pose_fallback: bool = Field(default=True, validation_alias="FACE_USE_POSE_FALLBACK")
    face_center: bool = Field(default=True, validation_alias="FACE_CENTER")
    face_track: bool = Field(default=True, validation_alias="FACE_TRACK")
    face_sample_rate: float = Field(default=10.0, validation_alias="FACE_SAMPLE_RATE", gt=0.0)
    face_min_conf: float = Field(default=0.4, validation_alias="FACE_MIN_CONF")
    face_smooth_sec: float = Field(default=2.5, validation_alias="FACE_SMOOTH_SEC")
    face_keys: int = Field(default=15, validation_alias="FACE_KEYS", ge=1)
    face_forward_probe: float = Field(default=0.0, validation_alias="FACE_FORWARD_PROBE")
    fallback_third: str = Field(default="center", validation_alias="FALLBACK_THIRD")
    
    export_srt: bool = Field(default=False, validation_alias="EXPORT_SRT")
    subtitle_style: Optional[str] = Field(default=None, validation_alias="SUBTITLE_STYLE")
    reel: bool = Field(default=False, validation_alias="REEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_assignment=True,
        populate_by_name=True,
    )

settings = Settings(**load_yaml_defaults())
