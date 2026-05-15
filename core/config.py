from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import shutil

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
    gemini_api_key: Optional[SecretStr] = Field(default=None, env="GEMINI_API_KEY")
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:14b-instruct-q6_K", env="LLM_MODEL")
    gemini_model: str = Field(default="gemini-2.0-flash", env="GEMINI_MODEL")

    # --- WhisperX / Transcription ---
    whisper_model: str = Field(default="large-v2", env="WHISPER_MODEL")
    whisper_language: Optional[str] = Field(default=None, env="WHISPER_LANGUAGE")
    
    # --- CUDA / Device ---
    whisper_device: str = Field(default_factory=get_default_device, env="WHISPER_DEVICE")
    whisper_device_index: int = Field(default=0, env="WHISPER_DEVICE_INDEX")
    
    # --- Compute ---
    whisper_compute_type: str = Field(default_factory=get_default_compute_type, env="WHISPER_COMPUTE_TYPE")
    
    # --- Performance ---
    whisper_batch_size: int = Field(default=16, env="WHISPER_BATCH_SIZE")
    whisper_threads: int = Field(default=4, env="WHISPER_THREADS")
    whisper_chunk_size: int = Field(default=30, env="WHISPER_CHUNK_SIZE")
    
    # --- Alignment ---
    align_model: Optional[str] = Field(default=None, env="ALIGN_MODEL")
    enable_alignment: bool = Field(default=True, env="ENABLE_ALIGNMENT")
    return_char_alignments: bool = Field(default=False, env="RETURN_CHAR_ALIGNMENTS")
    
    # --- VAD ---
    vad_method: str = Field(default="pyannote", env="VAD_METHOD")
    vad_onset: float = Field(default=0.500, env="VAD_ONSET")
    vad_offset: float = Field(default=0.363, env="VAD_OFFSET")
    
    # --- Optional: Diarization ---
    diarization_enabled: bool = Field(default=False, env="DIARIZATION_ENABLED")
    min_speakers: Optional[int] = Field(default=None, env="MIN_SPEAKERS")
    max_speakers: Optional[int] = Field(default=None, env="MAX_SPEAKERS")

    # --- Pfade ---
    raw_dir: Path = Field(default="raw_videos", env="RAW_DIR")
    out_dir: Path = Field(default="out_clips", env="OUT_DIR")

    # --- FFmpeg & Video Encoding ---
    ffmpeg_crf: int = Field(default=20, env="FFMPEG_CRF", ge=0, le=51)
    ffmpeg_preset: str = Field(default="veryfast", env="FFMPEG_PRESET")
    ffmpeg_audio_bitrate: str = Field(default="192k", env="FFMPEG_AUDIO_BITRATE")

    # --- Audio Ducking (Musik) ---
    ffmpeg_ducking_threshold: float = Field(
        default=0.08, env="FFMPEG_DUCKING_THRESHOLD", ge=0.0
    )
    ffmpeg_ducking_ratio: float = Field(default=5.0, env="FFMPEG_DUCKING_RATIO", gt=0.0)
    ffmpeg_ducking_attack: float = Field(default=50.0, env="FFMPEG_DUCKING_ATTACK", ge=0.0)
    ffmpeg_ducking_release: float = Field(default=1200.0, env="FFMPEG_DUCKING_RELEASE", ge=0.0)

    # --- Allgemeine Pfad-Einstellungen ---
    manifest_path: Path = Field(default="sermon_windows.yaml", env="MANIFEST_PATH")

    # --- Pipeline Verhalten & Design ---
    subtitle_fontsize: int = Field(default=32, env="SUBTITLE_FONTSIZE", gt=0)
    overlay_title: bool = Field(default=False, env="OVERLAY_TITLE")
    overlay_fontsize: int = Field(default=40, env="OVERLAY_FONTSIZE", gt=0)
    overlay_margin_top: int = Field(default=140, env="OVERLAY_MARGIN_TOP")
    
    loudnorm: bool = Field(default=False, env="LOUDNORM")
    fade_in: bool = Field(default=False, env="FADE_IN")
    fade_in_sec: float = Field(default=0.5, env="FADE_IN_SEC", ge=0.0)
    fade_out: bool = Field(default=False, env="FADE_OUT")
    fade_sec: float = Field(default=1.5, env="FADE_SEC", ge=0.0)
    
    reencode: bool = Field(default=False, env="REENCODE")
    nvenc: bool = Field(default=False, env="NVENC")
    blur_bg: bool = Field(default=False, env="BLUR_BG")
    
    prefix: str = Field(default="highlight_", env="PREFIX")
    preroll: float = Field(default=0.0, env="PREROLL", ge=0.0)
    postroll: float = Field(default=0.0, env="POSTROLL", ge=0.0)
    max_window: float = Field(default=60.0, env="MAX_WINDOW", gt=0.0)
    srt_min_first: float = Field(default=0.6, env="SRT_MIN_FIRST")
    
    color_grade: bool = Field(default=False, env="COLOR_GRADE")
    music_volume: float = Field(default=0.15, env="MUSIC_VOLUME", ge=0.0, le=1.0)
    
    # --- Face Tracking ---
    face_detect_on_start: bool = Field(default=True, env="FACE_DETECT_ON_START")
    face_first_robust: bool = Field(default=True, env="FACE_FIRST_ROBUST")
    face_first_range: float = Field(default=1.0, env="FACE_FIRST_RANGE")
    face_first_grid: float = Field(default=0.4, env="FACE_FIRST_GRID")
    face_first_steps: int = Field(default=3, env="FACE_FIRST_STEPS")
    face_first_sample_rate: float = Field(default=8.0, env="FACE_FIRST_SAMPLE_RATE", gt=0.0)
    face_use_pose_fallback: bool = Field(default=True, env="FACE_USE_POSE_FALLBACK")
    face_center: bool = Field(default=True, env="FACE_CENTER")
    face_track: bool = Field(default=True, env="FACE_TRACK")
    face_sample_rate: float = Field(default=10.0, env="FACE_SAMPLE_RATE", gt=0.0)
    face_min_conf: float = Field(default=0.4, env="FACE_MIN_CONF")
    face_smooth_sec: float = Field(default=2.5, env="FACE_SMOOTH_SEC")
    face_keys: int = Field(default=15, env="FACE_KEYS", ge=1)
    face_forward_probe: float = Field(default=0.0, env="FACE_FORWARD_PROBE")
    fallback_third: str = Field(default="center", env="FALLBACK_THIRD")
    
    export_srt: bool = Field(default=False, env="EXPORT_SRT")
    subtitle_style: Optional[str] = Field(default=None, env="SUBTITLE_STYLE")
    reel: bool = Field(default=False, env="REEL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
