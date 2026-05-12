import pytest
from pydantic import ValidationError
from core.config import Settings

def test_default_settings():
    """Prüft, ob die Default-Settings gültig sind."""
    s = Settings()
    assert s.ffmpeg_crf == 20
    assert s.music_volume == 0.15

def test_invalid_crf():
    """Prüft, ob ungültige CRF-Werte abgelehnt werden."""
    with pytest.raises(ValidationError):
        Settings(ffmpeg_crf=-1)
    with pytest.raises(ValidationError):
        Settings(ffmpeg_crf=52)

def test_invalid_music_volume():
    """Prüft, ob ungültige Lautstärken abgelehnt werden."""
    with pytest.raises(ValidationError):
        Settings(music_volume=-0.1)
    with pytest.raises(ValidationError):
        Settings(music_volume=1.1)

def test_invalid_face_sample_rate():
    """Prüft, ob face_sample_rate > 0 sein muss."""
    with pytest.raises(ValidationError):
        Settings(face_sample_rate=0)
    with pytest.raises(ValidationError):
        Settings(face_sample_rate=-1.0)

def test_invalid_max_window():
    """Prüft, ob max_window > 0 sein muss."""
    with pytest.raises(ValidationError):
        Settings(max_window=0)
    with pytest.raises(ValidationError):
        Settings(max_window=-10.0)

def test_invalid_face_keys():
    """Prüft, ob face_keys >= 1 sein muss."""
    with pytest.raises(ValidationError):
        Settings(face_keys=0)
