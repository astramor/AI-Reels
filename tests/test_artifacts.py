import pytest
import json
from pathlib import Path
from core.artifacts import (
    ArtifactManager,
    ArtifactError,
    save_json,
    load_json,
    save_text,
    append_log,
    sanitize_filename
)

def test_sanitize_filename():
    assert sanitize_filename("Video_File-Name 123.mp4") == "Video_File-Name_123.mp4"
    assert sanitize_filename("!@#$%^&*()_+-=") == "____________-_"

def test_save_and_load_json(tmp_path):
    file_path = tmp_path / "test.json"
    data = {"key": "value", "list": [1, 2, 3]}
    
    # Save
    saved_path = save_json(file_path, data)
    assert saved_path == file_path
    assert file_path.exists()
    
    # Load
    loaded_data = load_json(file_path)
    assert loaded_data == data

def test_save_json_overwrite(tmp_path):
    file_path = tmp_path / "test.json"
    save_json(file_path, {"a": 1})
    
    # Should overwrite
    save_json(file_path, {"b": 2})
    assert load_json(file_path) == {"b": 2}
    
    # Should NOT overwrite
    save_json(file_path, {"c": 3}, overwrite=False)
    assert load_json(file_path) == {"b": 2}

def test_load_json_invalid(tmp_path):
    file_path = tmp_path / "invalid.json"
    file_path.write_text("not a json string")
    
    with pytest.raises(ArtifactError, match="Invalid JSON format"):
        load_json(file_path)

def test_load_json_missing(tmp_path):
    with pytest.raises(ArtifactError, match="File not found"):
        load_json(tmp_path / "missing.json")

def test_save_text(tmp_path):
    file_path = tmp_path / "test.txt"
    save_text(file_path, "Hello World")
    
    assert file_path.read_text() == "Hello World"

def test_append_log(tmp_path):
    log_path = tmp_path / "test.log"
    append_log(log_path, "First line")
    append_log(log_path, "Second line")
    
    content = log_path.read_text()
    assert "First line" in content
    assert "Second line" in content

def test_artifact_manager_paths(tmp_path):
    video = Path("/some/path/my video file!.mp4")
    manager = ArtifactManager(base_work_dir=tmp_path, video_path=video)
    
    assert manager.stem == "my_video_file_"
    assert manager.work_dir == tmp_path / "my_video_file_"
    assert manager.get_path("data.json") == tmp_path / "my_video_file_" / "data.json"

def test_artifact_manager_exists_logic(tmp_path):
    manager = ArtifactManager(base_work_dir=tmp_path, video_path=Path("test.mp4"))
    test_file = manager.get_path("test.txt")
    
    assert not manager.exists("test.txt")
    
    save_text(test_file, "content")
    assert manager.exists("test.txt")
    
    # Force rebuild should ignore existing file
    manager.force_rebuild = True
    assert not manager.exists("test.txt")

def test_artifact_manager_logging(tmp_path):
    manager = ArtifactManager(base_work_dir=tmp_path, video_path=Path("test.mp4"))
    manager.log_step("TestStep", "SUCCESS", duration_sec=5.12, details="All good")
    
    log_path = manager.get_path("pipeline.log")
    assert log_path.exists()
    content = log_path.read_text()
    assert "STEP [TestStep] | STATUS: SUCCESS | DURATION: 5.12s | DETAILS: All good" in content
