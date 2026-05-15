import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class ArtifactError(Exception):
    """Exception raised for errors in artifact management."""
    pass


def sanitize_filename(name: str) -> str:
    """Removes or replaces invalid characters for filenames."""
    return re.sub(r'[^\w\-_\.]', '_', name)


def save_json(path: Path, data: Union[Dict[str, Any], list], overwrite: bool = True) -> Path:
    """Saves a dictionary or list to a JSON file."""
    if path.exists() and not overwrite:
        logger.debug(f"File {path.name} already exists. Skipping save.")
        return path
        
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved JSON artifact to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save JSON to {path}: {e}")
        raise ArtifactError(f"IO Error saving JSON: {e}") from e


def load_json(path: Path) -> Union[Dict[str, Any], list]:
    """Loads a JSON file and returns its content."""
    if not path.exists():
        raise ArtifactError(f"File not found: {path}")
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        raise ArtifactError(f"Invalid JSON format: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load JSON from {path}: {e}")
        raise ArtifactError(f"IO Error loading JSON: {e}") from e


def save_text(path: Path, content: str, overwrite: bool = True) -> Path:
    """Saves string content to a text file."""
    if path.exists() and not overwrite:
        logger.debug(f"File {path.name} already exists. Skipping save.")
        return path
        
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Saved text artifact to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save text to {path}: {e}")
        raise ArtifactError(f"IO Error saving text: {e}") from e


def append_log(path: Path, message: str) -> Path:
    """Appends a timestamped message to a log file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat()
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
        return path
    except Exception as e:
        logger.error(f"Failed to append to log {path}: {e}")
        raise ArtifactError(f"IO Error appending log: {e}") from e


class ArtifactManager:
    """
    Manages intermediate artifacts for a pipeline run to ensure reproducibility
    and support idempotent re-runs.
    """
    
    def __init__(self, base_work_dir: Path, video_path: Path, force_rebuild: bool = False):
        self.base_work_dir = Path(base_work_dir)
        self.video_path = Path(video_path)
        self.force_rebuild = force_rebuild
        
        self.stem = sanitize_filename(self.video_path.stem)
        self.work_dir = self.base_work_dir / self.stem
        self._ensure_directory_structure()
        
    def _ensure_directory_structure(self):
        """Creates the necessary directory structure."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
    def get_path(self, artifact_name: str) -> Path:
        """Gets the absolute path for a specific artifact name."""
        return self.work_dir / artifact_name
        
    def exists(self, artifact_name: str) -> bool:
        """Checks if an artifact exists and we are not forced to rebuild."""
        path = self.get_path(artifact_name)
        return path.exists() and not self.force_rebuild

    def save_metadata(self, metadata: dict) -> None:
        """Saves run metadata for version tracking."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "video_file": self.video_path.name,
            **metadata
        }
        save_json(self.get_path("metadata.json"), data)
        
    def log_step(self, step_name: str, status: str, duration_sec: Optional[float] = None, details: str = ""):
        """Logs the progress of a specific pipeline step."""
        msg = f"STEP [{step_name}] | STATUS: {status}"
        if duration_sec is not None:
            msg += f" | DURATION: {duration_sec:.2f}s"
        if details:
            msg += f" | DETAILS: {details}"
            
        append_log(self.get_path("pipeline.log"), msg)
        logger.info(msg)
