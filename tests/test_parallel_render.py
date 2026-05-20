import pytest
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from domains.video.renderer import RenderJob, render_clip, run_batch_render, FFmpegError
from core.config import settings

@pytest.fixture
def mock_renderer_run():
    with patch("domains.video.renderer.FFmpegRenderer.run_ffmpeg") as mock:
        yield mock

@pytest.fixture
def mock_settings():
    original_max = settings.max_parallel_renders
    settings.max_parallel_renders = 2
    yield settings
    settings.max_parallel_renders = original_max

def test_render_job_dataclass():
    job = RenderJob(
        input_video=Path("in.mp4"),
        start_ts=10.0,
        duration=5.0,
        out_file=Path("out.mp4"),
        job_id="test-1"
    )
    assert job.job_id == "test-1"
    assert job.duration == 5.0
    assert isinstance(job.kwargs, dict)

def test_parallel_limit_respected(mock_renderer_run, mock_settings, tmp_path):
    """Testet, ob das Parallelitäts-Limit (Semaphore) eingehalten wird."""
    active_threads = []
    max_observed_parallel = 0
    lock = threading.Lock()

    def slow_render(cmd):
        nonlocal max_observed_parallel
        thread_id = threading.get_ident()
        with lock:
            active_threads.append(thread_id)
            max_observed_parallel = max(max_observed_parallel, len(active_threads))
        
        time.sleep(0.5) # Simuliere Arbeit
        
        with lock:
            active_threads.remove(thread_id)
        
        # Erstelle Dummy-Datei für Validierung
        out_path = Path(cmd[-1])
        out_path.write_text("dummy content")

    mock_renderer_run.side_effect = slow_render

    jobs = [
        RenderJob(Path("in.mp4"), 0, 1, tmp_path / f"out_{i}.mp4", job_id=str(i))
        for i in range(5)
    ]

    # Wir setzen max_parallel explizit auf 2
    run_batch_render(jobs, max_parallel=2)

    assert max_observed_parallel <= 2
    assert len(list(tmp_path.glob("out_*.mp4"))) == 5

def test_render_retry_logic(mock_renderer_run, tmp_path):
    """Testet den Retry-Mechanismus bei Fehlern."""
    call_count = 0

    def fail_once(cmd):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("FFmpeg temporary failure")
        out_path = Path(cmd[-1])
        out_path.write_text("some content") # Darf nicht leer sein

    mock_renderer_run.side_effect = fail_once

    from domains.video.renderer import render_with_retry
    job = RenderJob(Path("in.mp4"), 0, 1, tmp_path / "retry.mp4", max_retries=2)
    
    # Sollte beim zweiten Versuch klappen
    result = render_with_retry(job)
    
    assert result.exists()
    assert call_count == 2

def test_render_with_retry_fails_after_max(mock_renderer_run, tmp_path):
    """Testet, ob nach max_retries aufgegeben wird."""
    mock_renderer_run.side_effect = RuntimeError("Permanent Failure")

    from domains.video.renderer import render_with_retry
    job = RenderJob(Path("in.mp4"), 0, 1, tmp_path / "fail.mp4", max_retries=1)

    with pytest.raises(FFmpegError):
        render_with_retry(job)
    
    assert mock_renderer_run.call_count == 2 # Initial + 1 Retry

def test_batch_render_handles_partial_failures(mock_renderer_run, tmp_path):
    """Testet, ob das Batch-Rendering bei einzelnen Fehlern weitermacht."""
    def fail_job_2(cmd):
        if "out_1.mp4" in str(cmd[-1]):
            raise RuntimeError("Job 2 failed")
        out_path = Path(cmd[-1])
        out_path.write_text("success")

    mock_renderer_run.side_effect = fail_job_2

    jobs = [
        RenderJob(Path("in.mp4"), 0, 1, tmp_path / f"out_{i}.mp4", job_id=str(i), max_retries=0)
        for i in range(3)
    ]

    results = run_batch_render(jobs, max_parallel=3)

    assert len(results) == 2
    assert (tmp_path / "out_0.mp4").exists()
    assert not (tmp_path / "out_1.mp4").exists()
    assert (tmp_path / "out_2.mp4").exists()
