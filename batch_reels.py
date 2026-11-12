#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, subprocess, shlex, json, yaml, shutil, re, signal, math
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import srt
from core.time_utils import hms_to_seconds, parse_time_to_seconds, seconds_to_hms
from core.config import settings, SETTINGS_ALIASES
from core.span_contract import (
    ClipSpan,
    SpanValidationError,
    ensure_no_finalized_overlap,
    parse_finalized_flag,
    validate_span,
)
from domains.highlights.parser import HighlightParser
from domains.video.renderer import gpu_semaphore
from core.commands import run_command, ProcessRegistry

# --- Logging Configuration ---
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()  # Standard-Handler entfernen
logger.add(sys.stderr, level="INFO")  # Konsolenausgabe
logger.add(
    LOG_DIR / "reels_{time}.log",
    rotation="50 MB",
    retention="10 days",
    level="INFO",
    encoding="utf-8"
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".mp3", ".wav", ".webm"}


def signal_handler(sig, frame):
    """
    Abbruch-Handler für SIGINT und SIGTERM.
    Beendet alle laufenden Subprozesse.
    """
    signame = signal.Signals(sig).name
    logger.warning(f"🛑 Abbruch durch {signame} empfangen...")
    ProcessRegistry.terminate_all()
    logger.info("👋 Programm beendet.")
    sys.exit(0)


# Registriere Handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run(cmd, *, env=None, dry_run=False, cwd=None):
    if dry_run:
        logger.info(f"▶ (DRY-RUN) {' '.join(shlex.quote(str(c)) for c in cmd)}")
        return 0
    
    try:
        logger.info(f"▶ {' '.join(shlex.quote(str(c)) for c in cmd)}")
        if env and env.get("CT2_USE_CUDNN") == "0":
            logger.debug("  (ENV) CT2_USE_CUDNN=0")
            
        run_command(cmd, cwd=cwd, env=env, capture_output=False)
        return 0
    except Exception as e:
        logger.error(f"❌ Command failed: {e}")
        raise SystemExit(1)


# --- Helper ---

def parse_highlights_md(md_path: Path):
    if not md_path.exists():
        return []
    parser = HighlightParser()
    highlights = parser.load_highlights_from_md(md_path)
    spans = []
    for h in highlights:
        spans.append({
            "start": h.start,
            "end": h.end if h.end is not None else (h.start + 60),
            "title": h.label,
            "finalized": h.finalized,
        })
    ensure_no_finalized_overlap(spans)
    return spans


def parse_spans_json(json_path: Path):
    """Parse reel_spans.json (or a HighlightPayload) into numeric spans."""
    if not json_path.exists():
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Fehler beim Parsen der Spans-JSON: {e}")
        return []

    items = data.get("highlights", data) if isinstance(data, dict) else data
    # Legacy fallback rule:
    # - reel_spans.json without metadata -> finalized
    # - any other artifact without metadata -> candidate
    # Explicit finalized metadata always wins, even after renaming/copying.
    legacy_finalized = json_path.name == "reel_spans.json"
    spans = []
    for item in items:
        if not isinstance(item, dict):
            continue
        start_val = item.get("start", 0.0)
        end_val = item.get("end", parse_time_to_seconds(start_val) + 60.0)
        try:
            finalized = parse_finalized_flag(
                item.get("finalized"), legacy_finalized
            )
        except SpanValidationError as e:
            logger.warning(
                f"⚠️ Span mit Start {start_val!r} verworfen: {e}"
            )
            continue
        spans.append({
            "start": parse_time_to_seconds(start_val),
            "end": parse_time_to_seconds(end_val),
            "title": item.get("title", "Highlight"),
            "finalized": finalized,
        })
    ensure_no_finalized_overlap(spans)
    return spans


def probe_media_duration(video_path: Path) -> float:
    """Return the source media duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_format",
        "-print_format", "json",
        str(video_path),
    ]
    result = run_command(cmd)
    info = json.loads(result.stdout)
    duration = info.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError("Konnte Video-Dauer nicht ermitteln")
    duration = float(duration)
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(f"Ungültige Video-Dauer: {duration!r}")
    return duration


def get_json_slice(json_path, start_sec, end_sec):
    if not json_path or not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Fehler JSON: {e}")
        return None

    new_segments = []
    s_buf = start_sec - 0.05
    e_buf = end_sec + 0.2

    for seg in data.get("segments", []):
        if seg["end"] < s_buf:
            continue
        if seg["start"] > e_buf:
            continue

        valid_words = []
        for w in seg.get("words", []):
            if w["end"] > s_buf and w["start"] < e_buf:
                valid_words.append(w.copy())

        if valid_words:
            new_seg = seg.copy()
            new_seg["words"] = valid_words
            new_seg["start"] = valid_words[0]["start"]
            new_seg["end"] = valid_words[-1]["end"]
            new_seg["text"] = " ".join([w["word"].strip() for w in valid_words])
            new_segments.append(new_seg)

    return {"segments": new_segments}


def get_times_from_srt(subs):
    if not subs:
        return None, None
    return subs[0].start.total_seconds(), subs[-1].end.total_seconds()


def load_srt_file(path):
    return list(srt.parse(path.read_text(encoding="utf-8")))


def get_srt_slice(subs, start_sec, end_sec):
    slice_subs = []
    start_td = timedelta(seconds=start_sec - 0.2)
    end_td = timedelta(seconds=end_sec + 0.2)
    for sub in subs:
        if sub.end > start_td and sub.start < end_td:
            new_sub = srt.Subtitle(
                index=len(slice_subs) + 1,
                start=sub.start,
                end=sub.end,
                content=sub.content,
            )
            slice_subs.append(new_sub)
    return slice_subs


def json_to_srt(json_data):
    slice_subs = []
    for i, seg in enumerate(json_data.get("segments", [])):
        start_td = timedelta(seconds=seg["start"])
        end_td = timedelta(seconds=seg["end"])
        content = seg.get("text", "").strip()
        slice_subs.append(
            srt.Subtitle(index=i + 1, start=start_td, end=end_td, content=content)
        )
    return slice_subs


def sanitize_stem(raw_stem: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", raw_stem)


def find_whisper_json(out_dir: Path, stem: str) -> Path:
    cand = out_dir / f"{stem}.json"
    if cand.exists():
        return cand
    return None


def find_whisper_srt(out_dir: Path, stem: str) -> Path:
    cand = out_dir / f"{stem}.srt"
    if cand.exists():
        return cand
    return None


def render_clip(item, vid, reels_out, summ_dir, stem, args):
    """
    Rendert einen einzelnen Clip asynchron über die Smart-Pipeline.
    """
    idx = item["idx"]
    current_start = item["start"]
    current_end = item["end"]
    final_srt_arg = item.get("final_srt_arg")
    final_json_arg = item.get("final_json_arg")

    tmp_span_json = summ_dir / f"tmp_span_{idx}.json"
    try:
        payload = {
            "highlights": [
                {
                    "title": item["title"],
                    "start": current_start,
                    "end": current_end,
                    "reason": item.get("reason", "Batch clip"),
                    "finalized": item.get("finalized", True),
                }
            ]
        }
        tmp_span_json.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            "video_highlight_pipeline_smart.py",
            "--video",
            str(vid),
            "--spans_json",
            str(tmp_span_json),
            "--out_dir",
            str(reels_out),
            "--prefix",
            f"{stem}_{idx:02d}_",
        ]

        if final_srt_arg:
            cmd += ["--srt_input", str(final_srt_arg), "--subtitles", str(final_srt_arg)]
        if final_json_arg:
            cmd += ["--transcription_json", str(final_json_arg)]
        if args.music_dir:
            cmd += ["--music_dir", str(args.music_dir)]
        if args.nvenc:
            cmd.append("--nvenc")

        cmd += [
            "--face-sample-rate",
            str(args.face_sample_rate),
            "--face-min-conf",
            str(args.face_min_conf),
            "--face-smooth-sec",
            str(args.face_smooth_sec),
            "--face-keys",
            str(args.face_keys),
            "--face-first-range",
            str(args.face_first_range),
            "--face-first-grid",
            str(args.face_first_grid),
            "--face-first-steps",
            str(args.face_first_steps),
            "--face-first-sample-rate",
            str(args.face_first_sample_rate),
        ]

        logger.info(f"🎬 Starte Rendering für Clip {idx}: {item['title']}")
        run(cmd, dry_run=args.dry_run)
        logger.info(f"✅ Clip {idx} erfolgreich gerendert.")
    except Exception as e:
        logger.info(f"❌ Fehler beim Rendern von Clip {idx}: {e}")
    finally:
        if tmp_span_json.exists():
            tmp_span_json.unlink()


def process_pre_cuts(
    raw_dir: Path,
    input_dir: Path,
    manifest_path: Path,
    out_whisper_root: Path,
    force: bool = False,
):
    if not manifest_path.exists():
        return
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    windows = data.get("sermon_windows", {})
    if not windows:
        return
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    for vid in raw_dir.glob("*"):
        if vid.suffix.lower() not in VIDEO_EXTS:
            continue
        stem = sanitize_stem(vid.stem)
        if stem in windows:
            entry = windows[stem]
            start_str = entry.get("start", "00:00:00")
            end_str = entry.get("end", None)
            if not end_str:
                continue
            # Audio-only WAV extraction avoids keyframe/stream-copy timestamp drift.
            target_file = input_dir / f"{vid.stem}.wav"
            if target_file.exists() and not force:
                continue
            s_sec = parse_time_to_seconds(start_str)
            e_sec = parse_time_to_seconds(end_str)
            duration = e_sec - s_sec
            if duration <= 0:
                continue
            logger.info(f"✂️  Schneide Predigt: {stem} (Start: {start_str}, Dauer: {duration}s)")

            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(start_str), "-to", str(end_str), "-i", str(vid),
                "-vn", "-af", "highpass=f=80,lowpass=f=8000",
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                str(target_file),
            ]
            run(cmd)
            whisper_dir = out_whisper_root / stem
            if whisper_dir.exists():
                shutil.rmtree(whisper_dir)


def setup_argparse(defaults):
    ap = argparse.ArgumentParser()
    ap.set_defaults(**defaults)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw_dir", default="raw_videos")
    ap.add_argument("--input_dir", required=False)
    ap.add_argument("--out_root", default=".")
    ap.add_argument("--manifest", default="sermon_windows.yaml")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--whisper", action="store_true")
    ap.add_argument("--whisper-model", default="large-v3")
    ap.add_argument("--whisper-device", default="cuda")
    ap.add_argument("--whisper-batch-size", type=int, default=16)
    ap.add_argument("--whisper-language", default="de")
    ap.add_argument("--whisper-compute-type", default="float16")
    ap.add_argument("--whisper-vad", action="store_true")
    ap.add_argument("--no-cudnn", action="store_true")

    ap.add_argument("--summarizer-script", default="srt_summarizer_local.py")
    ap.add_argument("--summarizer-preset", default="reel")
    ap.add_argument("--summarizer-timeout", type=int, default=900)
    ap.add_argument("--strict-snap-tol", type=int, default=8)
    ap.add_argument("--min-gap-sec", type=int, default=12)
    ap.add_argument("--max-highlights", type=int, default=12)
    ap.add_argument("--sermon-start")
    ap.add_argument("--sermon-end")
    ap.add_argument("--auto-sermon", action="store_true")
    ap.add_argument("--min-sermon-minutes", type=int, default=10)
    ap.add_argument("--fix-spelling", action="store_true")

    ap.add_argument("--build-spans", action="store_true")
    ap.add_argument("--spans-script", default="build_highlight_spans.py")
    ap.add_argument("--target-count", type=int, default=6)
    ap.add_argument("--span-min", type=int, default=15)
    ap.add_argument("--span-max", type=int, default=30)
    ap.add_argument("--title-min", type=int, default=3)
    ap.add_argument("--title-max", type=int, default=10)
    ap.add_argument("--end-boundary", default="end")
    ap.add_argument("--end-forward-tol", type=int, default=2)
    ap.add_argument("--llm-titles", action="store_true")
    ap.add_argument("--church-yaml", default="besuchen.md")

    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-provider", default=None)
    ap.add_argument("--gemini-api-key", default=None)
    ap.add_argument("--openrouter-api-key", default=None)
    ap.add_argument("--gemini-model", default=None)

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--confirm-spans", action="store_true")
    ap.add_argument("--confirm-subtitles", action="store_true")

    ap.add_argument("--reel", action="store_true")
    ap.add_argument("--loudnorm", action="store_true")
    ap.add_argument("--fade-in", action="store_true")
    ap.add_argument("--fade-in-sec", type=float, default=0.5)
    ap.add_argument("--fade-out", action="store_true")
    ap.add_argument("--fade-sec", type=float, default=1.5)
    ap.add_argument("--overlay-title", action="store_true")
    ap.add_argument("--overlay-fontsize", type=int, default=40)
    ap.add_argument("--overlay-margin-top", type=int, default=140)
    ap.add_argument("--subtitle-fontsize", type=int, default=32)
    ap.add_argument("--subtitle-style", default=None)
    ap.add_argument("--export-srt", action="store_true")
    ap.add_argument("--reencode", action="store_true")
    ap.add_argument("--nvenc", action="store_true")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--audio-bitrate", default="192k")
    ap.add_argument("--blur-bg", action="store_true")

    ap.add_argument("--color-grade", action="store_true")
    ap.add_argument("--music-dir")
    ap.add_argument("--music-volume", type=float)
    ap.add_argument("--ducking-threshold", type=float, default=0.08)
    ap.add_argument("--ducking-ratio", type=float, default=5.0)
    ap.add_argument("--ducking-attack", type=float, default=50.0)
    ap.add_argument("--ducking-release", type=float, default=1200.0)

    ap.add_argument("--preroll", type=float, default=0.0)
    ap.add_argument("--postroll", type=float, default=0.0)
    ap.add_argument("--max-window", type=float, default=60.0)
    ap.add_argument("--srt-min-first", type=float, default=0.6)

    ap.add_argument("--face-center", action="store_true")
    ap.add_argument("--face-track", action="store_true")
    ap.add_argument("--face-sample-rate", type=float, default=3.0)
    ap.add_argument("--face-min-conf", type=float, default=0.6)
    ap.add_argument("--face-smooth-sec", type=float, default=0.6)
    ap.add_argument("--face-keys", type=int, default=10)
    ap.add_argument("--face-detect-on-start", action="store_true")
    ap.add_argument("--face-first-robust", action="store_true")
    ap.add_argument("--face-first-range", type=float, default=1.0)
    ap.add_argument("--face-first-grid", type=float, default=0.4)
    ap.add_argument("--face-first-steps", type=int, default=3)
    ap.add_argument("--face-first-sample-rate", type=float, default=8.0)
    ap.add_argument("--face-use-pose-fallback", action="store_true")
    ap.add_argument("--face-forward-probe", type=float, default=0.0)
    ap.add_argument("--fallback-third", default="none")
    return ap


def execute_transcription(vid, stem, out_whisper_dir, args):
    logger.info("Starte WhisperX...")
    w_cmd = [
        "whisperx",
        str(vid),
        "--model",
        settings.whisper_model,
        "--output_dir",
        str(out_whisper_dir),
        "--output_format",
        "all",
        "--device",
        settings.whisper_device,
        "--compute_type",
        settings.whisper_compute_type,
        "--batch_size",
        str(settings.whisper_batch_size),
        "--vad_onset",
        str(settings.vad_onset),
        "--vad_offset",
        str(settings.vad_offset),
    ]
    if settings.whisper_language:
        w_cmd.extend(["--language", settings.whisper_language])
    if settings.align_model:
        w_cmd.extend(["--align_model", settings.align_model])
    if settings.diarization_enabled:
        w_cmd.append("--diarize")
        if settings.min_speakers:
            w_cmd.extend(["--min_speakers", str(settings.min_speakers)])
        if settings.max_speakers:
            w_cmd.extend(["--max_speakers", str(settings.max_speakers)])
            
    env = os.environ.copy()
    if args.no_cudnn:
        env["CT2_USE_CUDNN"] = "0"
    
    run(w_cmd, env=env, dry_run=args.dry_run)


def execute_summarizer(srt_file, summ_dir, predigt_srt, defaults, args):
    if (
        not isinstance(args.max_highlights, int)
        or isinstance(args.max_highlights, bool)
        or args.max_highlights <= 0
    ):
        raise ValueError("--max-highlights muss eine positive Ganzzahl sein")

    logger.info("Starte Summarizer...")
    s_cmd = [
        sys.executable,
        args.summarizer_script,
        "--srt",
        str(srt_file),
        "--out_dir",
        str(summ_dir),
        "--export-sermon-srt",
        str(predigt_srt),
    ]
    if args.fix_spelling:
        s_cmd.append("--fix-spelling")
    if args.auto_sermon:
        s_cmd += [
            "--auto-sermon",
            "--min-sermon-minutes",
            str(args.min_sermon_minutes),
        ]
    if args.llm_provider:
        s_cmd += ["--llm-provider", args.llm_provider]
    if getattr(args, "gemini_api_key", None):
        s_cmd += ["--gemini-api-key", args.gemini_api_key]
    if getattr(args, "openrouter_api_key", None):
        s_cmd += ["--openrouter-api-key", args.openrouter_api_key]

    provider = (args.llm_provider or settings.llm_provider).lower()
    if provider in ("gemini", "gemini-api") and args.gemini_model:
        s_cmd += ["--model", args.gemini_model]
    elif args.llm_model:
        s_cmd += ["--model", args.llm_model]

    if args.sermon_start:
        s_cmd += ["--sermon-start", args.sermon_start]
    if args.sermon_end:
        s_cmd += ["--sermon-end", args.sermon_end]

    s_cmd += ["--max-highlights", str(args.max_highlights)]
    if args.strict_snap_tol:
        s_cmd += ["--strict-snap-tol", str(args.strict_snap_tol)]

    if "summarizer_preset" in defaults:
        s_cmd += ["--preset", defaults["summarizer_preset"]]
    run(s_cmd, dry_run=args.dry_run)


def execute_span_builder(hl_md, predigt_srt, spans_md, spans_json, args, transcription_json=None):
    logger.info("Berechne Spans & Titel...")
    b_cmd = [
        sys.executable,
        args.spans_script,
        "--highlights",
        str(hl_md),
        "--srt",
        str(predigt_srt),
        "--out-md",
        str(spans_md),
        "--out-json",
        str(spans_json),
        "--target-count",
        str(args.target_count),
        "--target-min",
        str(args.span_min),
        "--target-max",
        str(args.span_max),
        "--title-min",
        str(args.title_min),
        "--title-max",
        str(args.title_max),
    ]
    if getattr(args, "llm_provider", None):
        b_cmd += ["--llm-provider", args.llm_provider]
    if getattr(args, "llm_model", None):
        b_cmd += ["--llm-model", args.llm_model]
    if getattr(args, "gemini_model", None):
        b_cmd += ["--gemini-model", args.gemini_model]
    if getattr(args, "gemini_api_key", None):
        b_cmd += ["--gemini-api-key", args.gemini_api_key]
    if getattr(args, "openrouter_api_key", None):
        b_cmd += ["--openrouter-api-key", args.openrouter_api_key]
    if transcription_json and Path(transcription_json).exists():
        b_cmd += ["--transcription-json", str(transcription_json)]
    if args.llm_titles:
        b_cmd.append("--llm-titles")
    else:
        b_cmd.append("--no-llm")
    if args.church_yaml:
        b_cmd += ["--church-yaml", str(args.church_yaml)]
    run(b_cmd, dry_run=args.dry_run)


def execute_render_orchestration(vid, stem, summ_dir, json_file, predigt_srt, spans_json, out_reels, args):
    try:
        spans_content = parse_spans_json(spans_json)
    except SpanValidationError as e:
        logger.error(f"❌ Spans-JSON ungültig: {e}")
        return

    reels_out = out_reels / stem
    reels_out.mkdir(parents=True, exist_ok=True)

    render_queue = []
    all_subs = load_srt_file(predigt_srt)

    try:
        media_duration = probe_media_duration(vid)
    except Exception as e:
        logger.warning(f"⚠️ ffprobe-Dauer nicht ermittelbar: {e}")
        proxy = all_subs[-1].end.total_seconds() if all_subs else None
        if proxy is not None and math.isfinite(proxy) and proxy > 0:
            logger.warning(
                f"⚠️ Nutze SRT-Ende als Mediendauer-Proxy: {proxy:.3f}s"
            )
            media_duration = proxy
        else:
            logger.warning(
                "⚠️ Mediendauer unbekannt; keine Media-Boundary-Prüfung"
            )
            media_duration = None

    for idx, span in enumerate(spans_content, 1):
        try:
            validated = validate_span(
                ClipSpan(
                    start=span["start"],
                    end=span["end"],
                    title=span["title"],
                    finalized=span.get("finalized", True),
                ),
                media_duration=media_duration,
                min_duration=settings.span_min,
                max_duration=settings.max_window,
            )
        except SpanValidationError as e:
            logger.warning(f"⚠️ Clip {idx} verworfen: {e}")
            continue

        c_start = validated.start
        c_end = validated.end
        label = validated.title
        temp_srt_path = summ_dir / f"temp_edit_reel_{idx}.srt"
        temp_json_path = summ_dir / f"temp_edit_reel_{idx}.json"

        if args.confirm_subtitles:
            if json_file:
                mini_json = get_json_slice(json_file, c_start, c_end)
                if mini_json:
                    with open(temp_json_path, "w", encoding="utf-8") as f:
                        json.dump(mini_json, f, indent=2, ensure_ascii=False)
                    slice_subs = json_to_srt(mini_json)
                    temp_srt_path.write_text(srt.compose(slice_subs), encoding="utf-8")
            else:
                slice_subs = get_srt_slice(all_subs, c_start, c_end)
                temp_srt_path.write_text(srt.compose(slice_subs), encoding="utf-8")

        render_queue.append({
            "idx": idx,
            "label": label,
            "start": c_start,
            "end": c_end,
            "temp_srt": temp_srt_path,
            "temp_json": temp_json_path,
            "title": label,
            "finalized": validated.finalized,
        })

    if args.confirm_subtitles:
        print(f"\n✋ PAUSE: 'Endkontrolle' für {len(render_queue)} Clips. ENTER zum Rendern...")
        input()

    # Final logic for each item before rendering.
    # Finalized spans keep their authoritative boundaries; edited subtitle
    # files may change the subtitle payload, but not the clip boundaries.
    for item in render_queue:
        current_start = item["start"]
        current_end = item["end"]
        final_srt_arg = str(predigt_srt)
        final_json_arg = str(json_file) if json_file else ""

        if args.confirm_subtitles:
            if json_file and item["temp_json"].exists():
                with open(item["temp_json"], "r", encoding="utf-8") as f:
                    edited_json = json.load(f)
                if edited_json and edited_json.get("segments"):
                    if not item["finalized"]:
                        current_start = edited_json["segments"][0]["start"]
                        current_end = edited_json["segments"][-1]["end"]
                    new_subs = json_to_srt(edited_json)
                    item["temp_srt"].write_text(srt.compose(new_subs), encoding="utf-8")
                    final_srt_arg = str(item["temp_srt"])
                    final_json_arg = str(item["temp_json"])
            elif item["temp_srt"].exists():
                edited_subs = load_srt_file(item["temp_srt"])
                if edited_subs:
                    if not item["finalized"]:
                        ns, ne = get_times_from_srt(edited_subs)
                        if ns is not None:
                            current_start, current_end = ns, ne
                    final_srt_arg = str(item["temp_srt"])

        item["start"] = current_start
        item["end"] = current_end
        item["final_srt_arg"] = final_srt_arg
        item["final_json_arg"] = final_json_arg

    logger.info(f"🚀 Starte paralleles Rendering (Max Parallel: {settings.max_parallel_renders})...")
    
    # Da render_clip hier Subprozesse startet, nutzen wir den globalen gpu_semaphore
    def render_with_semaphore(item, vid, reels_out, summ_dir, stem, args):
        with gpu_semaphore:
            render_clip(item, vid, reels_out, summ_dir, stem, args)

    with ThreadPoolExecutor(max_workers=settings.max_parallel_renders) as executor:
        futures = [
            executor.submit(render_with_semaphore, item, vid, reels_out, summ_dir, stem, args)
            for item in render_queue
        ]
        for future in futures:
            future.result() # Warten auf Abschluss und Fehler abfangen


def main():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", default="config.yaml")
    known, _ = conf_parser.parse_known_args()

    defaults = {}
    if Path(known.config).exists():
        with open(known.config, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}

    ap = setup_argparse(defaults)
    args = ap.parse_args()

    # Sync explicit CLI overrides into the global Pydantic settings.
    # Empty string defaults must not overwrite values loaded from config.yaml.
    for key, value in vars(args).items():
        target = SETTINGS_ALIASES.get(key, key)
        if hasattr(settings, target) and value not in (None, ""):
            setattr(settings, target, value)

    if not args.input_dir:
        print("❌ Fehler: 'input_dir' fehlt!")
        sys.exit(1)

    inp = Path(args.input_dir).resolve()
    raw_inp = Path(args.raw_dir).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else Path(".").resolve()
    out_whisper = out_root / "out_srt"
    out_summ = out_root / "out_summ"
    out_reels = out_root / "out_reels"
    
    for d in [out_whisper, out_summ, out_reels]:
        d.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        process_pre_cuts(raw_inp, inp, Path(args.manifest), out_whisper, args.force)

    files = sorted([f for f in inp.iterdir() if f.suffix.lower() in VIDEO_EXTS])
    logger.info(f"Verarbeite {len(files)} Videos...")

    for vid in files:
        stem = sanitize_stem(vid.stem)
        logger.info(f"=== {stem} ===")

        json_file = find_whisper_json(out_whisper / stem, stem)
        srt_file = find_whisper_srt(out_whisper / stem, stem)

        if args.whisper:
            if not (json_file and srt_file) or args.force:
                execute_transcription(vid, stem, out_whisper / stem, args)
                json_file = find_whisper_json(out_whisper / stem, stem)
                srt_file = find_whisper_srt(out_whisper / stem, stem)
            else:
                logger.info("✅ Whisper Daten vorhanden.")

        if not srt_file:
            continue

        summ_dir = out_summ / stem
        summ_dir.mkdir(parents=True, exist_ok=True)
        hl_json = summ_dir / "highlights.json"
        predigt_srt = summ_dir / f"{stem}_predigt.srt"

        if not (hl_json.exists() and predigt_srt.exists()) or args.force:
            execute_summarizer(srt_file, summ_dir, predigt_srt, defaults, args)

        if not predigt_srt.exists() and srt_file.exists():
            logger.info(f"⚠️ ACHTUNG: '{predigt_srt.name}' wurde nicht erstellt. Nutze Fallback.")
            shutil.copy(srt_file, predigt_srt)

        # Diese Dateien sind die Ausgaben von build_highlight_spans.py (Title/Captions)
        spans_md = summ_dir / "highlight_spans.md"
        spans_json = summ_dir / "reel_spans.json"

        if args.build_spans:
            if not spans_json.exists() or args.force:
                execute_span_builder(
                    hl_json,
                    predigt_srt,
                    spans_md,
                    spans_json,
                    args,
                    transcription_json=json_file,
                )
            else:
                logger.info("ℹ️  Spans vorhanden.")

            if args.confirm_spans:
                input(f"\n✋ PAUSE: Prüfe '{spans_json}'. ENTER weiter...")

        if args.render and spans_json.exists():
            execute_render_orchestration(vid, stem, summ_dir, json_file, predigt_srt, spans_json, out_reels, args)

    logger.info("Fertig.")


if __name__ == "__main__":
    main()

