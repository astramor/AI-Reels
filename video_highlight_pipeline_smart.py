#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import shutil
import tempfile
import random
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

from domains.video.renderer import FFmpegRenderer
from domains.vision.tracker import FaceTracker
from domains.transcription.processor import SubtitleProcessor
from domains.highlights.parser import HighlightParser, Highlight
from domains.llm.summarizer import SermonSummarizer
from core.config import settings
from core.commands import run_command

# ==============================================================================
# 1. HELPERS
# ==============================================================================

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def titleize_label(raw: str, max_chars_line: int = 30, max_lines: int = 3) -> str:
    txt = re.sub(r"\s+", " ", raw.strip())
    txt = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", txt)
    words = txt.split(" ")
    lines, line = [], ""
    for w in words:
        add = w if not line else " " + w
        if len(line) + len(add) <= max_chars_line:
            line += add
        else:
            lines.append(line)
            line = w
            if len(lines) >= max_lines:
                break
    if line and len(lines) < max_lines:
        lines.append(line)
    return "\n".join([l.strip() for l in lines])

def ffprobe_json(path: Path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-print_format", "json",
        str(path),
    ]
    result = run_command(cmd)
    return json.loads(result.stdout)

def probe_wh(path: Path) -> Tuple[int, int]:
    info = ffprobe_json(path)
    for st in info.get("streams", []):
        if st.get("codec_type") == "video":
            return int(st["width"]), int(st["height"])
    raise RuntimeError("Konnte Video-Breite/Höhe nicht ermitteln")

def get_random_music(music_dir: Path) -> Optional[Path]:
    if not music_dir or not music_dir.exists():
        return None
    songs = (
        list(music_dir.glob("*.mp3"))
        + list(music_dir.glob("*.wav"))
        + list(music_dir.glob("*.m4a"))
    )
    if not songs:
        return None
    return random.choice(songs)

# ==============================================================================
# 2. CORE CLIP PROCESSING
# ==============================================================================

def process_clip(
    idx: int,
    h: Highlight,
    video_path: Path,
    out_dir: Path,
    json_path: Optional[Path],
    temp_dir_path: Path,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
    renderer: FFmpegRenderer,
    tracker: FaceTracker,
    sub_processor: SubtitleProcessor,
    music_file: Optional[Path],
    subtitles_path: Optional[Path]
):
    """
    Verarbeitet einen einzelnen Highlight-Clip: Timing, Tracking, Karaoke, Rendering.
    """
    # 1. Timing-Berechnung
    approx_start = max(0, h.start - settings.preroll)
    start_abs = sub_processor.snap_start_to_json_word(json_path, approx_start, window=0.3)

    if h.end is not None:
        calculated_dur = h.end - start_abs + settings.postroll
        dur = min(calculated_dur, settings.max_window)
    else:
        dur = settings.max_window

    if dur < 1.0:
        dur = 5.0

    # 2. Dateinamen & Titel
    title_text = titleize_label(h.label, 50, 3) if settings.overlay_title else None
    safe_title = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", "_", h.label).strip("_")[:60]
    out_file = out_dir / f"{settings.prefix}{idx:02d}_{safe_title}.mp4"

    # 3. Karaoke-Generierung
    ass_file = None
    if json_path and json_path.exists():
        ass_file = temp_dir_path / f"temp_subs_{idx}.ass"
        logger.info(f"Generiere Karaoke für Clip {idx} ({start_abs:.2f}s - {start_abs + dur:.2f}s)...")
        m_v = 280 if settings.blur_bg else 250
        sub_processor.generate_karaoke_ass(json_path, start_abs, dur, ass_file, margin_v=m_v)

    # 4. Bildausschnitt & Tracking (nur wenn kein Blur-BG)
    cover_override = None
    if not settings.blur_bg:
        start_center = None
        if settings.face_detect_on_start:
            start_center = tracker.robust_start_center(
                video_path,
                start_abs,
                settings.face_first_range,
                settings.face_first_steps,
                settings.face_first_grid,
                settings.face_first_sample_rate,
                settings.face_min_conf,
                settings.face_use_pose_fallback,
                None,
            )
            if (start_center is None) and (settings.fallback_third != "none"):
                thirds = {
                    "left": (0.33 * src_w, 0.45 * src_h),
                    "center": (0.50 * src_w, 0.45 * src_h),
                    "right": (0.67 * src_w, 0.45 * src_h),
                }
                start_center = thirds.get(settings.fallback_third, (src_w / 2, src_h / 2))

        track = []
        if (settings.face_track or settings.face_center) and src_w:
            track = tracker.compute_face_centers(
                video_path,
                start_abs,
                start_abs + dur,
                sample_fps=20.0,
                min_conf=settings.face_min_conf,
            )
            if (not track) and settings.face_use_pose_fallback:
                track = tracker.compute_pose_head_centers(
                    video_path,
                    start_abs,
                    start_abs + dur,
                    sample_fps=12.0,
                )

            if settings.face_track and track and len([p for p in track if p[0] > 0]) < 4:
                track = []

            if settings.face_track and start_center and track:
                cx0, cy0 = start_center
                snap_r = 220
                track = [(t, x, y) for (t, x, y) in track if ((x - cx0) ** 2 + (y - cy0) ** 2) ** 0.5 <= snap_r]

        if track and settings.face_track:
            # Dead-Zone wiederhergestellt: Hält das Bild bei kleinen Bewegungen still
            dead_zone_amount = src_w * 0.04 # 4% des Bildes als Puffer
            track = tracker.apply_dead_zone(track, dead_zone_px=dead_zone_amount)
            
            if settings.face_smooth_sec > 0:
                track = tracker.smooth_track(track, settings.face_smooth_sec)
            
            # Massive Erhöhung der Keyframes (60 statt 40) für ultra-glatte Kurven in FFmpeg
            track = tracker.reduce_keyframes(track, 60)

        # Filter-Generierung
        if start_center:
            if settings.face_track and track:
                keys = [(0.0, start_center[0], start_center[1])] + [(t, x, y) for (t, x, y) in track if t > 0.0]
                cover_override = renderer.cover_scale_crop_filter_with_track(src_w, src_h, target_w, target_h, keys)
            else:
                cover_override = renderer.cover_scale_crop_filter_with_center(src_w, src_h, target_w, target_h, start_center[0], start_center[1])
        elif track and settings.face_track:
            cover_override = renderer.cover_scale_crop_filter_with_track(src_w, src_h, target_w, target_h, track)
        elif settings.face_center and track:
            cx = tracker._median([p[1] for p in track])
            cy = tracker._median([p[2] for p in track])
            cover_override = renderer.cover_scale_crop_filter_with_center(src_w, src_h, target_w, target_h, cx, cy)
        else:
            cx, cy = src_w / 2, src_h / 2
            cover_override = renderer.cover_scale_crop_filter_with_center(src_w, src_h, target_w, target_h, cx, cy)

    # 5. FFmpeg Command & Rendering
    cmd = renderer.build_ffmpeg_cmd(
        input_video=video_path,
        start_ts=start_abs,
        duration=dur,
        out_file=out_file,
        ass_file=ass_file,
        subtitles=subtitles_path,
        overlay_title=title_text,
        reencode=settings.reencode,
        video_codec="h264_nvenc" if settings.nvenc else "libx264",
        target_w=target_w,
        target_h=target_h,
        cover_filter_override=cover_override,
        loudnorm=settings.loudnorm,
        fade_in=settings.fade_in,
        fade_in_sec=settings.fade_in_sec,
        fade_out=settings.fade_out,
        fade_sec=settings.fade_sec,
        overlay_margin_top=settings.overlay_margin_top,
        overlay_fontsize=settings.overlay_fontsize,
        color_grading=settings.color_grade,
        music_file=music_file,
        music_volume=settings.music_volume,
        blur_bg=settings.blur_bg,
    )

    renderer.run_ffmpeg(cmd)
    logger.success(f"Clip {idx} erfolgreich gerendert: {out_file.name}")

# ==============================================================================
# 3. PIPELINE ORCHESTRATION
# ==============================================================================

def run_smart_pipeline(
    video_path: Path,
    spans_md: Path,
    srt_input: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    transcription_json: Optional[Path] = None,
    music_dir: Optional[Path] = None,
    subtitles: Optional[Path] = None
):
    """
    Setup der Pipeline: Validierung, Highlight-Generierung, Instanziierung der Domänen.
    """
    video = video_path.resolve()
    out_dir = (out_dir or settings.out_dir).resolve()
    ensure_outdir(out_dir)

    # Schritt 0: Automatische Transkription
    if not srt_input:
        srt_input = out_dir / f"{video.stem}.srt"
    if not transcription_json:
        transcription_json = out_dir / f"{video.stem}.json"

    if not srt_input.exists():
        logger.info(f"🎙️ Starte WhisperX für {video.name}...")
        whisper_cmd = [
            "whisperx", str(video),
            "--model", "large-v3",
            "--language", "de",
            "--output_dir", str(out_dir),
            "--device", "cuda",
            "--compute_type", "float16"
        ]
        run_command(whisper_cmd)

    # Schritt 1: Highlights generieren (LLM)
    if not spans_md.exists():
        logger.info(f"✨ Generiere Highlights aus {srt_input.name}...")
        summarizer = SermonSummarizer(settings)
        res = summarizer.process(str(srt_input), str(out_dir))
        
        # --- NEU: VRAM freigeben ---
        summarizer.unload()
        del summarizer
        # ---------------------------

        # Pfad-Abgleich
        gen_hl = Path(res["highlights_path"])
        if gen_hl.resolve() != spans_md.resolve():
            shutil.copy(gen_hl, spans_md)
    else:
        # Falls summarizer nicht benötigt wird, trotzdem versuchen zu entladen (Sicherheit)
        try:
            temp_sum = SermonSummarizer(settings)
            temp_sum.unload()
            del temp_sum
        except:
            pass

    # 2. Parser & Daten laden
    hl_parser = HighlightParser()
    marks = hl_parser.load_highlights_from_md(spans_md)
    src_w, src_h = probe_wh(video)
    target_w, target_h = 1080, 1920

    # 3. Musik-Setup
    music_file = get_random_music(music_dir) if music_dir else None
    if music_file:
        logger.info(f"🎵 Nutze Hintergrundmusik: {music_file.name}")

    # 4. Domänen-Instanzen
    renderer = FFmpegRenderer()
    tracker = FaceTracker()
    sub_processor = SubtitleProcessor()

    # --- NEU: Smart Cut Fallback ---
    srt_items = sub_processor.load_srt_file(srt_input) if srt_input.exists() else []
    for h in marks:
        if h.end is None and srt_items:
            logger.info(f"📍 Clip '{h.label}' hat keine Endzeit. Suche Smart Cut...")
            h.end = sub_processor.find_smart_end(srt_items, h.start, min_dur=18.0, max_dur=settings.max_window)
    # -------------------------------

    # 5. Iterative Verarbeitung
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for idx, h in enumerate(marks, start=1):
            process_clip(
                idx=idx,
                h=h,
                video_path=video,
                out_dir=out_dir,
                json_path=transcription_json,
                temp_dir_path=temp_dir_path,
                src_w=src_w,
                src_h=src_h,
                target_w=target_w,
                target_h=target_h,
                renderer=renderer,
                tracker=tracker,
                sub_processor=sub_processor,
                music_file=music_file,
                subtitles_path=subtitles
            )

if __name__ == "__main__":
    # Fallback-Aufruf für Kompatibilität
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--spans_md")
    p.add_argument("--out_dir")
    args = p.parse_args()
    
    v_path = Path(args.video)
    o_dir = Path(args.out_dir) if args.out_dir else settings.out_dir
    s_md = Path(args.spans_md) if args.spans_md else o_dir / "highlights.md"
    
    run_smart_pipeline(v_path, s_md, out_dir=o_dir)
