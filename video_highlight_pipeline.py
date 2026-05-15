#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, json, tempfile, random
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

# Import der Domänen
from domains.video.renderer import FFmpegRenderer
from domains.vision.tracker import FaceTracker
from domains.transcription.processor import SubtitleProcessor
from domains.highlights.parser import HighlightParser, Highlight
from domains.video.hardware import detect_available_nvenc_codecs
from core.config import settings

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

def ffprobe_json(path: Path) -> dict:
    import subprocess
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams", "-show_format",
        "-print_format", "json",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)

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
# 2. MAIN
# ==============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--spans_json", help="Pfad zur JSON-Datei mit den Highlights")
    p.add_argument("--spans_md", help="Abwärtskompatibilität: Pfad zur Markdown-Datei")
    p.add_argument("--out_dir", default=str(settings.out_dir))
    p.add_argument("--subtitles")
    p.add_argument("--transcription_json", help="Pfad zur WhisperX JSON für Karaoke")
    p.add_argument("--music_dir", help="Ordner mit Hintergrundmusik (.mp3/.wav)")
    p.add_argument("--nvenc", action="store_true", help="NVENC nutzen (falls verfügbar)")

    args = p.parse_args()

    # Hardware Fallback
    if args.nvenc or settings.nvenc:
        available = detect_available_nvenc_codecs()
        if "h264_nvenc" in available:
            settings.nvenc = True
        else:
            settings.nvenc = False
            logger.warning("NVENC 'h264_nvenc' nicht verfügbar. Nutze CPU (libx265).")

    video = Path(args.video).resolve()
    ensure_outdir(Path(args.out_dir))
    
    # Parser instanziieren
    hl_parser = HighlightParser()
    if args.spans_json:
        marks = hl_parser.load_highlights_from_json(Path(args.spans_json))
    elif args.spans_md:
        marks = hl_parser.load_highlights_from_json(Path(args.spans_md)) # JSON parser is now standard
    else:
        logger.error("Keine Highlights-Datei angegeben.")
        return

    src_w, src_h = probe_wh(video)
    target_w, target_h = 1080, 1920

    json_path = (
        Path(args.transcription_json).resolve() if args.transcription_json else None
    )

    music_file = None
    if args.music_dir:
        music_file = get_random_music(Path(args.music_dir))
        if music_file:
            logger.info(f"🎵 Nutze Hintergrundmusik: {music_file.name}")

    renderer = FFmpegRenderer()
    tracker = FaceTracker()
    sub_processor = SubtitleProcessor()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        for idx, h in enumerate(marks, start=1):
            approx_start = max(0, h.start - settings.preroll)
            start_abs = sub_processor.snap_start_to_json_word(json_path, approx_start, window=0.3)

            if h.end is not None:
                calculated_dur = h.end - start_abs + settings.postroll
                dur = min(calculated_dur, settings.max_window)
            else:
                dur = settings.max_window

            if dur < 1.0:
                dur = 5.0

            title_text = titleize_label(h.label, 50, 3) if settings.overlay_title else None
            safe_title = re.sub(r"[^a-zA-Z0-9äöüÄÖÜß]+", "_", h.label).strip("_")[:60]
            out_file = Path(args.out_dir) / f"{settings.prefix}{idx:02d}_{safe_title}.mp4"

            ass_file = None
            if json_path and json_path.exists():
                ass_file = temp_dir_path / f"temp_subs_{idx}.ass"
                logger.info(f"Generiere Karaoke für Clip {idx}...")
                sub_processor.generate_karaoke_ass(json_path, start_abs, dur, ass_file)

            cover_override = None
            if not settings.blur_bg:
                start_center = None
                if settings.face_detect_on_start:
                    start_center = tracker.robust_start_center(video, start_abs, settings.face_first_range, settings.face_first_steps, settings.face_first_grid, settings.face_first_sample_rate, settings.face_min_conf, settings.face_use_pose_fallback, None)
                    if (start_center is None) and (settings.fallback_third != "none"):
                        thirds = {"left": (0.33 * src_w, 0.45 * src_h), "center": (0.50 * src_w, 0.45 * src_h), "right": (0.67 * src_w, 0.45 * src_h)}
                        start_center = thirds.get(settings.fallback_third)

                track = []
                if (settings.face_track or settings.face_center) and src_w:
                    track = tracker.compute_face_centers(video, start_abs, start_abs + dur, settings.face_sample_rate, settings.face_min_conf)
                    if track and settings.face_track:
                        track = tracker.smooth_track(track, window_length=51, polyorder=2, deadzone_px=80.0)
                        track = tracker.interpolate_track_to_fps(track, target_fps=30.0)

                if start_center:
                    if settings.face_track and track:
                        keys = [(0.0, start_center[0], start_center[1])] + [(t, x, y) for (t, x, y) in track if t > 0.0]
                        cover_override = renderer.cover_scale_crop_filter_with_track(src_w, src_h, target_w, target_h, keys)
                    else:
                        cover_override = renderer.cover_scale_crop_filter_with_center(src_w, src_h, target_w, target_h, start_center[0], start_center[1])
                elif track:
                    cover_override = renderer.cover_scale_crop_filter_with_track(src_w, src_h, target_w, target_h, track)
                else:
                    cover_override = renderer.cover_scale_crop_filter_with_center(src_w, src_h, target_w, target_h, src_w/2, src_h/2)

            cmd = renderer.build_ffmpeg_cmd(
                input_video=video,
                start_ts=start_abs,
                duration=dur,
                out_file=out_file,
                ass_file=ass_file,
                subtitles=Path(args.subtitles) if args.subtitles else None,
                overlay_title=title_text,
                reencode=settings.reencode,
                video_codec="h264_nvenc" if settings.nvenc else "libx265",
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

    logger.info(f"✅ Fertig. Clips liegen in: {args.out_dir}")

if __name__ == "__main__":
    main()
