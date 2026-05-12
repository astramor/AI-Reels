#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, subprocess, shlex, json, yaml, shutil, re
from pathlib import Path
from datetime import datetime, timedelta
import srt
from core.time_utils import hms_to_seconds, parse_time_to_seconds, seconds_to_hms
from domains.highlights.parser import HighlightParser
from core.commands import run_command

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".m4v", ".avi", ".mp3", ".wav", ".webm"}


def log(msg: str):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}")


def run(cmd, *, env=None, dry_run=False, cwd=None):
    if dry_run:
        print("▶ (DRY-RUN)", " ".join(shlex.quote(str(c)) for c in cmd))
        return 0
    
    try:
        # run_command loggt intern via logger.debug, batch_reels möchte aber explizit print
        print("▶", " ".join(shlex.quote(str(c)) for c in cmd))
        if env and env.get("CT2_USE_CUDNN") == "0":
            print("  (ENV) CT2_USE_CUDNN=0")
            
        run_command(cmd, cwd=cwd, env=env, capture_output=False)
        return 0
    except Exception as e:
        print(f"❌ Command failed: {e}")
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
            "title": h.label
        })
    return spans


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

    # NEU: Der Puffer am Anfang ist jetzt extrem klein (-0.05 statt -0.2),
    # damit wir keine vorherigen Wörter mehr versehentlich mit in den Clip ziehen.
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
            # NEU: Der Fix für die fehlenden Leerzeichen zwischen den Wörtern
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
    """
    Entfernt alle nicht-alphanumerischen Zeichen außer Unter- und Bindestrichen,
    um CWE-22 (Path Traversal) Schwachstellen zu verhindern.
    """
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
            target_file = input_dir / vid.name
            if target_file.exists() and not force:
                continue
            s_sec = parse_time_to_seconds(start_str)
            e_sec = parse_time_to_seconds(end_str)
            duration = e_sec - s_sec
            if duration <= 0:
                continue
            log(f"✂️  Schneide Predigt: {stem} (Start: {start_str}, Dauer: {duration}s)")
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(start_str),
                "-i",
                str(vid),
                "-t",
                str(duration),
                "-c",
                "copy",
                "-map",
                "0",
                str(target_file),
            ]
            run(cmd)
            whisper_dir = out_whisper_root / stem
            if whisper_dir.exists():
                shutil.rmtree(whisper_dir)


def main():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", default="config.yaml")
    known, _ = conf_parser.parse_known_args()

    defaults = {}
    if Path(known.config).exists():
        with open(known.config, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}

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

    ap.add_argument("--llm-model", default="")
    ap.add_argument("--llm-provider", default="")
    ap.add_argument("--gemini-api-key", default="")
    ap.add_argument("--gemini-model", default="")

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

    args = ap.parse_args()

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
    log(f"Verarbeite {len(files)} Videos...")

    for vid in files:
        stem = sanitize_stem(vid.stem)
        log(f"=== {stem} ===")

        json_file = find_whisper_json(out_whisper / stem, stem)
        srt_file = find_whisper_srt(out_whisper / stem, stem)

        if args.whisper:
            if (json_file and srt_file) and not args.force:
                log("✅ Whisper Daten vorhanden.")
            else:
                log("Starte WhisperX...")
                w_cmd = [
                    "whisperx",
                    str(vid),
                    "--model",
                    args.whisper_model,
                    "--output_dir",
                    str(out_whisper / stem),
                    "--output_format",
                    "all",
                    "--language",
                    args.whisper_language,
                ]
                if args.whisper_device:
                    w_cmd += ["--device", args.whisper_device]
                if args.whisper_compute_type:
                    w_cmd += ["--compute_type", args.whisper_compute_type]
                if args.whisper_batch_size:
                    w_cmd += ["--batch_size", str(args.whisper_batch_size)]
                if args.whisper_vad:
                    w_cmd.append("--vad_filter")
                env = os.environ.copy()
                if args.no_cudnn:
                    env["CT2_USE_CUDNN"] = "0"
                run(w_cmd, env=env, dry_run=args.dry_run)
                json_file = find_whisper_json(out_whisper / stem, stem)
                srt_file = find_whisper_srt(out_whisper / stem, stem)

        if not srt_file:
            continue

        summ_dir = out_summ / stem
        summ_dir.mkdir(parents=True, exist_ok=True)
        hl_md = summ_dir / "highlights.md"
        predigt_srt = summ_dir / f"{stem}_predigt.srt"

        files_missing = not (hl_md.exists() and predigt_srt.exists())

        if files_missing or args.force:
            log("Starte Summarizer...")
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
            if "summarizer_preset" in defaults:
                s_cmd += ["--preset", defaults["summarizer_preset"]]
            run(s_cmd, dry_run=args.dry_run)

        if not predigt_srt.exists() and srt_file.exists():
            log(f"⚠️ ACHTUNG: '{predigt_srt.name}' wurde nicht erstellt.")
            log(f"   -> Kopiere Original-SRT als Fallback, damit es weitergeht.")
            shutil.copy(srt_file, predigt_srt)

        spans_md = summ_dir / "highlight_spans.md"
        spans_json = summ_dir / "reel_spans.json"

        if args.build_spans:
            if spans_md.exists() and not args.force:
                log("ℹ️  Spans vorhanden.")
            else:
                log("Berechne Spans & Titel...")
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
                    "--title-min",
                    str(args.title_min),
                    "--title-max",
                    str(args.title_max),
                    "--gemini-model",
                    args.gemini_model,
                ]
                if args.llm_titles:
                    b_cmd.append("--llm-titles")
                else:
                    b_cmd.append("--no-llm")
                if args.church_yaml:
                    b_cmd += ["--church-yaml", str(args.church_yaml)]
                run(b_cmd, dry_run=args.dry_run)

            if args.confirm_spans:
                print(f"\n✋ PAUSE: Prüfe '{spans_md}'. ENTER weiter...")
                input()

        if args.render and spans_md.exists():
            spans_content = parse_highlights_md(spans_md)
            reels_out = out_reels / stem
            reels_out.mkdir(parents=True, exist_ok=True)

            render_queue = []
            all_subs = load_srt_file(predigt_srt)

            for idx, span in enumerate(spans_content, 1):
                c_start = span["start"]
                c_end = span["end"]
                label = span["title"]
                temp_srt_path = summ_dir / f"temp_edit_reel_{idx}.srt"
                temp_json_path = summ_dir / f"temp_edit_reel_{idx}.json"

                if args.confirm_subtitles:
                    if json_file:
                        mini_json = get_json_slice(json_file, c_start, c_end)
                        if mini_json:
                            # 1. JSON speichern
                            with open(temp_json_path, "w", encoding="utf-8") as f:
                                json.dump(mini_json, f, indent=2, ensure_ascii=False)
                            # 2. SRT DIREKT aus dem mini_json generieren (100% synchron!)
                            slice_subs = json_to_srt(mini_json)
                            temp_srt_path.write_text(
                                srt.compose(slice_subs), encoding="utf-8"
                            )
                    else:
                        # Fallback, falls kein JSON existiert
                        slice_subs = get_srt_slice(all_subs, c_start, c_end)
                        temp_srt_path.write_text(
                            srt.compose(slice_subs), encoding="utf-8"
                        )

                render_queue.append(
                    {
                        "idx": idx,
                        "label": label,
                        "start": c_start,
                        "end": c_end,
                        "temp_srt": temp_srt_path,
                        "temp_json": temp_json_path,
                        "title": label,
                    }
                )

            if args.confirm_subtitles:
                print(
                    f"\n✋ PAUSE: 'Endkontrolle' für {len(render_queue)} Clips. ENTER zum Rendern..."
                )
                input()

            for item in render_queue:
                idx = item["idx"]
                current_start = item["start"]
                current_end = item["end"]
                final_srt_arg = str(predigt_srt)
                final_json_arg = str(json_file) if json_file else ""

                if args.confirm_subtitles:
                    if json_file and item["temp_json"].exists():
                        # Lade das (möglicherweise von dir bearbeitete) JSON
                        with open(item["temp_json"], "r", encoding="utf-8") as f:
                            edited_json = json.load(f)

                        if edited_json and edited_json.get("segments"):
                            # 1. Hole die NEUEN Start- und Endzeiten direkt aus dem JSON
                            current_start = edited_json["segments"][0]["start"]
                            current_end = edited_json["segments"][-1]["end"]

                            # 2. Überschreibe die SRT-Datei basierend auf dem bearbeiteten JSON
                            new_subs = json_to_srt(edited_json)
                            item["temp_srt"].write_text(
                                srt.compose(new_subs), encoding="utf-8"
                            )

                            final_srt_arg = str(item["temp_srt"])
                            final_json_arg = str(item["temp_json"])
                    elif item["temp_srt"].exists():
                        # Fallback: Falls kein JSON existiert, nimm die SRT
                        edited_subs = load_srt_file(item["temp_srt"])
                        if edited_subs:
                            ns, ne = get_times_from_srt(edited_subs)
                            if ns is not None:
                                current_start, current_end = ns, ne
                                final_srt_arg = str(item["temp_srt"])

                tmp_span = summ_dir / f"tmp_span_{idx}.md"
                tmp_span.write_text(
                    f"- [{seconds_to_hms(current_start)} -> {seconds_to_hms(current_end)}] {item['title']}",
                    encoding="utf-8",
                )

                cmd = [
                    sys.executable,
                    "video_highlight_pipeline.py",
                    "--video",
                    str(vid),
                    "--spans_md",
                    str(tmp_span),
                    "--out_dir",
                    str(reels_out),
                    "--prefix",
                    f"{stem}_{idx:02d}_",
                    "--subtitles",
                    final_srt_arg,
                    "--transcription_json",
                    final_json_arg,
                    "--preroll",
                    str(args.preroll),
                    "--postroll",
                    str(args.postroll),
                    "--max-window",
                    str(args.max_window),
                    "--srt-min-first",
                    str(args.srt_min_first),
                    "--crf",
                    str(args.crf),
                    "--preset",
                    args.preset,
                    "--audio-bitrate",
                    args.audio_bitrate,
                ]

                if args.reel:
                    cmd.append("--reel")
                if args.loudnorm:
                    cmd.append("--loudnorm")
                if args.fade_in:
                    cmd += ["--fade-in", "--fade-in-sec", str(args.fade_in_sec)]
                if args.fade_out:
                    cmd += ["--fade-out", "--fade-sec", str(args.fade_sec)]
                if args.overlay_title:
                    cmd += [
                        "--overlay-title",
                        "--overlay-fontsize",
                        str(args.overlay_fontsize),
                        "--overlay-margin-top",
                        str(args.overlay_margin_top),
                    ]
                if args.export_srt:
                    cmd.append("--export-srt")
                if args.reencode:
                    cmd.append("--reencode")
                if args.nvenc:
                    cmd.append("--nvenc")
                if args.blur_bg:
                    cmd.append("--blur-bg")
                if args.color_grade:
                    cmd.append("--color-grade")
                if args.music_dir:
                    cmd += ["--music-dir", str(args.music_dir)]
                if args.music_volume is not None:
                    cmd += ["--music-volume", str(args.music_volume)]

                cmd += [
                    "--ducking-threshold",
                    str(args.ducking_threshold),
                    "--ducking-ratio",
                    str(args.ducking_ratio),
                    "--ducking-attack",
                    str(args.ducking_attack),
                    "--ducking-release",
                    str(args.ducking_release),
                ]

                if args.face_detect_on_start:
                    cmd.append("--face-detect-on-start")
                if args.face_first_robust:
                    cmd.append("--face-first-robust")
                if args.face_track:
                    cmd.append("--face-track")
                if args.face_center:
                    cmd.append("--face-center")
                if args.face_use_pose_fallback:
                    cmd.append("--face-use-pose-fallback")
                if args.fallback_third and args.fallback_third != "none":
                    cmd += ["--fallback-third", args.fallback_third]

                run(cmd, dry_run=args.dry_run)
                tmp_span.unlink()

    log("Fertig.")


if __name__ == "__main__":
    main()
