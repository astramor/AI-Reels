#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
-------
Zentraler Einstiegspunkt für den Video-Highlight-Generator (PythonReels).
Orchestriert die Argument-Verarbeitung und ruft die Pipeline auf.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Import der Kern-Pipeline
from video_highlight_pipeline_smart import run_smart_pipeline
from core.config import settings


def main():
    parser = argparse.ArgumentParser(
        description="PythonReels: Automatisierter Video-Highlight-Generator mit KI-Transkription und Smart-Tracking."
    )

    # Basis-Argumente
    parser.add_argument(
        "--video",
        required=True,
        help="Pfad zum Quellvideo (16:9)."
    )
    parser.add_argument(
        "--spans_md",
        required=True,
        help="Pfad zur Markdown-Datei mit den Highlights (Zeitstempel + Text)."
    )
    
    # Optionale Erweiterungen
    parser.add_argument(
        "--srt_input",
        help="Optional: SRT-Datei für die automatische Highlight-Generierung (falls spans_md fehlt)."
    )
    parser.add_argument(
        "--out_dir",
        default=str(settings.out_dir),
        help=f"Ausgabeverzeichnis für die Clips (Standard: {settings.out_dir})."
    )
    parser.add_argument(
        "--transcription_json",
        help="Pfad zur WhisperX JSON-Datei für Karaoke-Untertitel."
    )
    parser.add_argument(
        "--music_dir",
        help="Verzeichnis mit Hintergrundmusik-Dateien (.mp3, .wav)."
    )
    parser.add_argument(
        "--subtitles",
        help="Pfad zu einer SRT-Datei für klassische Untertitel (Burn-in)."
    )

    # Flag-Overrides
    parser.add_argument(
        "--blur_bg",
        action="store_true",
        help="Hintergrund weichzeichnen (statt Smart-Tracking Crop)."
    )
    parser.add_argument(
        "--nvenc",
        action="store_true",
        help="NVIDIA Hardware-Encoding nutzen."
    )

    args = parser.parse_args()

    # Validierung
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video-Datei nicht gefunden: {args.video}")
        sys.exit(1)

    # Settings Overrides (optional, falls via CLI übergeben)
    if args.blur_bg:
        settings.blur_bg = True
    if args.nvenc:
        settings.nvenc = True

    logger.info("Starte PythonReels Pipeline...")
    
    try:
        run_smart_pipeline(
            video_path=video_path,
            spans_md=Path(args.spans_md),
            srt_input=Path(args.srt_input) if args.srt_input else None,
            out_dir=Path(args.out_dir),
            transcription_json=Path(args.transcription_json) if args.transcription_json else None,
            music_dir=Path(args.music_dir) if args.music_dir else None,
            subtitles=Path(args.subtitles) if args.subtitles else None
        )
        logger.success("Pipeline erfolgreich abgeschlossen!")
    except Exception as e:
        logger.exception(f"Fehler während der Pipeline-Ausführung: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
