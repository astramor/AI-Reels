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
from typing import List, Optional
from loguru import logger

# Import der Kern-Pipeline
from video_highlight_pipeline_smart import run_smart_pipeline
from core.config import settings
from domains.highlights.parser import HighlightParser
from domains.video.hardware import detect_available_nvenc_codecs


def setup_logging(verbose: bool):
    """Konfiguriert das Logging-Level."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """
    Hauptfunktion des Programms.
    Gibt 0 bei Erfolg, 1 bei Laufzeitfehlern und 2 bei Benutzerfehlern zurück.
    """
    if argv is None:
        argv = sys.argv[1:]

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
        "--spans_json",
        help="Pfad zur JSON-Datei mit den Highlights. Falls nicht vorhanden, wird sie generiert."
    )
    
    # Optionale Erweiterungen
    parser.add_argument(
        "--srt_input",
        help="Optional: SRT-Datei für die automatische Highlight-Generierung (falls spans_json fehlt)."
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
    
    # Qualitätsanforderungen / Flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Gibt nur die Konfiguration und die geplanten Schritte aus, ohne die Verarbeitung zu starten."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Aktiviert detailliertes Logging (DEBUG level)."
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # Bei --help oder falschen Argumenten gibt argparse SystemExit aus.
        # Wenn es ein Fehler war (code != 0), geben wir 2 zurück.
        if e.code == 0:
            return 0
        return 2

    # Logging Setup
    setup_logging(args.verbose)

    # Validierung (Benutzerfehler -> Code 2)
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video-Datei nicht gefunden: {args.video}")
        return 2

    # Settings Overrides
    if args.blur_bg:
        settings.blur_bg = True
    
    if args.nvenc:
        available_codecs = detect_available_nvenc_codecs()
        if "h264_nvenc" in available_codecs:
            settings.nvenc = True
            logger.info("NVIDIA NVENC Hardware-Beschleunigung aktiviert.")
        else:
            settings.nvenc = False
            logger.warning("NVENC 'h264_nvenc' nicht verfügbar. Automatischer Fallback auf CPU (libx265).")

    out_dir = Path(args.out_dir)
    spans_json = Path(args.spans_json) if args.spans_json else out_dir / "highlights.json"
    
    # Dry Run Logik
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        logger.info("Zentrale Konfiguration:")
        config_dict = settings.model_dump()
        for key, value in config_dict.items():
            # Sensitive Daten maskieren
            if "key" in key.lower() or "secret" in key.lower():
                logger.info(f"  {key}: [VERSTECKT]")
            else:
                logger.info(f"  {key}: {value}")
            
        logger.info("Geplante Pipeline-Schritte:")
        logger.info(f"  - Quellvideo: {video_path}")
        logger.info(f"  - Ausgabeverzeichnis: {out_dir}")
        
        if spans_json.exists():
            logger.info(f"  - Highlights-Datei gefunden: {spans_json}")
            parser_hl = HighlightParser()
            try:
                marks = parser_hl.load_highlights_from_json(spans_json)
                logger.info(f"  - Geplante Clips ({len(marks)}):")
                for idx, m in enumerate(marks, start=1):
                    end_str = f" bis {m.end}s" if m.end else " (Endzeit wird berechnet)"
                    logger.info(f"    {idx:02d}. {m.label} [Start: {m.start}s{end_str}]")
            except Exception as e:
                logger.warning(f"  - Konnte Highlights nicht parsen: {e}")
        else:
            logger.info(f"  - Highlights-Datei fehlt. Wird generiert aus: {args.srt_input or 'WhisperX Auto-Transkription'}")
            
        logger.info("Dry-run erfolgreich beendet. Keine Videoverarbeitung wurde gestartet.")
        return 0

    logger.info("Starte PythonReels Pipeline...")
    
    try:
        run_smart_pipeline(
            video_path=video_path,
            spans_json=spans_json,
            srt_input=Path(args.srt_input) if args.srt_input else None,
            out_dir=out_dir,
            transcription_json=Path(args.transcription_json) if args.transcription_json else None,
            music_dir=Path(args.music_dir) if args.music_dir else None,
            subtitles=Path(args.subtitles) if args.subtitles else None
        )
        logger.success("Pipeline erfolgreich abgeschlossen!")
        return 0
    except Exception as e:
        logger.exception(f"Kritischer Fehler während der Pipeline-Ausführung: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
