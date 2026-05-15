#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
srt_summarizer_local.py
----------------------
Zentrale Zusammenfassung und Highlight-Extraktion.
Nutzt nun die SermonSummarizer-Klasse für striktes JSON.
"""

import argparse
import pathlib
import sys
from loguru import logger
from domains.llm.summarizer import SermonSummarizer
from core.config import settings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out_dir", default="out_summ")
    ap.add_argument("--max-highlights", type=int, default=8)
    ap.add_argument("--model")
    ap.add_argument("--export-sermon-srt") # Kompatibilität
    ap.add_argument("--fix-spelling", action="store_true") # Ignoriert
    ap.add_argument("--auto-sermon", action="store_true") # Ignoriert
    ap.add_argument("--min-sermon-minutes", type=int) # Ignoriert
    ap.add_argument("--preset") # Ignoriert

    args = ap.parse_args()

    # Settings Overrides
    if args.model:
        if settings.llm_provider == "gemini":
            settings.gemini_model = args.model
        else:
            settings.llm_model = args.model

    try:
        summarizer = SermonSummarizer(settings)
        result = summarizer.process(
            srt_path_str=args.srt,
            out_dir_str=args.out_dir,
            max_highlights=args.max_highlights
        )
        logger.success(f"Highlights gespeichert in: {result['highlights_path']}")
        
        # Falls ein spezielles Predigt-SRT angefordert wurde, kopieren wir es einfach
        if args.export_sermon_srt:
            import shutil
            shutil.copy(args.srt, args.export_sermon_srt)
            
    except Exception as e:
        logger.exception(f"Fehler im Summarizer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
