# AGENTS.md - Reel Pipeline Architektur & Regeln

## Aktive Projektstruktur (WICHTIG)
- Ignoriere das Verzeichnis `old_files/` bei allen Refactorings.
- `batch_reels.py`: Orchestrator (nutzt `execute_span_builder()`, `render_clip()` etc.).
- `srt_summarizer_local.py`: Erstellt `summary.md`, `highlights_longlist.md`, `highlights.md` und `highlights.json` (für `batch_reels.py` / `build_highlight_spans.py`).
- `build_highlight_spans.py`: Berechnet Schnittfenster (`highlight_spans.md`, `reel_spans.json`), virale Titel und Social-Media-Captions (`reels_content.md`).
- `video_highlight_pipeline_smart.py` (plus `core/` und `domains/`): Aktive Rendering-Pipeline für 9:16-Reels, Karaoke-ASS-Untertitel, Face-Tracking und FFmpeg-Audio-Mixing (ersetzt das veraltete `video_highlight_pipeline.py`).
- `core/config.py`: Zentrale Settings- und YAML-Ladelogik (`load_yaml_defaults()`).
- `core/llm_client.py`: Zentrale LLM-Anbindung (`UnifiedLLM`, `clean_thinking`, Provider- und Fallback-Logik).
- `config.yaml`: Zentrale Konfiguration (nutzt `openai_api_key` bzw. Umgebungsvariablen).

## Strikte Vorgaben für Code-Änderungen
- Inspiziere vor jeder Änderung die tatsächlich aktiven Dateien im Root sowie in `core/` und `domains/`.
- Gib niemals gekürzten Code mit Platzhaltern wie `# ... restlicher Code ...` aus.
- Erhalte alle CLI-Argumente kompatibel zwischen `batch_reels.py`, den Subprozessen und `core/config.py`.
- Speichere keine API-Keys im Klartext; leere YAML-Keys dürfen gesetzte Umgebungsvariablen (`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`) nicht überschreiben.
- Verifiziere jede Änderung zwingend mit `py_compile` UND der Testsuite:
  `OPENAI_API_KEY=testenv .venv/bin/python -m pytest -q`
