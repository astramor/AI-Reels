# Architektur-Dokumentation: WhisperX Reel Pipeline

Diese Dokumentation beschreibt die Architektur, die Modulverantwortlichkeiten und den Workflow der WhisperX Reel Pipeline, inklusive der neuesten Refactoring-Maßnahmen.

## 1. Ordnerstruktur

Die Codebasis ist modular aufgebaut und trennt Konfiguration, Domänenlogik und Orchestrierungs-Skripte.

```text
/whisperXBuild
├── core/
│   ├── config.py           # Zentrale Konfiguration (Pydantic-Settings)
│   ├── prompts.py          # Zentrale LLM-Prompts (Summarizer, Titel, Social Media)
│   └── time_utils.py       # Einheitliche Zeit-Helfer (Parsing & Formatierung)
├── domains/                # Domänenspezifische Logik (Wiederverwendbar)
│   ├── highlights/
│   │   └── parser.py       # Parsen von Zeitstempeln aus Markdown
│   ├── llm/
│   │   └── summarizer.py   # LLM-Orchestrierung (Ollama/Gemini)
│   ├── transcription/
│   │   └── processor.py    # WhisperX-Datenverarbeitung & Karaoke-Generierung
│   ├── video/
│   │   └── renderer.py     # FFmpeg-Wrapper für komplexes Video-Editing
│   └── vision/
│       └── tracker.py      # Face- & Pose-Tracking (Mediapipe Tasks API)
├── models/                 # KI-Modelle (.task Dateien für Mediapipe)
├── input/                  # Eingabevideos
├── out_clips/              # Generierte Video-Highlights
├── out_srt/                # Transkriptions-Ergebnisse (SRT, JSON)
├── out_summ/               # LLM-Ergebnisse (Summaries, Highlight-Listen)
├── batch_reels.py          # Orchestrierung des kompletten Batch-Workflows
├── video_highlight_pipeline_smart.py  # Smarte Pipeline mit autonomem Workflow
├── build_highlight_spans.py # Veredelung von Highlights mit LLM-Titeln & Posts
└── run_reel.sh             # Komfort-Einstiegspunkt für Einzelvideos
```

---

## 2. Modul-Logik & Zuständigkeiten

### Core-Modul (Infrastruktur)
*   **core/config.py**: Zentrale Verwaltung aller Settings via Pydantic mit Feld-Validierung (CRF, Volume, Timing-Windows).
*   **core/prompts.py**: Beinhaltet alle LLM-Vorgaben für eine konsistente KI-Persönlichkeit.
*   **core/time_utils.py**: Robuste Funktionen zur Umrechnung zwischen Sekunden, HH:MM:SS und `timedelta`.
*   **core/commands.py**: Sicherer Subprocess-Wrapper (`run_command`) mit Stderr-Erfassung, Logging und Exception-Handling (kein `shell=True`).

### Testing & Qualitätssicherung
*   **pytest-Suite**: Umfassende Tests in `tests/` für alle Core-Komponenten.
*   **Smoke-Tests**: Integrationstests (`test_smoke_pipeline.py`) prüfen Parser, Zeit-Logik und SRT-Slicing ohne Rendering.
*   **Validierung**: Automatisierte Prüfung via `compileall` und `pytest` im CI-Flow vorgesehen.

### Domains-Ordner (Wiederverwendbare Fachlogik)
*   **domains/highlights/parser.py**: Extrahiert Highlights aus Markdown.
*   **domains/llm/summarizer.py**: Schnittstelle zu LLMs für Inhaltsanalyse.
*   **domains/transcription/processor.py**: Word-Snapping und Karaoke-ASS-Generierung.
*   **domains/video/renderer.py**: FFmpeg-Orchestrierung mit Fokus auf Filterstabilität.
*   **domains/vision/tracker.py**: Hochpräzises Tracking mittels Mediapipe Tasks API.

### Orchestrierungs-Skripte
*   **batch_reels.py**: Der Manager für Massenverarbeitung (Whisper -> LLM -> Rendering).
*   **video_highlight_pipeline_smart.py**: Die Kern-Logik für intelligentes Clip-Rendering.
*   **build_highlight_spans.py**: Generiert virale Titel und Social Media Captions.

---

## 3. Datenfluss & Workflow (Smart Pipeline)

1.  **Transkription**: `run_reel.sh` stellt sicher, dass WhisperX-Daten vorliegen.
2.  **Highlight-Extraktion**: Falls keine Highlights vorliegen, wird der `SermonSummarizer` genutzt.
3.  **Veredelung**: `build_highlight_spans.py` erstellt Titel und Texte.
4.  **Tracking & Schnitt**: Die Smart Pipeline berechnet den Bildausschnitt und führt den FFmpeg-Export durch.

---

## 4. Änderungshistorie (Changelog)

### Version 2.2 - Härtung & Qualitätssicherung (Aktuell)
*   **Konfigurations-Validierung**: `core/config.py` nutzt nun Pydantic-Field-Constraints (z.B. CRF 0-51, Volume 0-1) für "Fail-Fast"-Fehlererkennung.
*   **Zentraler CommandRunner**: Einführung von `core/commands.py` zur sicheren Ausführung externer Befehle (FFmpeg, FFprobe). Verbessert das Debugging durch automatische Stderr-Erfassung bei Fehlern.
*   **Test-Infrastruktur**: Einführung von `pytest` mit Units-Tests für Time-Utils, Config und Commands sowie einem Smoke-Test für die Pipeline-Integration.
*   **Abhängigkeitsmanagement**: Umstellung auf `uv` und Bereinigung der `pyproject.toml` (Fixing missing dependencies like `srt` and `loguru`).
*   **Bugfixes**: Behebung von Syntaxfehlern in `srt_summarizer_local.py` und Portabilitätsproblemen in der Subprocess-Logik.

### Version 2.1 - Refactoring & Konsolidierung
*   **Zentralisierung der Zeit-Helfer**: Einführung von `core/time_utils.py`. Alle Skripte nutzen nun einheitliche Parser und Formatierer für Zeitstempel.
*   **Prompt-Management**: Auslagerung aller LLM-Prompts in `core/prompts.py`. Verhindert Redundanz und ermöglicht einfache Stil-Anpassungen.
*   **Code-Bereinigung**: Entfernung doppelter Funktionen (z.B. `json_to_srt`) in `batch_reels.py`.
*   **Modul-Schnittstellen**: Anpassung aller Importe an die neue `core/` Struktur.

### Version 2.0 - Tracking & Stabilität
*   **Migration auf Mediapipe Tasks API**: Volle Kompatibilität mit Python 3.14+.
*   **Intelligentes Tracking**: Shot Detection, Multi-Face Scoring und cinematic Smoothing implementiert.
*   **Keyframe-Reduktion**: FFmpeg-Filteroptimierung zur Vermeidung von Fehlern bei langen Clips.
*   **Environment-Fixes**: Reparatur der `.venv` und automatisierte Erkennung in `run_reel.sh`.
