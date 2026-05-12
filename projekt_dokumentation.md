# Architektur-Dokumentation: AI Reels Pipeline

Diese Dokumentation beschreibt die Architektur, die Modulverantwortlichkeiten und den Workflow der AI Reels Pipeline (ehemals WhisperX Reel Pipeline), inklusive der neuesten Refactoring-Maßnahmen.

## 1. Ordnerstruktur

Die Codebasis ist modular aufgebaut und trennt Konfiguration, Domänenlogik und Orchestrierungs-Skripte.

```text
/AI-Reels
├── core/
│   ├── commands.py         # Sicherer Subprocess-Wrapper & Prozess-Registry
│   ├── config.py           # Zentrale Konfiguration (Pydantic-Settings)
│   ├── prompts.py          # Zentrale LLM-Prompts
│   └── time_utils.py       # Einheitliche Zeit-Helfer
├── domains/                # Domänenspezifische Logik
│   ├── highlights/         # Markdown-Parsing von Zeitstempeln
│   ├── llm/                # LLM-Orchestrierung (Ollama/Gemini)
│   ├── transcription/      # Karaoke-Generierung & WhisperX-Logik
│   ├── video/              # FFmpeg-Filter & Rendering
│   └── vision/             # Face- & Pose-Tracking (Mediapipe)
├── main.py                 # Zentraler CLI-Einstiegspunkt
├── batch_reels.py          # Orchestrierung des Batch-Workflows
├── video_highlight_pipeline_smart.py  # Kern-Logik für intelligentes Rendering
└── README.md               # Projekt-Übersicht & Setup
```

---

## 2. Modul-Logik & Zuständigkeiten

### Core-Modul (Infrastruktur)
*   **core/config.py**: Verwaltung aller Settings via Pydantic.
*   **core/commands.py**: **ProcessRegistry** trackt aktive Subprozesse. Ermöglicht sauberes Beenden von FFmpeg/WhisperX bei Programmabbruch (SIGINT/SIGTERM).
*   **core/time_utils.py**: Einheitliche Zeitberechnungen.

### Domains (Wiederverwendbare Fachlogik)
*   **domains/video/renderer.py**: **High-Performance Tracking**. Nutzt binäre Suchbäume für FFmpeg-Filter-Interpolation (O(log N) statt O(N)).
*   **domains/transcription/processor.py**: Präzises Snapping von Clip-Zeiten an WhisperX-Wortgrenzen.

### Orchestrierung
*   **main.py**: Neuer, sauberer CLI-Entrypoint mit `argparse`.
*   **batch_reels.py**: Unterstützt jetzt **paralleles Rendering** via `ThreadPoolExecutor` (standardmäßig 3 Arbeiter).

---

## 3. Änderungshistorie (Changelog)

### Version 2.3 - AI Reels Neustart & Performance (Heute)
*   **Rebranding**: Projekt in **"AI Reels"** umbenannt und Repository auf GitHub neu initialisiert.
*   **CLI-Revolution**: `main.py` als zentraler Einstiegspunkt implementiert. Vollständige Argument-Validierung und Routing.
*   **Performance-Boost**:
    *   **Paralleles Rendering**: `batch_reels.py` nutzt nun `ThreadPoolExecutor` für gleichzeitiges Rendern mehrerer Clips.
    *   **FFmpeg-Optimierung**: Binärer Suchbaum für Tracking-Koordinaten in `renderer.py` implementiert. Reduziert Parse-Last für FFmpeg massiv.
    *   **Keyframe-Limiting**: Begrenzung auf max. 20 Tracking-Punkte pro Clip für maximale Stabilität.
*   **Sicherheit & Robustheit**:
    *   **CWE-22 Mitigation**: Pfad-Sanitierung (`sanitize_stem`) in `batch_reels.py` verhindert Path-Traversal-Angriffe durch Dateinamen.
    *   **Signal Handling**: Globales Abfangen von `SIGINT` und `SIGTERM` sorgt für sauberes Beenden aller Hintergrundprozesse (FFmpeg/Whisper).
    *   **Abhängigkeits-Schlankheitskur**: `pyproject.toml` radikal vereinfacht. WhisperX wird nun als direkte Git-Abhängigkeit geladen.
    *   **Python 3.13 Protection**: Kompatibilität auf Python 3.10 bis 3.12 eingeschränkt, um ML-Bibliothek-Konflikte zu vermeiden.
*   **Code-Qualität**:
    *   Vollständiges Refactoring von `batch_reels.py` in modulare Sub-Funktionen zur besseren Testbarkeit.
    *   Umstellung auf Python-Standardbibliothek (`statistics.median`) in der Rendering-Pipeline.
    *   Professionelle `README.md` mit Fokus auf Features und `uv`-Setup erstellt.

### Version 2.2 - Härtung & Qualitätssicherung
*   **Konfigurations-Validierung**: Pydantic-Field-Constraints für alle Settings.
*   **Zentraler CommandRunner**: Einführung von `core/commands.py` mit Stderr-Erfassung.
*   **Test-Infrastruktur**: Einführung von `pytest` mit 100% Erfolg bei Core-Komponenten.
