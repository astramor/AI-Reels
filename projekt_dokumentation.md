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

### Version 2.4 - Workflow-Automatisierung & KI-Optimierung (Aktuell)
*   **One-Command-Workflow**: `main.py` und `video_highlight_pipeline_smart.py` wurden vollständig automatisiert.
    *   **Auto-Transkription**: Integrierter "Schritt 0" führt WhisperX automatisch aus, falls keine SRT-Datei vorhanden ist (Optimiert für RTX 4070 Ti SUPER: `large-v3`, `float16`, `cuda`).
    *   **Auto-Highlight-Generierung**: Falls `spans_md` fehlt, wird diese automatisch via LLM aus der Transkription erstellt.
    *   **Vereinfachte CLI**: Die Pipeline kann nun mit nur einem Parameter (`--video`) gestartet werden; alle Zwischenschritte werden intelligent abgeleitet.
*   **Gemini Pro Integration**:
    *   **Strukturierter Output**: Umstellung der Highlight-Extraktion auf natives JSON-Schema (`google-genai` SDK) für 100% verlässliche Zeitstempel.
    *   **Zweistufiger Prozess**: Implementierung eines Scout/Editor-Verfahrens zur besseren Filterung viraler Zitate.
*   **Robustheit**:
    *   Verbesserte Pfad-Propagation für nahtlosen Übergang zwischen Transkription, Zusammenfassung und Rendering.
    *   Integration von `shutil` für verlässliches Datei-Management generierter Metadaten.

### Version 2.3 - AI Reels Neustart & Performance

### Version 2.2 - Härtung & Qualitätssicherung
*   **Konfigurations-Validierung**: Pydantic-Field-Constraints für alle Settings.
*   **Zentraler CommandRunner**: Einführung von `core/commands.py` mit Stderr-Erfassung.
*   **Test-Infrastruktur**: Einführung von `pytest` mit 100% Erfolg bei Core-Komponenten.
