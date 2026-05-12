# 🎬 WhisperX Reel Pipeline v2.2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-BSD--2--Clause-orange.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen.svg)](tests/)

Eine hochautomatisierte Pipeline zur Erstellung von viralen Social Media Reels (TikTok, Instagram, YouTube Shorts) aus langen Videoaufnahmen. Basierend auf **WhisperX** für präzise Transkription und **MediaPipe** für autonomes Bildausschnitt-Tracking.

---

## 🚀 Kern-Features

- **⚡ High-Speed Transkription**: Nutzt WhisperX (faster-whisper Backend) für 70x Echtzeit-Geschwindigkeit mit Wort-genauen Timestamps.
- **🤖 KI-Highlight Scouting**: Autonome Extraktion der besten Momente via LLM (Ollama lokal oder Gemini API).
- **🎯 Smart Face-Tracking**: Automatischer Crop von 16:9 auf 9:16 durch intelligentes MediaPipe-Tracking (Gesichts- und Posenerkennung).
- **🎶 Dynamic Audio Ducking**: Automatisches Absenken der Hintergrundmusik während gesprochen wird.
- **🎨 Karaoke-Subtitles**: Generierung von animierten ASS-Untertiteln für maximale Retention.

---

## 📂 Projektstruktur

Das Projekt folgt einer modularen **Domain-Driven-Structure**:

```text
├── core/               # Infrastruktur, Config & Shared Utils
├── domains/            # Fachlogik (Transcription, Video, Vision, LLM)
├── tests/              # Umfassende Unit- & Smoke-Tests
├── batch_reels.py      # Manager für Massenverarbeitung
├── run_reel.sh         # Einfacher Einstiegspunkt für Einzelvideos
└── smart_pipeline.py   # Kern-Logik für autonomes Rendering
```

---

## 🛠 Setup & Installation

Das Projekt nutzt [uv](https://docs.astral.sh/uv/) für ein blitzschnelles Abhängigkeitsmanagement.

### 1. Repository klonen
```bash
git clone <dein-repo-url>
cd whisperXBuild
```

### 2. Umgebung einrichten
```bash
uv sync --all-extras
```

### 3. Konfiguration
Erstelle eine `.env` Datei basierend auf der `.env.example`:
```bash
cp .env.example .env
# Trage hier deinen GEMINI_API_KEY ein (optional)
```

---

## 📋 Nutzung

### Einzelnes Video verarbeiten
```bash
./run_reel.sh input/mein_video.mp4
```

### Batch-Verarbeitung (Massen-Highlights)
```bash
uv run batch_reels.py --video input/predigt.mp4
```

### Intelligentes Highlight-Rendering
Die Pipeline analysiert das Video, findet Highlights, generiert Titel & Captions und rendert die finalen Clips mit automatischem Fokus auf den Sprecher.

---

## ⚙️ Konfiguration

Alle Parameter können in der `.env` oder direkt in der `core/config.py` angepasst werden:
- `FFMPEG_CRF`: Videoqualität (18-28 empfohlen).
- `LLM_PROVIDER`: `ollama` (lokal) oder `gemini` (Cloud).
- `FACE_TRACK`: Aktiviert/Deaktiviert das autonome Tracking.

---

## 🧪 Qualitätssicherung

Die Pipeline ist durch automatisierte Tests abgesichert:
```bash
uv run pytest
```

---

## 📄 Lizenz

Dieses Projekt ist unter der **BSD-2-Clause Lizenz** lizenziert - siehe [LICENSE](LICENSE) für Details.

---
*Entwickelt für effizientes Content-Repurposing von Predigten, Vorträgen und Podcasts.*
