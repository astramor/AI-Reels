# PythonReels – WhisperX Reel Pipeline

Kompakte technische Dokumentation für die aktive Pipeline. Sie erzeugt aus langen Videos bzw. Predigten 9:16-Reels mit Karaoke-Untertiteln, Face-Tracking und EBU-R128-Audio-Normalisierung.

## Architektur

- `batch_reels.py` – Orchestrator für Batch-Verarbeitung: Transkription, Summarizer, Highlight-Spans, Rendering.
- `srt_summarizer_local.py` – Erzeugt `summary.md`, `highlights_longlist.md`, `highlights.md` und `highlights.json`; optional gefilterte Sermon-SRT über `--export-sermon-srt`.
- `build_highlight_spans.py` – Berechnet Schnittfenster (`highlight_spans.md`, `reel_spans.json`) sowie virale Titel und Social-Media-Captions (`reels_content.md`).
- `video_highlight_pipeline_smart.py` – Aktive Rendering-Pipeline für 9:16-Reels, Karaoke-ASS-Untertitel, Face-Tracking und FFmpeg-Audio-Mixing.
- `core/config.py` – Zentrale Settings- und YAML-Ladelogik (`load_yaml_defaults()`).
- `core/llm_client.py` – Zentrale LLM-Anbindung (`UnifiedLLM`) mit Provider-/Fallback-Logik.
- `core/prompts.py` – Provider-spezifische MAP/REDUCE/CHUNK/MERGE-Prompts.
- `domains/` – Fachlogik: `transcription/`, `highlights/`, `video/`, `vision/`, `llm/`.

`main.py` ist ein optionaler Einzelvideo-CLI-Entrypoint und wird von `pyproject.toml` als `pythonreels`-Konsolenkommando referenziert.

## Ordnerstruktur

```text
core/          # Config, LLM-Client, Prompts, Commands, Artifacts, Zeit-Utils
domains/       # Transcription, Highlights, Video, Vision, LLM
tests/         # Unit- und Smoke-Tests
input/         # Quellvideos / SRT-Dateien
raw_videos/    # Batch-Input-Fallback
out_srt/       # exportierte / gefilterte SRT-Dateien
out_summ/      # summary, highlights, spans
out_reels/     # finale Reels
out_clips/     # alternative Clip-Ausgabe
music/         # Hintergrundmusik
config.yaml    # aktive Konfiguration
example.env    # Beispiel für Umgebungsvariablen
old_files/     # Altlasten, für Refactorings ignorieren
```

## Umgebungsvariablen

API-Keys werden primär aus der Umgebung gelesen. Leere YAML-Keys in `config.yaml` überschreiben gesetzte Umgebungsvariablen nicht.

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GEMINI_API_KEY=...
```

Weitere wichtige Variablen:

- `LLM_PROVIDER` – `openai`, `openrouter`, `gemini-api`/`gemini`, `ollama`, `llama-server`
- `LLM_MODEL` – allgemeiner Modellname
- provider-spezifisch: `OPENAI_MODEL`, `OPENROUTER_MODEL`, `GEMINI_MODEL`, `OLLAMA_MODEL`, `LLAMA_SERVER_MODEL`
- `OPENAI_BASE_URL` – OpenAI-kompatibler Endpoint (z. B. OpenRouter oder lokaler llama-server)
- `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_LANGUAGE`, `WHISPER_COMPUTE_TYPE`
- `FFMPEG_CRF`, `FFMPEG_PRESET`, `FFMPEG_AUDIO_BITRATE`
- `FACE_TRACK`, `FACE_KEYS`, `FACE_SAMPLE_RATE`

## Nutzung

Batch-Verarbeitung:

```bash
.venv/bin/python batch_reels.py --input_dir input --out_root .
```

Einzelvideo:

```bash
./run_reel.sh input/mein_video.mp4
```

oder:

```bash
.venv/bin/python main.py --video input/mein_video.mp4
```

Summarizer direkt:

```bash
.venv/bin/python srt_summarizer_local.py \
  --srt input/mein_video.srt \
  --out_dir out_summ \
  --max-highlights 8 \
  --llm-provider openai \
  --model ds41 \
  --export-sermon-srt out_srt/mein_video_predigt.srt
```

Spans und Titel:

```bash
.venv/bin/python build_highlight_spans.py \
  --highlights out_summ/highlights.json \
  --srt input/mein_video.srt \
  --out-md out_summ/highlight_spans.md \
  --out-json out_summ/reel_spans.json
```

Rendering:

```bash
.venv/bin/python video_highlight_pipeline_smart.py \
  --video input/mein_video.mp4 \
  --spans_json out_summ/reel_spans.json \
  --out_dir out_reels
```

## Tests

```bash
OPENAI_API_KEY=testenv .venv/bin/python -m pytest -q
```

Die Pipeline verwendet EBU-R128-nahe Audio-Normalisierung (`-14 LUFS`, True-Peak `-1.0 dB`) und reduziert Face-Tracking-Keyframes, um überlange FFmpeg-Filtergraphen zu vermeiden.

## Lizenz

BSD-2-Clause – siehe `LICENSE`.
