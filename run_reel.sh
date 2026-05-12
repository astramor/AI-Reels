#!/bin/bash

# Prüfen, ob ein Video als Argument übergeben wurde
if [ -z "$1" ]; then
    echo "❌ Fehler: Bitte ein Video angeben."
    echo "👉 Nutzung: ./run_reel.sh input/mein_video.mp4"
    exit 1
fi

FILE="$1"
VIDEO_DIR=$(dirname "$FILE")
VIDEO_NAME=$(basename "$FILE")
BASENAME="${VIDEO_NAME%.*}"
SRT_PATH="${VIDEO_DIR}/${BASENAME}.srt"
OUT_DIR="out_clips"
MD_PATH="${OUT_DIR}/${BASENAME}_highlights.md"

echo "========================================"
echo "🚀 Verarbeite: $VIDEO_NAME"
echo "========================================"

# --- SCHRITT 1: Transkription (WhisperX) ---
if [ ! -f "$SRT_PATH" ]; then
    echo "🎙️ SRT fehlt. Starte Transkription für $VIDEO_NAME..."
    
    # Nutze whisperx (da whisper oft nicht installiert ist oder schlechter performt)
    # Wir nutzen die Parameter aus der config.yaml (cuda, no_cudnn)
    if [ -f ".venv/bin/whisperx" ]; then
        WHISPER_CMD=".venv/bin/whisperx"
    else
        WHISPER_CMD="whisperx"
    fi

    # CT2_USE_CUDNN=0 wird oft benötigt, wenn cuDNN Probleme macht (siehe config.yaml no_cudnn: true)
    export CT2_USE_CUDNN=0
    
    $WHISPER_CMD "$FILE" \
        --model large-v3 \
        --language de \
        --output_format srt \
        --output_dir "$VIDEO_DIR" \
        --device cuda \
        --compute_type float16 \
        --batch_size 16
    
    # Sicherstellen, dass die Datei jetzt existiert
    if [ ! -f "$SRT_PATH" ]; then
        # Fallback: Suche nach der generierten Datei
        GENERATED_SRT=$(ls "${VIDEO_DIR}/${BASENAME}"*.srt | head -n 1)
        if [ -n "$GENERATED_SRT" ] && [ -f "$GENERATED_SRT" ]; then
            mv "$GENERATED_SRT" "$SRT_PATH"
        fi
    fi
else
    echo "✅ SRT bereits vorhanden: $SRT_PATH"
fi

# --- SCHRITT 2: Smart Pipeline (Summarizer & Renderer) ---
echo "🎬 Starte KI-Analyse und Clips-Erstellung..."

if [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
else
    PYTHON_CMD="python"
fi

$PYTHON_CMD video_highlight_pipeline_smart.py \
    --video "$FILE" \
    --srt_input "$SRT_PATH" \
    --spans_md "$MD_PATH" \
    --out_dir "$OUT_DIR"

echo "========================================"
echo "🏁 Fertig! Die Clips liegen in: $OUT_DIR"
echo "========================================"
