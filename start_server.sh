#!/bin/bash

# Drum Transcription API Startup Script

echo "Starting Drum Transcription API..."

# Check if model checkpoint exists
MODEL_CHECKPOINT="/mnt/hdd/drum-tranxn/checkpoints/full-training-epoch=99-val_loss=0.0528.ckpt"
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "WARNING: Model checkpoint not found at $MODEL_CHECKPOINT"
    echo "Please update the model_checkpoint path in drum_transcription_api/main.py"
    echo "The server will start but transcription will not work until the checkpoint is available."
fi

# Start the server
echo "Starting server on http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Health Check: http://localhost:8000/health"

uv run uvicorn drum_transcription_api.main:app --host 0.0.0.0 --port 8000 --reload
