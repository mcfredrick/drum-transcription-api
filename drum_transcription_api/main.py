"""FastAPI server for drum transcription."""

import os
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from .transcription import DrumTranscriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
transcriber: Optional[DrumTranscriber] = None


# Response models
class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    midi_file_url: Optional[str] = None
    statistics: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="Drum Transcription API",
    description="API for transcribing drum audio to MIDI using hierarchical CRNN model with optional stem separation and onset refinement",
    version="0.5.0",
)


@app.on_event("startup")
async def startup_event():
    """Initialize the transcriber on startup."""
    global transcriber

    # Model checkpoint path - using best hierarchical model from completed training
    model_checkpoint = (
        "/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=99-val_kick_roc_auc=1.000.ckpt"
    )

    # Fallback to flat model if hierarchical not found
    if not os.path.exists(model_checkpoint):
        logger.warning(f"Hierarchical model not found: {model_checkpoint}")
        logger.warning("Falling back to flat model")
        model_checkpoint = (
            "/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt"
        )

    if not os.path.exists(model_checkpoint):
        logger.error(f"Model checkpoint not found: {model_checkpoint}")
        logger.error("Please update the model_checkpoint path in main.py")
        return

    try:
        transcriber = DrumTranscriber(model_checkpoint, model_type="auto")
        logger.info("Drum transcriber initialized successfully")
        logger.info(f"Model type: {transcriber.model_type}")
        if hasattr(transcriber.model, 'n_classes'):
            logger.info(f"Output classes: {transcriber.model.n_classes}")
    except Exception as e:
        logger.error(f"Failed to initialize transcriber: {e}")
        transcriber = None


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if transcriber else "unhealthy",
        model_loaded=transcriber is not None,
        device=transcriber.device if transcriber else "unknown",
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (MP3, WAV, etc.)"),
    threshold: float = Query(
        0.5, ge=0.1, le=1.0, description="Onset detection threshold"
    ),
    min_interval: float = Query(
        0.05, ge=0.01, le=0.5, description="Minimum time between onsets (seconds)"
    ),
    tempo: int = Query(120, ge=60, le=200, description="Output MIDI tempo (BPM)"),
    export_predictions: bool = Query(
        False, description="Export raw model predictions for debugging"
    ),
    separate_stems: bool = Query(
        True, description="Separate drums from full mix before transcription (recommended)"
    ),
    refine_with_onsets: bool = Query(
        False, description="Refine predictions using audio onset detection (experimental)"
    ),
    refinement_threshold: float = Query(
        0.05, ge=0.01, le=0.2, description="Max distance (seconds) to snap to detected onset"
    ),
    onset_delta: float = Query(
        0.05, ge=0.01, le=0.2, description="Onset detection sensitivity parameter"
    ),
):
    """
    Transcribe drum audio to MIDI.

    Upload an audio file and receive the transcribed MIDI file.

    **Stem Separation:**
    - Default: `separate_stems=true` - Isolates drums before transcription (recommended for full mixes)
    - Set `separate_stems=false` - Transcribes full mix directly (only for isolated drum tracks)

    **Onset Refinement (Experimental):**
    - Set `refine_with_onsets=true` - Aligns predictions to audio onsets detected by librosa
    - Useful for improving timing accuracy by snapping to detected transients
    - Adds refinement statistics to the response
    """
    if not transcriber:
        raise HTTPException(status_code=503, detail="Transcriber not initialized")

    # Validate file type
    allowed_types = [
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/mp3",
        "audio/m4a",
        "audio/flac",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed types: {allowed_types}",
        )

    # Create temporary files
    temp_dir = tempfile.mkdtemp()
    # Sanitize filename to handle spaces and special characters
    safe_filename = (
        Path(file.filename).stem.replace(" ", "_").replace("(", "").replace(")", "")
    )
    input_path = os.path.join(temp_dir, f"input_{safe_filename}.mp3")
    output_path = os.path.join(temp_dir, f"output_{safe_filename}.mid")

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Processing file: {file.filename} (separate_stems={separate_stems})")

        # Transcribe audio (with optional stem separation and onset refinement)
        statistics = transcriber.transcribe_to_midi(
            audio_path=input_path,
            output_midi_path=output_path,
            threshold=threshold,
            min_interval=min_interval,
            tempo=tempo,
            export_predictions=export_predictions,
            separate_stems=separate_stems,
            refine_with_onsets=refine_with_onsets,
            refinement_threshold=refinement_threshold,
            onset_delta=onset_delta,
        )

        # Generate download URL
        midi_filename = f"{Path(file.filename).stem}_transcribed.mid"
        midi_url = f"/download/{midi_filename}"

        # Store the file path for download (in production, use proper file storage)
        app.state.temp_files = getattr(app.state, "temp_files", {})
        app.state.temp_files[midi_filename] = output_path

        logger.info(
            f"Transcription completed: {statistics['total_hits']} hits detected"
        )

        return TranscriptionResponse(
            success=True,
            message=f"Transcription completed successfully. {statistics['total_hits']} drum hits detected." +
                    (" (Stems separated)" if separate_stems else " (Full mix)"),
            midi_file_url=midi_url,
            statistics=statistics,
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    finally:
        # Clean up input file
        if os.path.exists(input_path):
            os.remove(input_path)


@app.get("/download/{filename}")
async def download_midi(filename: str):
    """Download the transcribed MIDI file."""
    temp_files = getattr(app.state, "temp_files", {})

    if filename not in temp_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = temp_files[filename]

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    return FileResponse(path=file_path, filename=filename, media_type="audio/midi")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    from .transcription import DRUM_NAMES, DRUM_MAP

    model_info = {
        "checkpoint": "auto-detected",
        "type": transcriber.model_type if transcriber else "unknown",
        "classes": 12,
        "class_names": DRUM_NAMES,
        "midi_mapping": DRUM_MAP,
        "stem_separation": "Demucs (htdemucs) - default enabled",
    }

    return {
        "name": "Drum Transcription API",
        "version": "0.5.0",
        "description": "API for transcribing drum audio to MIDI using hierarchical CRNN model with optional stem separation and onset refinement",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "download": "/download/{filename}",
        },
        "model_info": model_info,
        "features": {
            "stem_separation": {
                "enabled": True,
                "default": True,
                "method": "Demucs htdemucs",
                "purpose": "Isolates drums from full mix for better accuracy"
            },
            "onset_refinement": {
                "enabled": True,
                "default": False,
                "method": "librosa onset detection",
                "purpose": "Refines timing by snapping predictions to detected audio onsets (experimental)"
            }
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "drum_transcription_api.main:app", host="0.0.0.0", port=8000, reload=True
    )
