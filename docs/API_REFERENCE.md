# API Reference

## Overview

The Drum Transcription API provides endpoints for converting drum audio files to MIDI format using a trained CRNN model.

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication required.

## Endpoints

### Health Check

#### GET /health

Check the health status of the API and model.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

**Status Codes:**
- `200 OK` - API is healthy
- `503 Service Unavailable` - Model not loaded

### Root Information

#### GET /

Get API information and configuration.

**Response:**
```json
{
  "name": "Drum Transcription API",
  "version": "0.1.0",
  "description": "API for transcribing drum audio to MIDI using trained CRNN model",
  "endpoints": {
    "health": "/health",
    "transcribe": "/transcribe (POST)",
    "download": "/download/{filename}"
  },
  "model_info": {
    "checkpoint": "/mnt/hdd/drum-tranxn/checkpoints/full-training-epoch=99-val_loss=0.0528.ckpt",
    "classes": ["kick", "snare", "hihat", "hi_tom", "mid_tom", "low_tom", "crash", "ride"],
    "midi_mapping": {
      "kick": [35, 36],
      "snare": [37, 38, 40],
      "hihat": [42, 44, 46],
      "hi_tom": [48, 50],
      "mid_tom": [45, 47],
      "low_tom": [41, 43],
      "crash": [49, 52, 55, 57],
      "ride": [51, 53, 59]
    }
  }
}
```

### Transcribe Audio

#### POST /transcribe

Upload an audio file and receive MIDI transcription.

**Request:**
- Content-Type: `multipart/form-data`

**Parameters:**
| Name | Type | Required | Description | Constraints |
|------|------|----------|-------------|-------------|
| file | File | Yes | Audio file to transcribe | MP3, WAV, M4A, FLAC |
| threshold | float | No | Onset detection threshold | 0.1 - 1.0, default 0.5 |
| min_interval | float | No | Minimum time between onsets (seconds) | 0.01 - 0.5, default 0.05 |
| tempo | int | No | Output MIDI tempo (BPM) | 60 - 200, default 120 |
| use_alternative_notes | bool | No | Use alternative MIDI notes for variety | default false |

**Response:**
```json
{
  "success": true,
  "message": "Transcription completed successfully. 42 drum hits detected.",
  "midi_file_url": "/download/song_transcribed.mid",
  "statistics": {
    "total_hits": 42,
    "per_drum": {
      "kick": 8,
      "snare": 12,
      "hihat": 15,
      "hi_tom": 2,
      "mid_tom": 3,
      "low_tom": 1,
      "crash": 1,
      "ride": 0
    },
    "duration": 180.5,
    "onsets": [
      [0.0, "kick", 95],
      [0.5, "hihat", 78],
      [1.0, "snare", 82]
    ]
  }
}
```

**Status Codes:**
- `200 OK` - Transcription successful
- `400 Bad Request` - Invalid file type or parameters
- `413 Payload Too Large` - File too large
- `500 Internal Server Error` - Transcription failed
- `503 Service Unavailable` - Model not loaded

### Download MIDI

#### GET /download/{filename}

Download the transcribed MIDI file.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| filename | string | Yes | Name of the MIDI file to download |

**Response:**
- Content-Type: `audio/midi`
- Body: Binary MIDI file data

**Status Codes:**
- `200 OK` - File downloaded successfully
- `404 Not Found` - File not found

## Data Models

### TranscriptionResponse

```python
class TranscriptionResponse(BaseModel):
    success: bool
    message: str
    midi_file_url: Optional[str] = None
    statistics: Optional[dict] = None
```

### HealthResponse

```python
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
```

### Statistics

```python
class Statistics(BaseModel):
    total_hits: int
    per_drum: Dict[str, int]
    duration: float
    onsets: List[List[Union[float, str, int]]]
```

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error description"
}
```

### Common Error Types

| Error | Description | Solution |
|-------|-------------|----------|
| Unsupported file type | File format not supported | Use MP3, WAV, M4A, or FLAC |
| Transcriber not initialized | Model not loaded | Check model checkpoint path |
| File not found | MIDI file expired | Re-run transcription |
| Transcription failed | Processing error | Check audio file quality |

## Rate Limiting

Currently no rate limiting is implemented. Consider implementing for production use.

## File Size Limits

Maximum file size: 50MB (configurable in server settings)

## Supported Audio Formats

| Format | MIME Types | Extensions |
|--------|------------|------------|
| MP3 | audio/mpeg, audio/mp3 | .mp3 |
| WAV | audio/wav, audio/x-wav | .wav |
| M4A | audio/m4a | .m4a |
| FLAC | audio/flac | .flac |

## MIDI Output Format

- **Format**: Standard MIDI File (Type 1)
- **Channel**: 10 (drums)
- **Note Mapping**: General MIDI standard
- **Velocity**: 1-127 based on detection confidence
- **Duration**: 100ms per note (configurable)
