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
    "checkpoint": "/mnt/hdd/drum-tranxn/checkpoints/drum-transcription-roland-epoch=99-val_loss=0.0528.ckpt",
    "classes": ["kick", "snare_head", "snare_xstick", "snare_rim", "tom1_head", "tom1_rim", "tom2_head", "tom2_rim", "tom3_head", "tom3_rim", "tom4_head", "tom4_rim", "hihat_closed", "hihat_closed_edge", "hihat_open", "hihat_open_edge", "hihat_pedal", "crash1_bow", "crash1_edge", "crash2_bow", "crash2_edge", "ride_bow", "ride_edge", "ride_bell", "tambourine", "cowbell"],
    "midi_mapping": "Roland TD-17",
    "mapping_standard": "Roland TD-17 electronic drum kit",
    "total_classes": 26
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
      "snare_head": 10,
      "snare_xstick": 2,
      "snare_rim": 0,
      "tom1_head": 1,
      "tom1_rim": 0,
      "tom2_head": 2,
      "tom2_rim": 1,
      "tom3_head": 0,
      "tom3_rim": 0,
      "tom4_head": 0,
      "tom4_rim": 0,
      "hihat_closed": 12,
      "hihat_closed_edge": 3,
      "hihat_open": 2,
      "hihat_open_edge": 1,
      "hihat_pedal": 0,
      "crash1_bow": 1,
      "crash1_edge": 0,
      "crash2_bow": 0,
      "crash2_edge": 0,
      "ride_bow": 0,
      "ride_edge": 0,
      "ride_bell": 0,
      "tambourine": 0,
      "cowbell": 0
    },
    "duration": 180.5,
    "onsets": [
      [0.0, "kick", 95],
      [0.5, "hihat_closed", 78],
      [1.0, "snare_head", 82]
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
- **Note Mapping**: Roland TD-17 standard
- **Velocity**: 1-127 based on detection confidence
- **Duration**: 100ms per note (configurable)

### Roland TD-17 Mapping

| Class | MIDI Note | Drum Sound |
|-------|-----------|------------|
| 0 | 36 | Kick |
| 1 | 38 | Snare Head |
| 2 | 37 | Snare X-Stick/Side Stick |
| 3 | 40 | Snare Rim/Hi Floor Tom |
| 4 | 48 | Tom 1 Head (Hi-Mid Tom) |
| 5 | 50 | Tom 1 Rim (High Tom) |
| 6 | 45 | Tom 2 Head (Low Tom) |
| 7 | 47 | Tom 2 Rim (Low-Mid Tom) |
| 8 | 43 | Tom 3 Head (High Floor Tom) |
| 9 | 58 | Tom 3 Rim (Vibraslap) |
| 10 | 41 | Tom 4 Head (Low Floor Tom) |
| 11 | 39 | Tom 4 Rim (Hand Clap) |
| 12 | 42 | Hi-Hat Closed (Bow) |
| 13 | 22 | Hi-Hat Closed (Edge) - Roland specific |
| 14 | 46 | Hi-Hat Open (Bow) |
| 15 | 26 | Hi-Hat Open (Edge) - Roland specific |
| 16 | 44 | Hi-Hat Pedal |
| 17 | 49 | Crash 1 (Bow) |
| 18 | 55 | Crash 1 (Edge/Splash) |
| 19 | 57 | Crash 2 (Bow) |
| 20 | 52 | Crash 2 (Edge/Chinese) |
| 21 | 51 | Ride (Bow) |
| 22 | 59 | Ride (Edge) |
| 23 | 53 | Ride Bell |
| 24 | 54 | Tambourine |
| 25 | 56 | Cowbell |
