# Drum Transcription API - Usage Examples

## Quick Start

1. **Start the server:**
   ```bash
   ./start_server.sh
   # or
   uv run uvicorn drum_transcription_api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Check API health:**
   ```bash
   curl http://localhost:8000/health
   ```

3. **View API documentation:**
   Open http://localhost:8000/docs in your browser

## API Usage Examples

### Using curl

**Basic transcription:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_drums.mp3" \
  -F "threshold=0.5" \
  -F "min_interval=0.05" \
  -F "tempo=120"
```

**With alternative MIDI notes for variety:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_drums.mp3" \
  -F "threshold=0.6" \
  -F "min_interval=0.03" \
  -F "tempo=140" \
  -F "use_alternative_notes=true"
```

### Using Python

```python
import requests

# Upload and transcribe audio file
with open('drums.mp3', 'rb') as f:
    files = {'file': ('drums.mp3', f, 'audio/mpeg')}
    data = {
        'threshold': 0.5,
        'min_interval': 0.05,
        'tempo': 120,
        'use_alternative_notes': False
    }
    
    response = requests.post(
        'http://localhost:8000/transcribe',
        files=files,
        data=data
    )

if response.status_code == 200:
    result = response.json()
    print(f"Transcription successful!")
    print(f"Total hits: {result['statistics']['total_hits']}")
    print(f"Per drum: {result['statistics']['per_drum']}")
    
    # Download MIDI file
    midi_url = f"http://localhost:8000{result['midi_file_url']}"
    midi_response = requests.get(midi_url)
    
    with open('output.mid', 'wb') as midi_file:
        midi_file.write(midi_response.content)
    print("MIDI file saved as output.mid")
else:
    print(f"Error: {response.text}")
```

### Using JavaScript/Fetch

```javascript
async function transcribeDrums(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);
    formData.append('threshold', '0.5');
    formData.append('min_interval', '0.05');
    formData.append('tempo', '120');
    formData.append('use_alternative_notes', 'false');

    try {
        const response = await fetch('http://localhost:8000/transcribe', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const result = await response.json();
            console.log('Transcription successful!', result.statistics);
            
            // Download MIDI file
            const midiResponse = await fetch(`http://localhost:8000${result.midi_file_url}`);
            const midiBlob = await midiResponse.blob();
            
            // Create download link
            const url = window.URL.createObjectURL(midiBlob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'transcribed_drums.mid';
            a.click();
        } else {
            console.error('Transcription failed:', await response.text());
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

// Usage with file input
document.getElementById('audioInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        transcribeDrums(file);
    }
});
```

## API Parameters

### Transcription Endpoint Parameters

- **file** (required): Audio file (MP3, WAV, M4A, FLAC)
- **threshold** (optional, default=0.5): Onset detection threshold (0.1-1.0)
  - Higher values = more selective, fewer false positives
  - Lower values = more sensitive, more hits detected
- **min_interval** (optional, default=0.05): Minimum time between onsets in seconds
  - Prevents double-triggering of the same drum
- **tempo** (optional, default=120): Output MIDI tempo in BPM (60-200)
- **use_alternative_notes** (optional, default=false): Use alternative MIDI notes for variety

### Response Format

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
            // ... more onsets
        ]
    }
}
```

## MIDI Note Mapping

The API outputs MIDI files compatible with your app:

| Drum Type | MIDI Notes | Description |
|-----------|------------|-------------|
| Kick | 35, 36 | Bass Drum 1, Bass Drum 2 |
| Snare | 37, 38, 40 | Side Stick, Acoustic Snare, Electric Snare |
| Hi-Hat | 42, 44, 46 | Closed Hi-Hat, Pedal Hi-Hat, Open Hi-Hat |
| High Tom | 48, 50 | Hi-Mid Tom, High Tom |
| Mid Tom | 45, 47 | Low Tom, Low-Mid Tom |
| Floor Tom | 41, 43 | Floor Tom (low), Floor Tom (high) |
| Crash | 49, 52, 55, 57 | Crash Cymbal 1, China Cymbal, Splash Cymbal, Crash Cymbal 2 |
| Ride | 51, 53, 59 | Ride Cymbal 1, Ride Bell, Ride Cymbal 2 |

All notes are on MIDI channel 10 (channel 9 in zero-indexed terms).

## Testing

Use the provided test script:

```bash
# Test API health
python test_api.py

# Test with audio file
python test_api.py /path/to/your/drums.mp3
```

## Troubleshooting

1. **Model not found**: Ensure the checkpoint path in `main.py` is correct
2. **CUDA out of memory**: Try using CPU by changing device to 'cpu' in main.py
3. **Unsupported audio format**: Convert to MP3 or WAV first
4. **No transcription results**: Try lowering the threshold parameter
