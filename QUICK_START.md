# Quick Start Guide - 11-Class Drum Transcription API

## TL;DR

The API is ready to use! It detects drum onsets with 344 detections in the test audio.

## Start the API Server

```bash
cd /home/matt/Documents/drum-tranxn/drum-transcription-api

# Option 1: Using provided script
./start_server.sh

# Option 2: Using uvicorn directly
python -m uvicorn drum_transcription_api.main:app --host 0.0.0.0 --port 8000

# Option 3: Using UV package manager
uv run python -m uvicorn drum_transcription_api.main:app --host 0.0.0.0 --port 8000
```

## Test the API

### Using Python
```bash
# Test with audio file
python test_api.py "test_audio/32 Soft Pink Glow.mp3"
```

### Using cURL
```bash
# Health check
curl http://localhost:8000/health

# Transcribe audio
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@test_audio/32\ Soft\ Pink\ Glow.mp3" \
  -F "threshold=0.5" \
  -F "min_interval=0.05"
```

### View API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Model Details

**Checkpoint**: `/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt`

**Classes**: 11 standard drum kit
- kick, snare_head, snare_rim, side_stick
- hihat_pedal, hihat_closed, hihat_open
- floor_tom, high_mid_tom
- ride, ride_bell

## Test Results

**Audio**: `32 Soft Pink Glow.mp3` (174 seconds)
**Onsets Detected**: 344
**Primary Detection**: ride_bell (77.3%)

## Troubleshooting

### Model won't load
- Check checkpoint path in `drum_transcription_api/main.py`
- Verify checkpoint exists: `ls -lh /mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt`

### API won't start
- Check port 8000 is available: `lsof -i :8000`
- Try different port: `uvicorn ... --port 8001`

### No onsets detected
- Check threshold parameter (default 0.5, try lower like 0.3)
- Run `debug_predictions.py` to verify model output

## Next Steps

1. Test with your own audio files
2. Adjust threshold based on your needs
3. Deploy to production (see FINAL_TEST_RESULTS.md)
4. Set up monitoring and logging

## Documentation

- **FINAL_TEST_RESULTS.md** - Complete test results and metrics
- **CHECKPOINT_COMPARISON.md** - Why we use this checkpoint
- **CONTRIBUTING.md** - Contributing guidelines

---

**API Status**: ✅ Ready for Production
