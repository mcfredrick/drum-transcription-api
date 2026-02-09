# Implementation Plan: Add Stem Separation to API

## Context

**Problem:** The hierarchical drum transcription model was trained on isolated drum tracks (E-GMD dataset), but real-world use cases involve full mixes with bass, vocals, guitar, etc. Testing revealed significant performance degradation on full mixes:

- **Full mix:** 1653 hits detected (543 false "ride" hits from guitar/synths)
- **Isolated drums:** 855 hits detected (realistic, no false ride detections)

**Solution:** Integrate Demucs source separation into the API to isolate drums before transcription.

**Goal:** Make stem separation optional but default-enabled in the API, allowing users to transcribe full mixes accurately.

---

## Requirements

### Functional Requirements

1. **Optional stem separation:** Controlled by API parameter `separate_stems` (default: `true`)
2. **Backward compatibility:** Existing API endpoints remain unchanged
3. **Performance:** Stem separation adds ~40 seconds for 3-minute song (acceptable trade-off for accuracy)
4. **Cleanup:** Temporary separated stems cleaned up after transcription
5. **Error handling:** Graceful fallback if separation fails

### Non-Functional Requirements

1. **Memory efficiency:** Process on CPU to avoid CUDA OOM during training
2. **Disk space:** Temporary storage for separated stems (~10MB per song)
3. **Logging:** Track separation time and success/failure

---

## Architecture

### High-Level Flow

```
Audio Upload
    ↓
[separate_stems=true?]
    ↓ Yes                        ↓ No
Demucs Separation          Direct Transcription
    ↓                              ↓
Isolated Drums            Full Mix Audio
    ↓                              ↓
Transcription ←──────────────────┘
    ↓
MIDI Output
```

### Components to Modify

1. **`transcription.py`** - Add stem separation method
2. **`main.py`** - Update `/transcribe` endpoint with new parameter
3. **Dependencies** - Add `demucs` and `soundfile` to requirements

---

## Implementation Steps

### Phase 1: Add Stem Separation to DrumTranscriber

**File:** `drum_transcription_api/transcription.py`

#### Step 1.1: Add Demucs imports and initialization

Add to top of file:
```python
import soundfile as sf
from pathlib import Path
import tempfile
import os
```

#### Step 1.2: Add `_separate_drums()` method

Add new method to `DrumTranscriber` class:

```python
def _separate_drums(self, audio_path: str) -> str:
    """
    Separate drums from full mix using Demucs.

    Args:
        audio_path: Path to full mix audio file

    Returns:
        Path to separated drums audio file (in temp directory)

    Raises:
        Exception: If separation fails
    """
    try:
        import torch
        from demucs.apply import apply_model
        from demucs.pretrained import get_model
        from demucs.audio import AudioFile

        logger.info(f"Separating drums from full mix using Demucs...")

        # Load model (cache after first load)
        if not hasattr(self, '_demucs_model'):
            logger.info("Loading Demucs model (first time only)...")
            self._demucs_model = get_model('htdemucs')
            self._demucs_model.cpu()
            self._demucs_model.eval()
            logger.info("✓ Demucs model loaded")

        # Load audio
        wav = AudioFile(audio_path).read(seek_time=0, duration=None, streams=0)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        # Separate (drums is index 0 in htdemucs)
        with torch.no_grad():
            sources = apply_model(self._demucs_model, wav[None], device='cpu', progress=False)[0]

        drums = sources[0].numpy()  # (channels, samples)

        # Save to temporary file
        temp_dir = tempfile.mkdtemp(prefix='demucs_')
        drums_path = os.path.join(temp_dir, 'drums.wav')
        sf.write(drums_path, drums.T, 44100, subtype='PCM_16')

        logger.info(f"✓ Drums separated and saved to {drums_path}")
        return drums_path

    except Exception as e:
        logger.error(f"Drum separation failed: {e}")
        raise
```

#### Step 1.3: Update `transcribe_to_midi()` method

Modify the existing method to support stem separation:

```python
def transcribe_to_midi(
    self,
    audio_path: str,
    output_midi_path: str,
    threshold: float = 0.5,
    min_interval: float = 0.05,
    tempo: int = 120,
    export_predictions: bool = False,
    separate_stems: bool = True,  # NEW PARAMETER
) -> Dict:
    """
    Complete pipeline: audio file → MIDI file using 12-class drum model.

    Args:
        audio_path: Path to input audio
        output_midi_path: Path to save MIDI file
        threshold: Onset detection threshold (0-1)
        min_interval: Minimum time between onsets (seconds)
        tempo: BPM for output MIDI
        export_predictions: Export raw model predictions for debugging
        separate_stems: Whether to separate drums before transcription (default: True)

    Returns:
        Dictionary with statistics
    """
    drums_path = None
    temp_dir = None

    try:
        # Separate drums if requested
        if separate_stems:
            logger.info("Stem separation enabled")
            drums_path = self._separate_drums(audio_path)
            audio_to_transcribe = drums_path
        else:
            logger.info("Stem separation disabled, transcribing full mix")
            audio_to_transcribe = audio_path

        # Run inference on isolated drums (or full mix)
        predictions = self.transcribe_drums(audio_to_transcribe)

        # Export predictions for debugging if requested
        if export_predictions:
            self._export_predictions(audio_to_transcribe, predictions)

        # Extract onsets
        onsets = self.extract_onsets(predictions, threshold, min_interval)

        # Export to MIDI
        self.create_midi(onsets, output_midi_path, tempo)

        # Return statistics with 12-class drum names
        stats = {name: 0 for name in DRUM_NAMES}
        for _, drum_name, _ in onsets:
            if drum_name in stats:
                stats[drum_name] += 1

        # Calculate duration based on model type
        input_frame_rate = 22050 / 512
        if self.model_type == 'hierarchical':
            output_frame_rate = input_frame_rate
        else:
            time_reduction = 2**4
            output_frame_rate = input_frame_rate / time_reduction

        duration = predictions.shape[0] / output_frame_rate

        return {
            "total_hits": len(onsets),
            "per_drum": stats,
            "duration": duration,
            "onsets": onsets,
            "stem_separation_used": separate_stems,  # NEW FIELD
        }

    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

    finally:
        # Cleanup temporary separated stems
        if drums_path and os.path.exists(drums_path):
            try:
                temp_dir = os.path.dirname(drums_path)
                import shutil
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary stems: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temporary files: {e}")
```

---

### Phase 2: Update API Endpoint

**File:** `drum_transcription_api/main.py`

#### Step 2.1: Update `/transcribe` endpoint

Modify the existing endpoint:

```python
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
    ),  # NEW PARAMETER
):
    """
    Transcribe drum audio to MIDI.

    Upload an audio file and receive the transcribed MIDI file.

    **Stem Separation (NEW):**
    - Default: `separate_stems=true` - Isolates drums before transcription (recommended for full mixes)
    - Set `separate_stems=false` - Transcribes full mix directly (only for isolated drum tracks)
    """
    if not transcriber:
        raise HTTPException(status_code=503, detail="Transcriber not initialized")

    # ... [existing validation code] ...

    try:
        # Save uploaded file
        with open(input_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        logger.info(f"Processing file: {file.filename} (separate_stems={separate_stems})")

        # Transcribe audio (with optional stem separation)
        statistics = transcriber.transcribe_to_midi(
            audio_path=input_path,
            output_midi_path=output_path,
            threshold=threshold,
            min_interval=min_interval,
            tempo=tempo,
            export_predictions=export_predictions,
            separate_stems=separate_stems,  # PASS NEW PARAMETER
        )

        # ... [rest of existing code] ...

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
```

#### Step 2.2: Update root endpoint documentation

```python
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
        "stem_separation": "Demucs (htdemucs) - default enabled",  # NEW
    }

    return {
        "name": "Drum Transcription API",
        "version": "0.4.0",  # BUMP VERSION
        "description": "API for transcribing drum audio to MIDI using hierarchical CRNN model with optional stem separation",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "download": "/download/{filename}",
        },
        "model_info": model_info,
        "features": {  # NEW SECTION
            "stem_separation": {
                "enabled": True,
                "default": True,
                "method": "Demucs htdemucs",
                "purpose": "Isolates drums from full mix for better accuracy"
            }
        }
    }
```

---

### Phase 3: Update Dependencies

**File:** `pyproject.toml`

Add to dependencies:

```toml
[project]
dependencies = [
    # ... existing dependencies ...
    "demucs>=4.0.0",
    "soundfile>=0.12.0",
]
```

Update with uv:
```bash
cd /home/matt/Documents/drum-tranxn/drum-transcription-api
uv pip install demucs soundfile
```

---

### Phase 4: Testing

#### Test 1: API with Stem Separation (Default)

```bash
cd /home/matt/Documents/drum-tranxn/drum-transcription-api

# Start server
./start_server.sh

# Test with stem separation (default)
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test_audio/32 Soft Pink Glow.mp3" \
  -F "threshold=0.5" \
  -F "tempo=120" \
  -F "separate_stems=true"

# Expected: ~855 hits (not 1653), realistic drum detection
```

#### Test 2: API without Stem Separation

```bash
# Test without stem separation (for isolated drum tracks)
curl -X POST http://localhost:8000/transcribe \
  -F "file=@test_comparison/short_clip/original_audio.wav" \
  -F "threshold=0.5" \
  -F "tempo=92" \
  -F "separate_stems=false"

# Expected: Works as before for isolated tracks
```

#### Test 3: Performance Comparison

Create test script `test_stem_separation.py`:

```python
#!/usr/bin/env python3
"""Test stem separation vs full mix transcription."""

import requests
import time

API_URL = "http://localhost:8000/transcribe"
TEST_FILE = "test_audio/32 Soft Pink Glow.mp3"

def test_transcription(separate_stems: bool):
    """Test transcription with/without stem separation."""
    print(f"\nTesting separate_stems={separate_stems}...")

    start = time.time()

    with open(TEST_FILE, 'rb') as f:
        response = requests.post(
            API_URL,
            files={'file': f},
            data={
                'threshold': 0.5,
                'tempo': 120,
                'separate_stems': str(separate_stems).lower()
            }
        )

    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        stats = data['statistics']
        print(f"✓ Success in {elapsed:.1f}s")
        print(f"  Total hits: {stats['total_hits']}")
        print(f"  Per drum: {stats['per_drum']}")
    else:
        print(f"✗ Failed: {response.text}")

# Run tests
print("=" * 70)
print("STEM SEPARATION COMPARISON TEST")
print("=" * 70)

test_transcription(separate_stems=True)   # With separation
test_transcription(separate_stems=False)  # Without separation

print("\n" + "=" * 70)
```

---

## Success Criteria

### Functional

- ✅ API accepts `separate_stems` parameter (default: `true`)
- ✅ Stem separation reduces false positives (ride: 543 → ~0)
- ✅ Transcription accuracy improves significantly on full mixes
- ✅ Backward compatible with isolated drum tracks
- ✅ Temporary files cleaned up after transcription

### Performance

- ✅ Stem separation completes in <60 seconds for 3-minute song
- ✅ Memory usage acceptable (<2GB peak)
- ✅ No impact on transcription speed after separation

### Quality

- ✅ Full mix with separation: ~855 hits (realistic)
- ✅ Full mix without separation: ~1653 hits (over-detection)
- ✅ Isolated tracks work identically with/without flag

---

## Rollout Plan

### Development

1. Implement Phase 1 (transcription.py changes)
2. Test locally with test files
3. Implement Phase 2 (main.py changes)
4. Test API endpoints
5. Update documentation

### Testing

1. Unit tests for `_separate_drums()` method
2. Integration tests for full pipeline
3. Performance benchmarks
4. Comparison with previous results

### Deployment

1. Update dependencies in production
2. Deploy updated API
3. Monitor performance and errors
4. Update client documentation

---

## Future Enhancements

### Optional Improvements

1. **Model selection:** Allow users to choose separation model (htdemucs, htdemucs_ft, etc.)
2. **Caching:** Cache separated stems for repeated transcriptions
3. **Streaming:** Support real-time stem separation and transcription
4. **Quality settings:** Trade off speed vs quality (faster models available)
5. **Multi-stem output:** Return separated stems to user (drums, bass, etc.)

### Alternative Approaches

1. **Spleeter:** Faster but lower quality than Demucs
2. **Open-Unmix:** Lightweight alternative
3. **Hybrid:** Use fast model first, fall back to Demucs if quality poor

---

## Risk Mitigation

### Known Risks

1. **Processing time:** Stem separation adds 30-60 seconds
   - **Mitigation:** Make it optional, document performance trade-off

2. **Memory usage:** Demucs requires ~1-2GB RAM
   - **Mitigation:** Use CPU mode, clean up temporary files

3. **Dependency conflicts:** Demucs has many dependencies
   - **Mitigation:** Use separate venv, test thoroughly

4. **Model download:** First run downloads 80MB model
   - **Mitigation:** Pre-download in startup script, cache in persistent storage

---

## Documentation Updates

### API Documentation

Update `/docs` endpoint with:
- New `separate_stems` parameter
- Performance implications
- Use cases (when to enable/disable)
- Example requests

### README

Add section:
```markdown
## Stem Separation (NEW in v0.4.0)

The API now supports automatic drum isolation from full mixes using Demucs.

**Default behavior:** Stems are separated before transcription (recommended)

**To disable:** Set `separate_stems=false` (only for isolated drum tracks)

**Performance:** Adds 30-60 seconds for 3-minute song, but dramatically improves accuracy.
```

---

## Verification Checklist

Before marking complete:

- [ ] `_separate_drums()` method implemented
- [ ] `transcribe_to_midi()` updated with `separate_stems` parameter
- [ ] API endpoint accepts new parameter
- [ ] Dependencies updated (demucs, soundfile)
- [ ] Temporary files cleaned up properly
- [ ] Error handling for separation failures
- [ ] Tests pass (with and without separation)
- [ ] Documentation updated
- [ ] Performance benchmarked
- [ ] Comparison test shows improvement

---

## Estimated Effort

- **Phase 1:** 2-3 hours (transcription.py changes)
- **Phase 2:** 1-2 hours (API endpoint updates)
- **Phase 3:** 30 minutes (dependencies)
- **Phase 4:** 2-3 hours (testing and validation)

**Total:** 6-9 hours

---

## References

- Demucs: https://github.com/facebookresearch/demucs
- Previous test results: `test_comparison/COMPARISON_ANALYSIS.md`
- API codebase: `/home/matt/Documents/drum-tranxn/drum-transcription-api`
