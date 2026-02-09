# Onset Refinement Feature

**Status:** ✅ Implemented (v0.5.0)
**Date:** 2026-02-09

## Overview

The onset refinement feature improves timing accuracy by aligning model predictions to audio onsets detected using librosa's signal processing algorithms. This is an **optional, experimental feature** designed to help evaluate whether audio-based onset detection can improve the model's timing accuracy.

## How It Works

### The Problem
- The model predicts drum hits from frame-level probabilities
- These predictions may have timing errors (early/late by 10-100ms)
- Timing accuracy is critical for realistic drum transcription

### The Solution
1. **Detect onsets** in the audio using librosa's spectral flux onset detector
2. **Match predictions** to nearest detected onset within a threshold (default 50ms)
3. **Snap predictions** to detected onset times if within threshold
4. **Keep originals** for predictions without nearby onsets

### Why Onset Detection (Not Beat Detection)?
- **Beat detection** finds the musical pulse (quarter notes, downbeats)
- **Onset detection** finds individual transients/hits in the audio
- Drums often play between beats (fills, ghost notes, syncopation)
- Onset detection captures every hit, making it perfect for drums

## Usage

### Python API

```python
from drum_transcription_api.transcription import DrumTranscriber

transcriber = DrumTranscriber(checkpoint_path, device='cpu')

# Baseline (no refinement)
baseline_stats = transcriber.transcribe_to_midi(
    audio_path='drums.wav',
    output_midi_path='baseline.mid',
    refine_with_onsets=False,  # Default
)

# With onset refinement
refined_stats = transcriber.transcribe_to_midi(
    audio_path='drums.wav',
    output_midi_path='refined.mid',
    refine_with_onsets=True,           # Enable refinement
    refinement_threshold=0.05,          # 50ms snap threshold
    onset_delta=0.05,                   # Onset detection sensitivity
)

# Check refinement statistics
if 'refinement' in refined_stats:
    print(f"Snapped: {refined_stats['refinement']['snapped']}")
    print(f"Kept: {refined_stats['refinement']['kept']}")
    print(f"Snap rate: {refined_stats['refinement']['snap_rate']*100:.1f}%")
```

### FastAPI Endpoint

```bash
# Baseline transcription
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@drums.wav" \
  -F "refine_with_onsets=false"

# With onset refinement
curl -X POST "http://localhost:8000/transcribe" \
  -F "file=@drums.wav" \
  -F "refine_with_onsets=true" \
  -F "refinement_threshold=0.05" \
  -F "onset_delta=0.05"
```

### Test Script

A standalone test script is provided for easy comparison:

```bash
# Compare baseline vs refined on a test file
./test_onset_refinement.py /path/to/test_audio.wav
```

This will:
- Transcribe the same file twice (baseline and refined)
- Save both MIDI files for comparison
- Print statistics showing the difference

## Parameters

### `refine_with_onsets` (bool)
- **Default:** `False`
- **Description:** Enable onset refinement
- Set to `True` to enable the feature

### `refinement_threshold` (float)
- **Default:** `0.05` (50ms)
- **Range:** 0.01 - 0.2 seconds
- **Description:** Maximum distance to snap prediction to onset
- Lower values = more conservative (only snap very close predictions)
- Higher values = more aggressive (snap predictions further away)

### `onset_delta` (float)
- **Default:** `0.05`
- **Range:** 0.01 - 0.2
- **Description:** Onset detection sensitivity parameter (librosa peak picking threshold)
- Lower values = detect more onsets (sensitive)
- Higher values = detect fewer onsets (conservative)

## Refinement Statistics

When `refine_with_onsets=True`, the API returns refinement statistics:

```python
{
    'refinement': {
        'snapped': 145,              # Number of predictions snapped to onsets
        'kept': 23,                  # Number of predictions kept (no nearby onset)
        'total': 168,                # Total predictions
        'snap_rate': 0.863,          # Proportion snapped (86.3%)
        'threshold': 0.05,           # Threshold used (50ms)
        'detected_onsets': 152,      # Total onsets detected in audio
    }
}
```

## Evaluation Strategy

To determine if onset refinement improves results:

### 1. Qualitative Comparison
- Transcribe test files with and without refinement
- Listen to both MIDI files in a DAW
- Compare timing to original audio
- **Expected improvement:** Tighter alignment to transients

### 2. Quantitative Comparison (Future)
Use the evaluation framework from `PLAN_ONSET_TIMING_EVALUATION.md`:
- Compare against ground truth MIDI (E-GMD dataset)
- Measure timing error, F1 score, precision/recall
- Calculate metrics for baseline vs refined
- **Target:** Mean timing error reduction >20%

### 3. Analysis Questions
- What is the snap rate? (Should be 70-90% for good onset detection)
- Does refinement improve timing for all drum types?
- Are there cases where refinement makes it worse?
- Does onset detection miss important hits?

## Example Workflow

```bash
# 1. Test on a single file
./test_onset_refinement.py test_drums.wav

# 2. Load both MIDI files in DAW (e.g., Reaper, Ableton)
# 3. Compare timing visually and audibly
# 4. Check refinement statistics in output

# 5. If helpful, enable by default:
#    - Set refine_with_onsets=True in API calls
#    - Update default in main.py
```

## Implementation Details

### Files Created/Modified

**Created:**
- `drum_transcription_api/onset_refinement.py` - Core refinement logic
- `test_onset_refinement.py` - Test script for comparison

**Modified:**
- `drum_transcription_api/transcription.py` - Added refinement to pipeline
- `drum_transcription_api/main.py` - Exposed parameters in API
- `PLAN_ONSET_TIMING_EVALUATION.md` - Updated with onset detection approach

### Algorithm Details

**Onset Detection** (librosa):
```python
onset_frames = librosa.onset.onset_detect(
    y=audio,
    sr=22050,
    hop_length=512,      # ~23ms frames
    backtrack=True,      # Refine onset times by backtracking
    delta=0.05,          # Peak picking threshold
    units='frames'
)
```

**Snapping Logic**:
```python
for prediction_time, drum, velocity in predictions:
    nearest_onset = find_nearest(detected_onsets, prediction_time)
    distance = abs(nearest_onset - prediction_time)

    if distance <= threshold:
        use_time = nearest_onset  # Snap to onset
    else:
        use_time = prediction_time  # Keep original
```

## Limitations

1. **Depends on onset detection quality**
   - May miss subtle hits (ghost notes, rimshots)
   - May detect false onsets (bleed from other instruments)

2. **Single snap threshold for all drums**
   - Could benefit from per-drum-type thresholds
   - Kick/snare might need different settings than cymbals

3. **No drum-specific onset detection**
   - Detects all onsets in audio, not per-drum
   - Could snap kick to snare onset if very close

4. **Experimental feature**
   - Needs quantitative evaluation on E-GMD dataset
   - May not improve all cases

## Future Improvements

1. **Per-drum-type thresholds**
   - Different snap thresholds for kick, snare, cymbals
   - Based on empirical timing error analysis

2. **Frequency-aware onset detection**
   - Detect onsets in specific frequency bands
   - Match kick predictions to low-freq onsets only

3. **Confidence-weighted snapping**
   - Only snap low-confidence predictions
   - Keep high-confidence predictions unchanged

4. **Integration with evaluation metrics**
   - Auto-tune thresholds based on ground truth
   - Report timing improvement quantitatively

## References

- **Librosa onset detection:** https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html
- **Spectral flux:** Standard method for onset detection in MIR
- **Evaluation plan:** `PLAN_ONSET_TIMING_EVALUATION.md`
