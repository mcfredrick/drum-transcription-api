# Fix Summary: MIDI Duration Issue Resolution

## Problem
The drum transcription API was generating MIDI files with incorrect duration - only ~21 seconds instead of the full ~174 seconds of the input audio file.

## Root Cause Analysis
The issue was a **time scaling bug** in the `extract_onsets` function in `transcription.py`. The model was correctly processing the entire audio, but the time conversion from model output frames to seconds was wrong.

### Technical Details
- **Input spectrogram**: 7494 frames at 43.07 FPS = 174.01 seconds ✅
- **Model output**: 936 frames due to 8x pooling reduction from CNN layers ✅
- **Bug**: Using input frame rate (43.07 FPS) for output frames instead of output frame rate (5.38 FPS)

The model architecture includes 3 pooling layers with pool_size=2, resulting in 2^3 = 8x reduction in the time dimension.

## Solution Implemented

### 1. Fixed Frame Rate Calculation
**File**: `drum_transcription_api/transcription.py` - `extract_onsets()` method

**Before (WRONG)**:
```python
frame_rate = 22050 / 512  # ~43 FPS
```

**After (CORRECT)**:
```python
# Model output frame rate accounts for 8x pooling reduction
input_frame_rate = 22050 / 512  # ~43 FPS (input spectrogram)
pooling_reduction = 8  # 2^3 from three pooling layers
frame_rate = input_frame_rate / pooling_reduction  # ~5.38 FPS (model output)
```

### 2. Fixed Duration Calculation
**File**: `drum_transcription_api/transcription.py` - `transcribe_to_midi()` method

Updated duration calculation to use output frame rate:
```python
'duration': predictions.shape[0] / ((22050 / 512) / 8)  # Use output frame rate
```

### 3. Added Predictions Export Functionality
**New Feature**: Added `export_predictions` parameter to API and transcription pipeline

- **API endpoint**: Added `export_predictions` query parameter
- **Transcription method**: Added `_export_predictions()` helper method
- **Output files**: Generates `.npy` and `.json` files with raw model predictions
- **Debug information**: Includes frame rates, statistics, and sample predictions

## Results

### Before Fix
- MIDI duration: 21.55 seconds ❌
- Note density: ~32 notes/second (unrealistic)

### After Fix
- MIDI duration: 171.69 seconds ✅ (matches 174-second audio)
- Note density: 4.0 notes/second (realistic for drums)
- Total notes: 694 (same count, correctly distributed over full duration)

## Files Modified
1. `drum_transcription_api/transcription.py`
   - Fixed time scaling in `extract_onsets()`
   - Fixed duration calculation in `transcribe_to_midi()`
   - Added `_export_predictions()` method
   - Added `export_predictions` parameter

2. `drum_transcription_api/main.py`
   - Added `export_predictions` query parameter to API endpoint
   - Passed parameter through to transcription method

## Testing
- ✅ Verified correct MIDI duration matches audio duration
- ✅ Confirmed realistic note density (4.0 notes/second)
- ✅ Tested predictions export functionality
- ✅ Generated debug files for future analysis

## Impact
This fix resolves the core transcription accuracy issue, ensuring that drum transcriptions now cover the entire input audio duration rather than just the first ~12% of the audio.
