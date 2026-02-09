# 11-Class API - Test Results with Audio File

## Test Summary

**Audio File**: `32 Soft Pink Glow.mp3` (6.64 MB, 174 seconds)  
**Model**: 11-class drum structure (trial_0-epoch=00-val_loss=0.2757.ckpt)  
**Test Date**: 2026-02-08

---

## API Update Status

### ✅ COMPLETED
- Updated `/drum-transcription-api/drum_transcription_api/main.py` to use 11-class model checkpoint
- Updated `/drum-transcription-api/drum_transcription_api/transcription.py` with 11-class DRUM_MAP and DRUM_NAMES
- Removed incorrect API files from `/drum-transcription/` folder
- API is now configured to load the 11-class model on startup

### Modified Files
```
/drum-transcription-api/drum_transcription_api/main.py
  - Changed model checkpoint from Roland TD-17 (26-class) to 11-class model
  - Updated logging messages for 11-class structure

/drum-transcription-api/drum_transcription_api/transcription.py
  - Updated DRUM_MAP: 26 classes → 11 classes
  - Updated DRUM_NAMES to match 11-class structure
  - All MIDI mappings verified
```

---

## Test Results

### 1. Model Loading
✅ **PASSED**
- Model loads successfully from checkpoint
- Device: CUDA (GPU available)
- Classes: 11 (correct)

### 2. Audio Processing
✅ **PASSED**
- Audio file loads successfully (6.64 MB)
- Spectrogram extraction works
- Spectrogram shape: (128, 7494) - correct

### 3. Model Inference
⚠️ **ISSUE DETECTED**

**Raw Predictions Statistics:**

| Drum Class | Min | Max | Mean | Frames > 0.5 |
|:-----------|:----|:----|:-----|:------------|
| kick | 0.041 | 0.130 | 0.063 | 0 |
| snare_head | 0.046 | 0.098 | 0.054 | 0 |
| snare_rim | 0.063 | 0.243 | 0.090 | 0 |
| side_stick | 0.025 | 0.102 | 0.046 | 0 |
| hihat_pedal | 0.024 | 0.105 | 0.047 | 0 |
| hihat_closed | 0.016 | 0.270 | 0.098 | 0 |
| hihat_open | 0.004 | 0.204 | 0.044 | 0 |
| floor_tom | 0.046 | 0.187 | 0.071 | 0 |
| high_mid_tom | 0.039 | 0.160 | 0.060 | 0 |
| ride | 0.026 | 0.110 | 0.045 | 0 |
| ride_bell | 0.002 | 0.080 | 0.014 | 0 |

**Key Issues:**
1. **Very low predictions**: Maximum prediction across all classes is only 0.270 (hihat_closed)
2. **No onsets detected**: No predictions exceed 0.5 threshold
3. **Duration mismatch**: 
   - Spectrogram duration: 174.01 seconds ✓
   - Predictions duration: 21.73 seconds ✗
   - This is ~8x downsampling, suggesting pooling/stride issues in model

### 4. Onset Detection
❌ **0 ONSETS DETECTED**

With threshold 0.5: **0 onsets** (no predictions reached this level)

---

## Analysis

### Root Cause

The model appears to have been trained on a different dataset or with different audio preprocessing than what's being used for inference. The most likely issues are:

#### 1. **Audio Preprocessing Mismatch**
- The model expects normalized/preprocessed audio from the training distribution
- The test audio ("32 Soft Pink Glow") may have different:
  - Loudness/amplitude characteristics
  - Frequency content
  - Drum kit/recording setup
  - Audio quality/encoding

#### 2. **Poor Model Convergence**
- Max validation loss was 0.2757, which is relatively high
- The model may not have trained successfully
- Limited training data or incorrect hyperparameters

#### 3. **Temporal Pooling Issue**
- Duration mismatch (174s → 21.7s = 8x compression) suggests excessive pooling
- Model architecture may have too much downsampling

#### 4. **Audio Characteristics**
- The test song may not contain strong drum hits in the mix
- It could be a softer/ambient track with minimal percussion

---

## Recommendations

### For Testing

1. **Test with training data**: Run inference on data similar to what the model was trained on
2. **Lower threshold**: Try detection with threshold 0.1-0.2 (may increase false positives)
3. **Check audio preprocessing**: Verify the test audio matches training audio specs

### For Model Improvement

1. **Retrain the model**:
   - Verify training data is properly loaded (check class distribution)
   - Confirm model architecture matches expectations
   - Check loss function is behaving correctly
   - Increase training duration/epochs

2. **Normalize audio**:
   - Apply consistent preprocessing (normalization, loudness matching)
   - Verify training and inference use identical preprocessing

3. **Fix temporal dimension**:
   - Reduce pooling/stride operations if unnecessary
   - Verify model output shape calculations

4. **Validate training data**:
   - Check that training audio actually contains the drum events
   - Verify MIDI labels are correctly aligned to audio

---

## API Deployment Status

### Ready to Deploy
- ✅ API code is updated for 11-class structure
- ✅ Model checkpoint path is configured correctly
- ✅ Endpoints are implemented

### Required Before Production
- ⚠️ **Model needs retraining or validation**
- ⚠️ **Test with known good audio file**
- ⚠️ **Verify model training data and preprocessing**

---

## Next Steps

### Option 1: Verify Existing Training
```bash
# Check training logs
cd /home/matt/Documents/drum-tranxn/drum_transcription
# Review: optuna_11class_study.db, optuna_fixed.log

# Run inference on original E-GMD training data
python scripts/transcribe.py \
  --checkpoint /mnt/hdd/drum-tranxn/optuna_checkpoints/trial_0/trial_0-epoch=00-val_loss=0.2757.ckpt \
  --audio /path/to/e_gmd/audio.mp3 \
  --output output.mid
```

### Option 2: Retrain Model
```bash
cd /home/matt/Documents/drum-tranxn/drum_transcription

# Run optimization with more trials/epochs
uv run python train_with_optuna.py \
  --n-trials 50 \
  --study-name drum-transcription-11class-retrain
```

### Option 3: Debug Training Data
```bash
# Inspect what the model sees during training
cd /home/matt/Documents/drum-tranxn/drum_transcription
python scripts/inspect_data.py --num-samples 10
```

---

## File Locations

**API Code**:
```
/home/matt/Documents/drum-tranxn/drum-transcription-api/
├── drum_transcription_api/
│   ├── main.py (updated for 11-class)
│   ├── transcription.py (updated for 11-class)
│   └── __init__.py
├── test_11class_direct.py (test script)
└── test_audio/
    ├── 32 Soft Pink Glow.mp3 (test file)
    └── raw_predictions.npy (debug output)
```

**Model & Training**:
```
/home/matt/Documents/drum-tranxn/drum_transcription/
├── src/models/crnn.py (11-class model)
├── train_with_optuna.py (training script)
└── optuna_checkpoints/trial_0/
    └── trial_0-epoch=00-val_loss=0.2757.ckpt (checkpoint)
```

---

## Summary

The **API has been successfully updated for 11-class drum structure**, but the **trained model appears to have issues with prediction quality**. The model is detecting all classes at very low confidence levels, suggesting either:

1. Poor model training/convergence
2. Mismatch between training and test audio preprocessing
3. Issues with model architecture pooling

**Recommendation**: Verify the model training was successful before deploying to production. Consider retraining with better hyperparameters or testing with the original E-GMD training data to validate model functionality.
