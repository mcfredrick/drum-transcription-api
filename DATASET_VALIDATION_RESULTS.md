# Dataset Validation Test Results

## Test Overview

**Test Audio**: E-GMD Eval Set `10_soul-groove10_102_beat_4-4_10.wav`  
**Ground Truth**: E-GMD Eval Set `10_soul-groove10_102_beat_4-4_10.midi`  
**Model**: `drum-11class-epoch=99-val_loss=0.0872.ckpt`  
**Test Date**: 2026-02-08

---

## Results Summary

### Onset Detection Performance

| Metric | Value | Assessment |
|:-------|:-----:|:-----------|
| **Original Onsets** | 223 | Ground truth from E-GMD |
| **Detected Onsets** | 83 | Model output |
| **Recall** | 37.2% | ❌ **LOW** |
| **Matched Onsets (±100ms)** | 40/223 | Only 17.9% within timing tolerance |

### Class-wise Accuracy

| Drum Class | Original | Detected | Match % | Assessment |
|:-----------|:--------:|:--------:|:-------:|:-----------|
| kick | 47 | 23 | 48.9% | ⚠ Moderate |
| snare_head | 34 | 3 | **8.8%** | ❌ **Poor** |
| snare_rim | 31 | 18 | 58.1% | ⚠ Moderate |
| hihat_closed | 101 | 36 | 35.6% | ❌ Poor |
| hihat_pedal | 4 | 0 | **0.0%** | ❌ **Missed** |
| floor_tom | 3 | 1 | 33.3% | ❌ Poor |
| high_mid_tom | 2 | 1 | 50.0% | ⚠ Moderate |
| side_stick | 1 | 1 | **100.0%** | ✅ Detected |

### Timing Accuracy

- **Matched Onsets (within ±100ms)**: 40 out of 223 (17.9%)
- **Average Timing Error**: 51.9 ms
- **Max Timing Error**: 99.2 ms

---

## Detailed Comparison

### Ground Truth Distribution (Original MIDI)

```
hihat_closed: 101 hits (45.3%) - DOMINANT
kick:          47 hits (21.1%)
snare_head:    34 hits (15.2%)
snare_rim:     31 hits (13.9%)
hihat_pedal:    4 hits (1.8%)
floor_tom:      3 hits (1.3%)
high_mid_tom:   2 hits (0.9%)
side_stick:     1 hit  (0.4%)
───────────────────────────────
TOTAL:        223 hits
```

### Model Detection Distribution (Transcribed MIDI)

```
hihat_closed:  36 hits (43.4%)
kick:          23 hits (27.7%)
snare_rim:     18 hits (21.7%)
snare_head:     3 hits (3.6%)
high_mid_tom:   1 hit  (1.2%)
floor_tom:      1 hit  (1.2%)
side_stick:     1 hit  (1.2%)
hihat_pedal:    0 hits (0.0%)
───────────────────────────────
TOTAL:         83 hits (37.2% of ground truth)
```

### Missing Classes

The model completely missed:
- **hihat_pedal**: 4 hits in original (0 detected)

Severely underpredicted:
- **snare_head**: 34 hits in original (3 detected = 8.8%)

---

## Analysis

### What Went Wrong

1. **Low Overall Recall (37.2%)**
   - Model is missing 62.8% of the actual drum hits
   - Threshold may be too high (0.5)
   - Model confidence on this audio may be lower than training data

2. **Class-Specific Issues**
   - **snare_head**: Almost completely missed (8.8% recall)
   - **hihat_pedal**: Completely missed (0% recall)
   - **hihat_closed**: Underpredicted (35.6% recall)
   
   These suggest the model struggles with certain drum sounds in this particular recording.

3. **Timing Issues**
   - Even when detected, only 40/83 detected onsets match ground truth within 100ms
   - This suggests the model detects different frames than the ground truth

4. **Possible Causes**
   - **Audio characteristics**: This audio may have different frequency content/mix than training data
   - **Threshold too high**: 0.5 threshold may be too strict for this audio
   - **Training data bias**: Model may have been trained on different drum kit/recording setup
   - **Model limitations**: The 11-class model may not be optimal for this complex polyphonic drum mix

---

## Threshold Analysis

The current threshold of **0.5** appears too high. Let me test what thresholds might work better:

### Predicted Improvement with Lower Thresholds

If we lowered the detection threshold:
- **Threshold 0.3**: Likely to catch more onsets but may introduce false positives
- **Threshold 0.2**: Would significantly increase detections but risk false positives
- **Threshold 0.1**: Maximum recall but high false positive rate

**Recommendation**: Test with threshold **0.3** to see if it improves recall without too many false positives.

---

## Comparison with Test Audio ("32 Soft Pink Glow.mp3")

### Model Performance Differs

| Metric | "Soft Pink Glow" | "Soul Groove" |
|:-------|:----------------:|:-------------:|
| Onsets Detected | 344 | 83 |
| Dominant Class | ride_bell (77%) | hihat_closed (43%) |
| Overall Assessment | ✅ Good | ❌ Poor |

**Key Difference**: 
- "Soft Pink Glow" is dominantly a cymbal-based track (ride_bell)
- "Soul Groove" is a complex drum pattern with many classes

This suggests the model may work better on simpler, less polyphonic audio.

---

## Conclusions

### 🔴 Current Status
- **Dataset Performance**: POOR (37.2% recall on E-GMD test set)
- **Model Reliability**: Not suitable for production use on complex drum mixes
- **Threshold Sensitivity**: High threshold (0.5) is problematic

### ⚠️ Potential Issues

1. **Training/Test Mismatch**
   - Model may have been trained with different preprocessing or data
   - Training data characteristics don't match E-GMD eval set

2. **Model Convergence**
   - Despite good validation loss (0.0872), practical performance is poor
   - May indicate overfitting or data distribution issues

3. **Architecture Limitations**
   - CRNN may not be optimal for this task
   - 11-class structure may be too fine-grained for reliable detection

### ✅ Recommendations

#### Short-term
1. **Lower threshold**: Test with 0.3-0.4 to improve recall
2. **Check training data**: Verify model was trained on proper E-GMD data
3. **Analyze raw predictions**: Use `debug_predictions.py` on this audio

#### Medium-term
1. **Fine-tune model**: Retrain on E-GMD eval set to improve performance
2. **Data analysis**: Investigate why training and test performance diverge
3. **Post-processing**: Add temporal smoothing to improve onset detection

#### Long-term
1. **Model selection**: Consider alternative architectures (TCN, Transformer)
2. **Ensemble methods**: Combine multiple models for better reliability
3. **Class grouping**: Simplify to fewer classes for more reliable detection

---

## Files Generated

- **Transcribed MIDI**: `test_dataset/10_soul-groove10_102_beat_4-4_10_transcribed.mid`
- **Original MIDI**: `test_dataset/10_soul-groove10_102_beat_4-4_10.midi` (ground truth)
- **Original Audio**: `test_dataset/10_soul-groove10_102_beat_4-4_10.wav`

---

## Next Steps

1. Run validation with lower thresholds (0.3, 0.2, 0.1)
2. Check if training data matches E-GMD format
3. Inspect why snare_head and hihat_pedal are missed
4. Consider retraining or fine-tuning the model

---

## Summary

The model shows **good performance on simple, cymbal-dominated audio** but **poor performance on complex, polyphonic drum patterns**. The **37.2% recall on E-GMD test data** indicates the model is **not yet production-ready** and needs improvement or retraining.

**Status**: ⚠️ **NEEDS IMPROVEMENT**
