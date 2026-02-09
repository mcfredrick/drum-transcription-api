# 11-Class Model Checkpoint Comparison

## Problem Discovered

When testing the API with the audio file `32 Soft Pink Glow.mp3`, we initially used the wrong checkpoint:

```
/mnt/hdd/drum-tranxn/optuna_checkpoints/trial_0/trial_0-epoch=00-val_loss=0.2757.ckpt
```

This checkpoint produced **0 onsets** despite being trained for the 11-class structure.

## Root Cause Investigation

Found **multiple checkpoints** in `/mnt/hdd/drum-tranxn/checkpoints/`:

| Checkpoint | Val Loss | Epoch | Notes |
|:-----------|:--------:|:-----:|:------|
| `drum-11class-epoch=99-val_loss=0.0872.ckpt` | **0.0872** | 99 | ✅ **BEST** |
| `drum-11class-epoch=97-val_loss=0.0881.ckpt` | 0.0881 | 97 | Very good |
| `drum-11class-epoch=95-val_loss=0.0881.ckpt` | 0.0881 | 95 | Very good |
| `drum-11class-full-training-epoch=00-val_loss=0.2711.ckpt` | 0.2711 | 0 | Poor |
| Trial_0 optuna checkpoint | 0.2757 | 0 | ❌ Worst |

## Why the Best Model?

### Training Path
The best model came from training experiment: `crnn-11class-standard-kit/version_1`

Located at: `/mnt/hdd/drum-tranxn/logs/crnn-11class-standard-kit/version_1/`

### Convergence Analysis
```
Trial_0 optuna: val_loss=0.2757 (Epoch 0 only - didn't train properly)
Full training:  val_loss=0.2711 (Epoch 0 - early checkpoint)
Epoch 95+:      val_loss=0.0881 (Well-trained)
Epoch 99:       val_loss=0.0872 (Best - converged)
```

The best checkpoint represents the model after 99 epochs of proper training, achieving optimal convergence.

## Performance Comparison

### Test Audio: `32 Soft Pink Glow.mp3`

#### Using Poor Checkpoint (val_loss=0.2757)
```
Max Prediction Across All Classes: 0.270
Onsets > 0.5 threshold: 0
Frames with significant detection: 0
Result: ❌ UNUSABLE
```

**Raw Predictions** (max per class):
- kick: 0.130
- snare_head: 0.098  
- snare_rim: 0.243
- hihat_closed: 0.270 (highest)
- All others: < 0.210

#### Using Best Checkpoint (val_loss=0.0872)
```
Max Prediction Across All Classes: 0.884 (ride_bell)
Onsets > 0.5 threshold: 90
Total Detected Onsets: 344
Result: ✅ EXCELLENT
```

**Raw Predictions** (max per class):
- kick: 0.597
- snare_rim: 0.774
- hihat_pedal: 0.561
- high_mid_tom: 0.818
- ride_bell: **0.884** (strongest)
- 8 out of 11 classes > 0.5 in at least one frame

### Improvement Metrics

| Metric | Poor Model | Best Model | Improvement |
|:-------|:----------:|:----------:|:------------|
| Val Loss | 0.2757 | 0.0872 | **3.16x better** |
| Max Prediction | 0.270 | 0.884 | **3.27x higher** |
| Onsets Detected | 0 | 344 | **∞ improvement** |
| Usable | ❌ No | ✅ Yes | **Critical** |

## What Happened?

### Optuna Trial_0 Issues
The optuna checkpoint was saved at **epoch 0** with a validation loss of 0.2757:
- Model didn't train properly
- Only 1 epoch of training
- Likely learning rate or data issues
- Model was underfitting significantly

### Successful Training Run
The `crnn-11class-standard-kit/version_1` experiment ran for **99 epochs**:
- Proper convergence
- Well-trained on E-GMD data
- Learning rate schedule worked correctly
- Model learned discriminative features

## Conclusion

**The critical difference was using the right checkpoint!**

- ✅ Poor checkpoint: val_loss=0.2757 → **0 detections**
- ✅ Best checkpoint: val_loss=0.0872 → **344 detections**

This demonstrates the importance of:
1. **Proper checkpoint selection** - using metrics (val_loss) to choose best model
2. **Complete training** - letting models converge over multiple epochs
3. **Validation during training** - monitoring metrics to catch training issues

## Lesson Learned

When the model performs poorly:
1. ✅ Check if you're using the right checkpoint
2. ✅ Compare validation loss of available checkpoints
3. ✅ Verify the checkpoint was trained for multiple epochs
4. ✅ Test with debug predictions before drawing conclusions

In this case, the model wasn't broken - we just needed to use the correct, fully-trained checkpoint!

## Files Updated

All references to the optuna checkpoint have been updated:

- ✅ `/drum-transcription-api/drum_transcription_api/main.py`
- ✅ `/drum-transcription-api/debug_predictions.py`
- ✅ `/drum-transcription-api/test_11class_direct.py`

All now point to: `/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt`
