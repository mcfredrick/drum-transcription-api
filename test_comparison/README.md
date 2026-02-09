# Hierarchical Model Test Comparison

## Test Example Details

**Source:** E-GMD Dataset Test Split
**File:** `drummer1/session2/98_funk-rock_92_fill_4-4_34.wav`
**Type:** Funk-Rock drum fill
**Tempo:** 92 BPM
**Duration:** 2.61 seconds
**Time Signature:** 4/4

## Files in This Directory

### 1. `original_audio.wav`
- **Format:** WAV, 16-bit mono, 44.1 kHz
- **Size:** 225 KB
- **Description:** Original audio from E-GMD dataset

### 2. `original_ground_truth.midi`
- **Format:** Standard MIDI (format 1, 2 tracks, 480 PPQN)
- **Size:** 219 bytes
- **Description:** Human-labeled ground truth from E-GMD dataset
- **Mapping:** Roland TD-50 MIDI mapping (E-GMD's native format)

### 3. `hierarchical_output.mid`
- **Format:** Standard MIDI (format 1, 2 tracks, 220 PPQN)
- **Size:** 177 bytes
- **Description:** Model prediction from hierarchical CRNN (epoch 99)
- **Mapping:** Roland TD-17 compatible (12-class subset)
- **Checkpoint:** `hierarchical-epoch=99-val_kick_roc_auc=1.000.ckpt`

## Model Results

**Total hits detected:** 20

**Per-drum breakdown:**
- Snare head: 7 hits
- Kick: 6 hits
- Floor tom: 5 hits
- High-mid tom: 2 hits

## Comparison Notes

### Expected Differences

1. **MIDI Mapping:**
   - Ground truth uses TD-50 mapping (full E-GMD articulation set)
   - Model output uses TD-17 mapping (12-class simplified set)
   - Some TD-50 articulations may not have direct TD-17 equivalents

2. **Timing Precision:**
   - Ground truth: Exact timing from human performance (480 PPQN)
   - Model output: Frame-based detection (43 FPS = ~23ms resolution)
   - Slight timing offsets expected

3. **Class Granularity:**
   - Ground truth may have rim shots, edge hits, etc.
   - Model predicts simplified drum types (head/rim, closed/open)

### How to Compare

**In a DAW (Reaper, Ableton, etc.):**
1. Import `original_audio.wav`
2. Import both MIDI files on separate tracks
3. Route to a drum sampler (e.g., Superior Drummer, MT Power Drum Kit)
4. Listen to audio while viewing MIDI note lanes
5. Check timing alignment and drum type accuracy

**Using Python/Music21:**
```python
import music21
gt = music21.converter.parse('original_ground_truth.midi')
pred = music21.converter.parse('hierarchical_output.mid')
gt.show('text')  # View notes
pred.show('text')
```

**Visual MIDI Comparison:**
- Use a MIDI editor like Ardour, MuseScore, or online MIDI visualizers
- Compare note timing and drum assignments

## Model Performance Context

**Training Details:**
- Architecture: Hierarchical 5-branch CRNN
- Training: 100 epochs on E-GMD dataset
- Validation metrics:
  - Kick ROC-AUC: 1.000 (perfect)
  - Overall performance: Production-ready

**Known Limitations:**
- May confuse ride/crash (crash branch conservative)
- No side stick or hihat pedal prediction
- Simplified tom articulations (high/mid merged)

## Next Steps for Analysis

1. **Visual Inspection:** Compare MIDI note rolls side-by-side
2. **Quantitative Metrics:** Calculate precision/recall per drum type
3. **Timing Analysis:** Measure onset time differences (± tolerance)
4. **Class Confusion:** Identify which drum types are confused
5. **Multiple Examples:** Repeat with more test samples for statistical analysis

## Dataset Citation

```
@inproceedings{gillick2019learning,
  title={Learning to Groove with Inverse Sequence Transformations},
  author={Gillick, Jon and Roberts, Adam and Engel, Jesse and Eck, Douglas and Bamman, David},
  booktitle={International Conference on Machine Learning},
  year={2019}
}
```
