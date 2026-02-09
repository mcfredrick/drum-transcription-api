# Implementation Plan: Onset Timing Evaluation & Improvement

## Implementation Status

✅ **Onset Refinement (Strategy 1) - IMPLEMENTED** (2026-02-09)
- Created `drum_transcription_api/onset_refinement.py` module
- Added optional `refine_with_onsets` parameter to API
- Uses librosa onset detection to refine model predictions
- Available for baseline vs refined comparison

⏳ **Evaluation Framework (Phase 1) - PENDING**
- Still needs implementation for quantitative metrics

---

## Context

**Current State:** The hierarchical drum transcription model detects drum hits but we have no quantitative measure of:
1. **Onset timing accuracy** - How close are predicted onsets to ground truth?
2. **Per-class performance** - Which drums are detected well vs poorly?
3. **Precision/Recall/F1** - What's the detection rate and false positive rate?

**Problem:** Without metrics, we can't:
- Establish a baseline for model performance
- Identify which drums need improvement
- Measure the impact of timing improvement strategies
- Compare different models or approaches

**Goal:** Create a comprehensive evaluation framework to measure onset timing accuracy against ground truth MIDI, then explore methods to improve timing (e.g., beat detection, post-processing).

---

## Part 1: Evaluation Framework

### Objectives

1. **Load ground truth** from E-GMD MIDI files
2. **Load predictions** from model transcriptions
3. **Match onsets** within tolerance window (±50ms typical for drums)
4. **Calculate metrics** per drum class:
   - Precision (% of predictions that are correct)
   - Recall (% of ground truth hits detected)
   - F1 score (harmonic mean)
   - Mean absolute timing error for matched onsets
5. **Generate reports** with visualizations and statistics

### Success Metrics

**Good performance:**
- Kick/Snare F1 > 0.85
- Timing error < 30ms mean
- Other drums F1 > 0.70

**Baseline to establish:**
- Current performance across all 12 classes
- Per-song variation
- Timing error distribution

---

## Architecture

### Evaluation Pipeline

```
Ground Truth MIDI → Parse → Onset List
                              ↓
                         [Alignment]
                              ↑
Model Predictions → Parse → Onset List

         ↓

[Calculate Metrics]
    - Precision/Recall/F1
    - Timing Error
    - Confusion Matrix

         ↓

[Generate Report]
    - Per-class metrics
    - Timing histograms
    - Error analysis
```

### Key Components

1. **MIDI Parser** - Extract onsets from MIDI files
2. **Onset Matcher** - Match predictions to ground truth within tolerance
3. **Metric Calculator** - Compute P/R/F1, timing errors
4. **Report Generator** - Create visualizations and tables
5. **Batch Evaluator** - Run on full test set

---

## Implementation Steps

### Phase 1: MIDI Parsing and Onset Extraction

**File:** `evaluation/onset_parser.py` (NEW)

#### Step 1.1: Create MIDI onset parser

```python
"""Parse MIDI files to extract drum onsets."""

import pretty_midi
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

# TD-17 to 12-class mapping
TD17_TO_12CLASS = {
    36: 'kick',           # Kick
    38: 'snare_head',     # Snare Head
    40: 'snare_rim',      # Snare Rim
    37: 'side_stick',     # Side Stick (cross stick)
    44: 'hihat_pedal',    # Hi-Hat Pedal
    42: 'hihat_closed',   # Hi-Hat Closed
    46: 'hihat_open',     # Hi-Hat Open
    43: 'floor_tom',      # Floor Tom (or Tom 3)
    48: 'high_mid_tom',   # High-Mid Tom (Tom 1)
    45: 'high_mid_tom',   # Tom 2 → merge with high_mid
    51: 'ride',           # Ride
    53: 'ride_bell',      # Ride Bell
    49: 'crash',          # Crash
    55: 'crash',          # Crash edge → merge with crash
    57: 'crash',          # Crash 2 → merge with crash
}

class OnsetParser:
    """Parse onsets from MIDI and prediction files."""

    @staticmethod
    def parse_midi(midi_path: str, note_mapping: Dict[int, str] = None) -> List[Tuple[float, str]]:
        """
        Parse MIDI file and extract drum onsets.

        Args:
            midi_path: Path to MIDI file
            note_mapping: Optional custom MIDI note to drum name mapping

        Returns:
            List of (time, drum_name) tuples, sorted by time
        """
        if note_mapping is None:
            note_mapping = TD17_TO_12CLASS

        midi = pretty_midi.PrettyMIDI(midi_path)
        onsets = []

        # Get drum track (usually the only instrument, or first instrument)
        for instrument in midi.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    drum_name = note_mapping.get(note.pitch)
                    if drum_name:
                        onsets.append((note.start, drum_name))

        # Sort by time
        onsets.sort(key=lambda x: x[0])
        return onsets

    @staticmethod
    def parse_predictions(predictions: List[Tuple[float, str, int]]) -> List[Tuple[float, str]]:
        """
        Parse model predictions (from transcription output).

        Args:
            predictions: List of (time, drum_name, velocity) tuples

        Returns:
            List of (time, drum_name) tuples, sorted by time
        """
        # Remove velocity, keep time and drum_name
        onsets = [(time, drum) for time, drum, _ in predictions]
        onsets.sort(key=lambda x: x[0])
        return onsets
```

---

### Phase 2: Onset Matching and Alignment

**File:** `evaluation/onset_matcher.py` (NEW)

#### Step 2.1: Create onset matching algorithm

```python
"""Match predicted onsets to ground truth within tolerance window."""

import numpy as np
from typing import List, Tuple, Dict, Set

class OnsetMatcher:
    """Match predicted onsets to ground truth onsets."""

    def __init__(self, tolerance: float = 0.05):
        """
        Initialize matcher.

        Args:
            tolerance: Time tolerance for matching (seconds). Default: 50ms
        """
        self.tolerance = tolerance

    def match_onsets(
        self,
        ground_truth: List[Tuple[float, str]],
        predictions: List[Tuple[float, str]]
    ) -> Dict:
        """
        Match predictions to ground truth within tolerance window.

        Args:
            ground_truth: List of (time, drum_name) from MIDI
            predictions: List of (time, drum_name) from model

        Returns:
            Dictionary with matching results:
            {
                'matches': [(gt_time, pred_time, drum_name, error), ...],
                'false_positives': [(pred_time, drum_name), ...],
                'false_negatives': [(gt_time, drum_name), ...],
                'per_class': {drum_name: {...}, ...}
            }
        """
        # Organize by drum class
        gt_by_class = self._group_by_class(ground_truth)
        pred_by_class = self._group_by_class(predictions)

        all_classes = set(gt_by_class.keys()) | set(pred_by_class.keys())

        matches = []
        false_positives = []
        false_negatives = []
        per_class_results = {}

        for drum_class in all_classes:
            gt_times = sorted(gt_by_class.get(drum_class, []))
            pred_times = sorted(pred_by_class.get(drum_class, []))

            class_matches, class_fp, class_fn = self._match_class(
                gt_times, pred_times, drum_class
            )

            matches.extend(class_matches)
            false_positives.extend(class_fp)
            false_negatives.extend(class_fn)

            # Calculate per-class metrics
            tp = len(class_matches)
            fp = len(class_fp)
            fn = len(class_fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            timing_errors = [error for _, _, _, error in class_matches]
            mean_error = np.mean(timing_errors) if timing_errors else 0.0
            std_error = np.std(timing_errors) if timing_errors else 0.0

            per_class_results[drum_class] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'mean_timing_error': mean_error,
                'std_timing_error': std_error,
            }

        return {
            'matches': matches,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'per_class': per_class_results,
            'tolerance': self.tolerance,
        }

    def _group_by_class(self, onsets: List[Tuple[float, str]]) -> Dict[str, List[float]]:
        """Group onsets by drum class."""
        grouped = {}
        for time, drum in onsets:
            if drum not in grouped:
                grouped[drum] = []
            grouped[drum].append(time)
        return grouped

    def _match_class(
        self,
        gt_times: List[float],
        pred_times: List[float],
        drum_class: str
    ) -> Tuple[List, List, List]:
        """
        Match predictions to ground truth for a single drum class.

        Uses greedy matching: each GT onset matched to closest prediction within tolerance.

        Returns:
            (matches, false_positives, false_negatives)
        """
        matches = []
        matched_preds = set()
        matched_gts = set()

        # For each ground truth onset, find closest prediction within tolerance
        for gt_idx, gt_time in enumerate(gt_times):
            best_pred_idx = None
            best_error = float('inf')

            for pred_idx, pred_time in enumerate(pred_times):
                if pred_idx in matched_preds:
                    continue

                error = abs(pred_time - gt_time)
                if error <= self.tolerance and error < best_error:
                    best_error = error
                    best_pred_idx = pred_idx

            if best_pred_idx is not None:
                matches.append((
                    gt_time,
                    pred_times[best_pred_idx],
                    drum_class,
                    best_error
                ))
                matched_preds.add(best_pred_idx)
                matched_gts.add(gt_idx)

        # False positives: predictions not matched to any ground truth
        false_positives = [
            (pred_times[i], drum_class)
            for i in range(len(pred_times))
            if i not in matched_preds
        ]

        # False negatives: ground truth not matched to any prediction
        false_negatives = [
            (gt_times[i], drum_class)
            for i in range(len(gt_times))
            if i not in matched_gts
        ]

        return matches, false_positives, false_negatives
```

---

### Phase 3: Metric Calculation and Reporting

**File:** `evaluation/metrics_calculator.py` (NEW)

#### Step 3.1: Calculate aggregate metrics

```python
"""Calculate evaluation metrics from matching results."""

import numpy as np
from typing import Dict, List
import pandas as pd

class MetricsCalculator:
    """Calculate and aggregate evaluation metrics."""

    @staticmethod
    def calculate_summary(match_results: Dict) -> Dict:
        """
        Calculate summary metrics across all classes.

        Args:
            match_results: Output from OnsetMatcher.match_onsets()

        Returns:
            Summary metrics dictionary
        """
        per_class = match_results['per_class']

        # Aggregate metrics
        total_tp = sum(m['true_positives'] for m in per_class.values())
        total_fp = sum(m['false_positives'] for m in per_class.values())
        total_fn = sum(m['false_negatives'] for m in per_class.values())

        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
            if (overall_precision + overall_recall) > 0 else 0.0

        # Timing errors (only for matched onsets)
        all_errors = [error for _, _, _, error in match_results['matches']]
        mean_timing_error = np.mean(all_errors) if all_errors else 0.0
        median_timing_error = np.median(all_errors) if all_errors else 0.0
        std_timing_error = np.std(all_errors) if all_errors else 0.0

        return {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1_score': overall_f1,
                'true_positives': total_tp,
                'false_positives': total_fp,
                'false_negatives': total_fn,
            },
            'timing': {
                'mean_error': mean_timing_error,
                'median_error': median_timing_error,
                'std_error': std_timing_error,
                'tolerance': match_results['tolerance'],
            },
            'per_class': per_class,
        }

    @staticmethod
    def to_dataframe(summary: Dict) -> pd.DataFrame:
        """Convert summary metrics to pandas DataFrame for easy viewing."""
        rows = []
        for drum_class, metrics in summary['per_class'].items():
            rows.append({
                'Drum Class': drum_class,
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1 Score': f"{metrics['f1_score']:.3f}",
                'TP': metrics['true_positives'],
                'FP': metrics['false_positives'],
                'FN': metrics['false_negatives'],
                'Mean Error (ms)': f"{metrics['mean_timing_error']*1000:.1f}",
            })

        # Add overall row
        overall = summary['overall']
        rows.append({
            'Drum Class': 'OVERALL',
            'Precision': f"{overall['precision']:.3f}",
            'Recall': f"{overall['recall']:.3f}",
            'F1 Score': f"{overall['f1_score']:.3f}",
            'TP': overall['true_positives'],
            'FP': overall['false_positives'],
            'FN': overall['false_negatives'],
            'Mean Error (ms)': f"{summary['timing']['mean_error']*1000:.1f}",
        })

        return pd.DataFrame(rows)
```

---

### Phase 4: Batch Evaluation Script

**File:** `evaluation/evaluate_test_set.py` (NEW)

#### Step 4.1: Create batch evaluation script

```python
#!/usr/bin/env python3
"""Evaluate model on E-GMD test set."""

import sys
from pathlib import Path
import json
from tqdm import tqdm

sys.path.insert(0, '/home/matt/Documents/drum-tranxn/drum-transcription-api')

from drum_transcription_api.transcription import DrumTranscriber
from evaluation.onset_parser import OnsetParser
from evaluation.onset_matcher import OnsetMatcher
from evaluation.metrics_calculator import MetricsCalculator

# Configuration
EGMD_ROOT = Path('/mnt/hdd/drum-tranxn/e-gmd-v1.0.0')
TEST_SPLIT_FILE = '/mnt/hdd/drum-tranxn/processed_data_hierarchical/splits/test_split.txt'
CHECKPOINT = '/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=99-val_kick_roc_auc=1.000.ckpt'
OUTPUT_DIR = Path('evaluation_results')
TOLERANCE = 0.05  # 50ms tolerance
MAX_FILES = 100  # Limit for faster testing (set None for full evaluation)

def evaluate_file(transcriber, audio_path, midi_path, matcher):
    """Evaluate a single test file."""
    try:
        # Run transcription
        stats = transcriber.transcribe_to_midi(
            audio_path=str(audio_path),
            output_midi_path='/tmp/temp_transcription.mid',
            threshold=0.5,
            min_interval=0.05,
            tempo=120,
            separate_stems=False  # E-GMD is already isolated drums
        )

        # Parse ground truth
        gt_onsets = OnsetParser.parse_midi(str(midi_path))

        # Parse predictions
        pred_onsets = OnsetParser.parse_predictions(stats['onsets'])

        # Match and calculate metrics
        results = matcher.match_onsets(gt_onsets, pred_onsets)

        return results, None

    except Exception as e:
        return None, str(e)


def main():
    """Run batch evaluation on test set."""
    print('=' * 70)
    print('ONSET TIMING EVALUATION - E-GMD TEST SET')
    print('=' * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load test split
    with open(TEST_SPLIT_FILE) as f:
        test_files = [line.strip() for line in f if line.strip()]

    if MAX_FILES:
        test_files = test_files[:MAX_FILES]
        print(f'Evaluating subset: {MAX_FILES} files')
    else:
        print(f'Evaluating full test set: {len(test_files)} files')
    print()

    # Initialize components
    print('Loading model...')
    transcriber = DrumTranscriber(CHECKPOINT, device='cpu', model_type='auto')
    print('✓ Model loaded')
    print()

    matcher = OnsetMatcher(tolerance=TOLERANCE)

    # Evaluate each file
    all_results = []
    errors = []

    print(f'Evaluating test files (tolerance={TOLERANCE*1000:.0f}ms)...')
    for file_path in tqdm(test_files):
        audio_path = EGMD_ROOT / file_path
        midi_path = EGMD_ROOT / file_path.replace('.wav', '.midi')

        # Skip if files don't exist
        if not audio_path.exists() or not midi_path.exists():
            continue

        results, error = evaluate_file(transcriber, audio_path, midi_path, matcher)

        if results:
            all_results.append({
                'file': file_path,
                'results': results
            })
        else:
            errors.append({'file': file_path, 'error': error})

    print(f'\n✓ Evaluated {len(all_results)} files successfully')
    if errors:
        print(f'✗ {len(errors)} files failed')
    print()

    # Aggregate metrics across all files
    print('Calculating aggregate metrics...')
    aggregate_metrics = aggregate_results(all_results)

    # Generate report
    print('Generating report...')
    generate_report(aggregate_metrics, all_results, OUTPUT_DIR)

    print()
    print('=' * 70)
    print(f'Results saved to: {OUTPUT_DIR}')
    print('=' * 70)


def aggregate_results(all_results):
    """Aggregate metrics across all test files."""
    # Combine all matches, FPs, FNs
    all_matches = []
    all_fps = []
    all_fns = []

    per_class_aggregated = {}

    for item in all_results:
        results = item['results']
        all_matches.extend(results['matches'])
        all_fps.extend(results['false_positives'])
        all_fns.extend(results['false_negatives'])

        # Aggregate per-class counts
        for drum_class, metrics in results['per_class'].items():
            if drum_class not in per_class_aggregated:
                per_class_aggregated[drum_class] = {
                    'tp': 0, 'fp': 0, 'fn': 0,
                    'timing_errors': []
                }

            per_class_aggregated[drum_class]['tp'] += metrics['true_positives']
            per_class_aggregated[drum_class]['fp'] += metrics['false_positives']
            per_class_aggregated[drum_class]['fn'] += metrics['false_negatives']

    # Calculate aggregate per-class metrics
    for drum_class in per_class_aggregated:
        tp = per_class_aggregated[drum_class]['tp']
        fp = per_class_aggregated[drum_class]['fp']
        fn = per_class_aggregated[drum_class]['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_aggregated[drum_class].update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
        })

    # Extract timing errors for matched onsets
    timing_errors = [error for _, _, _, error in all_matches]

    return {
        'per_class': per_class_aggregated,
        'timing_errors': timing_errors,
        'total_files': len(all_results),
    }


def generate_report(aggregate_metrics, all_results, output_dir):
    """Generate evaluation report."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create summary table
    summary_data = []
    for drum_class, metrics in sorted(aggregate_metrics['per_class'].items()):
        summary_data.append({
            'Drum Class': drum_class,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'TP': metrics['tp'],
            'FP': metrics['fp'],
            'FN': metrics['fn'],
        })

    # Save as JSON
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(aggregate_metrics, f, indent=2, default=str)

    # Save as CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(output_dir / 'evaluation_summary.csv', index=False)

    # Print summary
    print()
    print('=' * 70)
    print('EVALUATION SUMMARY')
    print('=' * 70)
    print(df.to_string(index=False))
    print()

    # Timing error histogram
    timing_errors = aggregate_metrics['timing_errors']
    if timing_errors:
        plt.figure(figsize=(10, 6))
        plt.hist([e * 1000 for e in timing_errors], bins=50, edgecolor='black')
        plt.xlabel('Timing Error (ms)')
        plt.ylabel('Count')
        plt.title('Onset Timing Error Distribution')
        plt.axvline(x=0, color='red', linestyle='--', label='Perfect alignment')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'timing_error_histogram.png', dpi=150)
        plt.close()

        print(f'Mean timing error: {np.mean(timing_errors)*1000:.2f}ms')
        print(f'Median timing error: {np.median(timing_errors)*1000:.2f}ms')
        print(f'Std timing error: {np.std(timing_errors)*1000:.2f}ms')

    # F1 scores bar chart
    plt.figure(figsize=(12, 6))
    drums = [m['Drum Class'] for m in summary_data]
    f1_scores = [m['F1 Score'] for m in summary_data]
    plt.bar(drums, f1_scores, edgecolor='black')
    plt.xlabel('Drum Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.axhline(y=0.85, color='green', linestyle='--', label='Target (0.85)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'f1_scores_per_class.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    import numpy as np
    main()
```

---

## Part 2: Timing Improvement Strategies

### Baseline Establishment

Before implementing improvements, establish baseline:

1. Run evaluation on test set (100-200 examples)
2. Identify problematic drum classes (low F1 or high timing error)
3. Analyze error patterns (early vs late, consistent vs random)

### Improvement Strategy 1: Onset-Aware Post-Processing

**Hypothesis:** Model predictions have timing errors. Refine by aligning to audio onsets detected via signal processing.

**Key insight:** Unlike beat detection (which finds musical pulse), onset detection finds individual transients/hits in the audio - perfect for drums.

#### Implementation Approach

1. **Onset detection:** Use `librosa.onset.onset_detect` to find all transients in audio
2. **Selective snapping:** Snap model predictions to nearest detected onset within threshold
3. **Conservative approach:** Only refine if onset is nearby (e.g., 50-100ms)

**File:** `improvement/onset_refinement.py`

```python
import librosa
import numpy as np

def detect_onsets(audio_path, sr=22050, hop_length=512, **kwargs):
    """
    Detect onsets in audio using librosa.

    Args:
        audio_path: Path to audio file
        sr: Sample rate for loading audio
        hop_length: Hop length for onset detection
        **kwargs: Additional arguments for librosa.onset.onset_detect
            - backtrack: Use onset backtracking (default True)
            - energy: Use energy-based onset detection
            - units: 'time' or 'frames' (default 'time')

    Returns:
        Array of onset times in seconds
    """
    y, sr = librosa.load(audio_path, sr=sr)

    # Detect onsets using spectral flux by default
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        backtrack=kwargs.get('backtrack', True),
        units='frames'
    )

    # Convert to time
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_times


def refine_onsets_with_detected(predictions, detected_onsets, threshold=0.05):
    """
    Refine model predictions by snapping to detected audio onsets.

    Args:
        predictions: List of (time, drum_name, velocity) tuples from model
        detected_onsets: Array of onset times from librosa
        threshold: Maximum distance (seconds) to snap to onset (default 50ms)

    Returns:
        List of refined (time, drum_name, velocity) tuples
    """
    refined = []

    for pred_time, drum, velocity in predictions:
        # Find nearest detected onset
        distances = np.abs(detected_onsets - pred_time)
        nearest_idx = np.argmin(distances)
        nearest_onset = detected_onsets[nearest_idx]
        distance = distances[nearest_idx]

        if distance <= threshold:
            # Snap to detected onset
            refined.append((nearest_onset, drum, velocity))
        else:
            # Keep original prediction (no nearby onset detected)
            refined.append((pred_time, drum, velocity))

    return refined
```

### Improvement Strategy 2: Onset Detection Refinement

**Hypothesis:** Frame-level predictions are noisy. Refine by:
1. Peak detection with adaptive thresholds
2. Gaussian smoothing before peak detection
3. Dynamic time warping for alignment

**File:** `improvement/onset_refinement.py`

```python
from scipy.signal import find_peaks, gaussian
from scipy.ndimage import gaussian_filter1d

def refine_onsets(predictions, min_distance_frames=2):
    """
    Refine onset detection using advanced peak picking.

    Args:
        predictions: (time, n_classes) array of probabilities
        min_distance_frames: Minimum distance between peaks

    Returns:
        Refined onsets
    """
    refined_onsets = []

    for class_idx in range(predictions.shape[1]):
        class_preds = predictions[:, class_idx]

        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(class_preds, sigma=1.0)

        # Adaptive threshold (mean + std)
        threshold = smoothed.mean() + smoothed.std()

        # Find peaks with prominence
        peaks, properties = find_peaks(
            smoothed,
            height=threshold,
            distance=min_distance_frames,
            prominence=0.1
        )

        # Refine peak location to exact frame with maximum
        for peak in peaks:
            # Look in window around detected peak
            window_start = max(0, peak - 2)
            window_end = min(len(class_preds), peak + 3)
            window = class_preds[window_start:window_end]
            refined_peak = window_start + np.argmax(window)

            refined_onsets.append((refined_peak, class_idx))

    return refined_onsets
```

### Improvement Strategy 3: Train with Timing Loss

**Hypothesis:** Add timing-aware loss to training to encourage precise onset prediction.

**Approach:**
1. Current loss: Frame-level BCE/CE
2. Add onset timing loss: Penalize distance between predicted peak and ground truth onset
3. Multi-task: Combine frame-level + timing loss

**Training modification:** (for future training runs)

```python
def onset_timing_loss(predictions, ground_truth_onsets, frame_rate):
    """
    Calculate onset timing loss.

    Penalizes predictions that peak far from ground truth onset times.
    """
    loss = 0
    for gt_time, drum_class in ground_truth_onsets:
        gt_frame = int(gt_time * frame_rate)

        # Get predictions around ground truth frame
        window = predictions[max(0, gt_frame-5):gt_frame+5, drum_class]

        # Peak should be at center (gt_frame)
        peak_frame = np.argmax(window)
        timing_error = abs(peak_frame - 5)  # 5 is center

        loss += timing_error

    return loss
```

---

## Evaluation and Iteration

### Testing Improvements

For each improvement strategy:

1. **Implement** method
2. **Run evaluation** on test set
3. **Compare metrics** to baseline
4. **Analyze** which drums improved/degraded
5. **Iterate** on parameters (thresholds, etc.)

### Success Criteria

**Strategy is successful if:**
- Mean timing error reduces by >20%
- F1 scores improve or stay same (no precision loss)
- Works across different musical styles

---

## Timeline

### Phase 1: Evaluation Framework (Week 1)
- Day 1-2: Implement parsers and matchers
- Day 3: Implement metrics calculator
- Day 4-5: Batch evaluation script and testing

### Phase 2: Baseline Establishment (Week 2)
- Day 1-2: Run evaluation on full test set
- Day 3-4: Analyze results, identify issues
- Day 5: Document baseline and problem areas

### Phase 3: Improvement Implementation (Week 3-4)
- Week 3: Implement Strategy 1 (beat-aware) and test
- Week 4: Implement Strategy 2 (refinement) and test
- Compare all strategies

### Phase 4: Integration (Week 5)
- Integrate best strategy into API
- Update training pipeline if needed
- Final evaluation and documentation

---

## Expected Outcomes

### Baseline (Current Model)

**Estimated performance:**
- Kick/Snare F1: 0.80-0.85
- Other drums F1: 0.60-0.75
- Mean timing error: 30-50ms
- Issues: Cymbals (ride/crash confusion), timing jitter

### After Improvements

**Target performance:**
- Kick/Snare F1: >0.85
- Other drums F1: >0.75
- Mean timing error: <30ms
- Reduced false positives on cymbals

---

## Deliverables

1. **Evaluation framework** - Reusable evaluation code
2. **Baseline report** - Current model performance
3. **Improvement implementations** - 2-3 tested strategies
4. **Comparison report** - Which strategy works best
5. **Integration** - Best strategy added to API
6. **Documentation** - Usage guide and results

---

## References

- **Metrics:** See MIR evaluation standards (MIREX)
- **Beat detection:** librosa.beat.beat_track, madmom
- **Onset refinement:** scipy.signal.find_peaks
- **Dataset:** E-GMD test set (623 files)
- **Current codebase:** `/home/matt/Documents/drum-tranxn/drum-transcription-api`
