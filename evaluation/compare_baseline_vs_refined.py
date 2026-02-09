#!/usr/bin/env python3
"""
Compare baseline vs onset-refined transcriptions against ground truth.

This script evaluates both approaches on the E-GMD test set to determine
if onset refinement improves timing accuracy.
"""

import sys
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '/home/matt/Documents/drum-tranxn/drum-transcription-api')

from drum_transcription_api.transcription import DrumTranscriber
from evaluation.onset_parser import OnsetParser
from evaluation.onset_matcher import OnsetMatcher
from evaluation.metrics_calculator import MetricsCalculator
import logging

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

# Configuration
EGMD_ROOT = Path('/mnt/hdd/drum-tranxn/e-gmd-v1.0.0')
TEST_SPLIT_FILE = '/mnt/hdd/drum-tranxn/processed_data_hierarchical/splits/test_split.txt'
CHECKPOINT = '/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=99-val_kick_roc_auc=1.000.ckpt'
OUTPUT_DIR = Path('evaluation_results')
TOLERANCE = 0.05  # 50ms tolerance for matching
MAX_FILES = 100  # Evaluate on subset for speed (set None for full test set)


def evaluate_file(transcriber, audio_path, midi_path, matcher, use_refinement=False):
    """
    Evaluate a single test file.

    Args:
        transcriber: DrumTranscriber instance
        audio_path: Path to audio file
        midi_path: Path to ground truth MIDI
        matcher: OnsetMatcher instance
        use_refinement: Whether to use onset refinement

    Returns:
        (results, error) tuple
    """
    try:
        # Run transcription
        stats = transcriber.transcribe_to_midi(
            audio_path=str(audio_path),
            output_midi_path='/tmp/temp_transcription.mid',
            threshold=0.5,
            min_interval=0.05,
            tempo=120,
            separate_stems=False,  # E-GMD is already isolated drums
            refine_with_onsets=use_refinement,
            refinement_threshold=0.05,
            onset_delta=0.05,
        )

        # Parse ground truth
        gt_onsets = OnsetParser.parse_midi(str(midi_path))

        # Parse predictions
        pred_onsets = OnsetParser.parse_predictions(stats['onsets'])

        # Match and calculate metrics
        results = matcher.match_onsets(gt_onsets, pred_onsets)

        # Add refinement stats if used
        if use_refinement and 'refinement' in stats:
            results['refinement_stats'] = stats['refinement']

        return results, None

    except Exception as e:
        return None, str(e)


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

    # Overall metrics
    total_tp = sum(m['tp'] for m in per_class_aggregated.values())
    total_fp = sum(m['fp'] for m in per_class_aggregated.values())
    total_fn = sum(m['fn'] for m in per_class_aggregated.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0.0

    return {
        'per_class': per_class_aggregated,
        'timing_errors': timing_errors,
        'total_files': len(all_results),
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
        }
    }


def main():
    """Run comparison evaluation."""
    print('=' * 80)
    print('BASELINE vs ONSET-REFINED COMPARISON - E-GMD TEST SET')
    print('=' * 80)
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
    print(f'✓ Model loaded ({transcriber.model_type})')
    print()

    matcher = OnsetMatcher(tolerance=TOLERANCE)

    # Evaluate baseline
    print('=' * 80)
    print('EVALUATING BASELINE (no onset refinement)...')
    print('=' * 80)
    baseline_results = []
    errors = []

    for file_path in tqdm(test_files, desc='Baseline'):
        audio_path = EGMD_ROOT / file_path
        midi_path = EGMD_ROOT / file_path.replace('.wav', '.midi')

        if not audio_path.exists() or not midi_path.exists():
            continue

        results, error = evaluate_file(transcriber, audio_path, midi_path, matcher, use_refinement=False)

        if results:
            baseline_results.append({'file': file_path, 'results': results})
        else:
            errors.append({'file': file_path, 'error': error})

    print(f'✓ Baseline: {len(baseline_results)} files evaluated')
    if errors:
        print(f'✗ {len(errors)} files failed')
    print()

    # Evaluate refined
    print('=' * 80)
    print('EVALUATING ONSET-REFINED...')
    print('=' * 80)
    refined_results = []
    errors = []

    for file_path in tqdm(test_files, desc='Refined'):
        audio_path = EGMD_ROOT / file_path
        midi_path = EGMD_ROOT / file_path.replace('.wav', '.midi')

        if not audio_path.exists() or not midi_path.exists():
            continue

        results, error = evaluate_file(transcriber, audio_path, midi_path, matcher, use_refinement=True)

        if results:
            refined_results.append({'file': file_path, 'results': results})
        else:
            errors.append({'file': file_path, 'error': error})

    print(f'✓ Refined: {len(refined_results)} files evaluated')
    if errors:
        print(f'✗ {len(errors)} files failed')
    print()

    # Aggregate results
    print('Calculating aggregate metrics...')
    baseline_metrics = aggregate_results(baseline_results)
    refined_metrics = aggregate_results(refined_results)

    # Generate comparison report
    print()
    print('=' * 80)
    print('COMPARISON RESULTS')
    print('=' * 80)
    print()

    # Overall metrics
    print('OVERALL METRICS:')
    print('-' * 80)
    print(f"{'Metric':<20} {'Baseline':<15} {'Refined':<15} {'Improvement':<15}")
    print('-' * 80)

    baseline_timing = baseline_metrics['timing_errors']
    refined_timing = refined_metrics['timing_errors']

    metrics_to_compare = [
        ('Precision', baseline_metrics['overall']['precision'], refined_metrics['overall']['precision'], True),
        ('Recall', baseline_metrics['overall']['recall'], refined_metrics['overall']['recall'], True),
        ('F1 Score', baseline_metrics['overall']['f1_score'], refined_metrics['overall']['f1_score'], True),
        ('Mean Timing (ms)', np.mean(baseline_timing)*1000 if baseline_timing else 0,
         np.mean(refined_timing)*1000 if refined_timing else 0, False),
        ('Median Timing (ms)', np.median(baseline_timing)*1000 if baseline_timing else 0,
         np.median(refined_timing)*1000 if refined_timing else 0, False),
        ('Std Timing (ms)', np.std(baseline_timing)*1000 if baseline_timing else 0,
         np.std(refined_timing)*1000 if refined_timing else 0, False),
    ]

    for metric_name, baseline_val, refined_val, higher_better in metrics_to_compare:
        if 'Timing' in metric_name:
            # For timing, lower is better
            improvement = baseline_val - refined_val
            pct_change = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            indicator = '✓' if improvement > 0 else '✗'
        else:
            # For P/R/F1, higher is better
            improvement = refined_val - baseline_val
            pct_change = (improvement / baseline_val * 100) if baseline_val > 0 else 0
            indicator = '✓' if improvement > 0 else '✗'

        print(f"{metric_name:<20} {baseline_val:<15.3f} {refined_val:<15.3f} {indicator} {pct_change:+.1f}%")

    print()
    print(f"Files evaluated: {baseline_metrics['total_files']}")
    print(f"Matching tolerance: {TOLERANCE*1000:.0f}ms")
    print()

    # Per-class comparison
    print('PER-CLASS F1 SCORES:')
    print('-' * 80)
    print(f"{'Drum Class':<20} {'Baseline F1':<15} {'Refined F1':<15} {'Change':<15}")
    print('-' * 80)

    all_drums = set(baseline_metrics['per_class'].keys()) | set(refined_metrics['per_class'].keys())
    for drum in sorted(all_drums):
        baseline_f1 = baseline_metrics['per_class'].get(drum, {}).get('f1_score', 0)
        refined_f1 = refined_metrics['per_class'].get(drum, {}).get('f1_score', 0)
        change = refined_f1 - baseline_f1
        indicator = '✓' if change > 0 else ('✗' if change < 0 else '=')

        print(f"{drum:<20} {baseline_f1:<15.3f} {refined_f1:<15.3f} {indicator} {change:+.3f}")

    print()

    # Save detailed results
    output_file = OUTPUT_DIR / 'baseline_vs_refined_comparison.json'
    with open(output_file, 'w') as f:
        json.dump({
            'baseline': {k: v for k, v in baseline_metrics.items() if k != 'timing_errors'},
            'refined': {k: v for k, v in refined_metrics.items() if k != 'timing_errors'},
            'config': {
                'tolerance': TOLERANCE,
                'max_files': MAX_FILES,
                'test_files_evaluated': len(baseline_results),
            }
        }, f, indent=2)

    print(f'Detailed results saved to: {output_file}')
    print()
    print('=' * 80)


if __name__ == '__main__':
    main()
