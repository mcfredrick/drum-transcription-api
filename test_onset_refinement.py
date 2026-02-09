#!/usr/bin/env python3
"""Test script for onset refinement feature.

This script demonstrates the difference between baseline and onset-refined predictions.
"""

import sys
from pathlib import Path

# Add API to path
sys.path.insert(0, '/home/matt/Documents/drum-tranxn/drum-transcription-api')

from drum_transcription_api.transcription import DrumTranscriber
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_onset_refinement(audio_path: str):
    """
    Test onset refinement by transcribing the same file twice:
    1. Baseline (without refinement)
    2. With onset refinement

    Args:
        audio_path: Path to test audio file
    """
    # Model checkpoint
    checkpoint = '/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=99-val_kick_roc_auc=1.000.ckpt'

    # Initialize transcriber
    logger.info("Loading model...")
    transcriber = DrumTranscriber(checkpoint, device='cpu', model_type='auto')
    logger.info(f"Model loaded: {transcriber.model_type}")
    print()

    # Test parameters
    output_dir = Path('/tmp/onset_refinement_test')
    output_dir.mkdir(exist_ok=True)

    # Run baseline transcription
    logger.info("=" * 70)
    logger.info("BASELINE TRANSCRIPTION (no refinement)")
    logger.info("=" * 70)
    baseline_midi = output_dir / 'baseline.mid'
    baseline_stats = transcriber.transcribe_to_midi(
        audio_path=audio_path,
        output_midi_path=str(baseline_midi),
        threshold=0.5,
        min_interval=0.05,
        tempo=120,
        separate_stems=False,  # Assuming input is already isolated drums
        refine_with_onsets=False,  # Baseline
    )

    print()
    logger.info("Baseline Results:")
    logger.info(f"  Total hits: {baseline_stats['total_hits']}")
    logger.info(f"  Duration: {baseline_stats['duration']:.2f}s")
    logger.info(f"  Per-drum counts:")
    for drum, count in baseline_stats['per_drum'].items():
        if count > 0:
            logger.info(f"    {drum}: {count}")
    print()

    # Run refined transcription
    logger.info("=" * 70)
    logger.info("ONSET-REFINED TRANSCRIPTION")
    logger.info("=" * 70)
    refined_midi = output_dir / 'refined.mid'
    refined_stats = transcriber.transcribe_to_midi(
        audio_path=audio_path,
        output_midi_path=str(refined_midi),
        threshold=0.5,
        min_interval=0.05,
        tempo=120,
        separate_stems=False,
        refine_with_onsets=True,  # Enable refinement
        refinement_threshold=0.05,  # 50ms snap threshold
        onset_delta=0.05,  # Default onset detection sensitivity
    )

    print()
    logger.info("Refined Results:")
    logger.info(f"  Total hits: {refined_stats['total_hits']}")
    logger.info(f"  Duration: {refined_stats['duration']:.2f}s")
    logger.info(f"  Per-drum counts:")
    for drum, count in refined_stats['per_drum'].items():
        if count > 0:
            logger.info(f"    {drum}: {count}")
    print()

    # Show refinement statistics
    if 'refinement' in refined_stats:
        ref = refined_stats['refinement']
        logger.info("Refinement Statistics:")
        logger.info(f"  Detected onsets in audio: {ref['detected_onsets']}")
        logger.info(f"  Predictions snapped: {ref['snapped']}/{ref['total']} ({ref['snap_rate']*100:.1f}%)")
        logger.info(f"  Predictions kept: {ref['kept']}/{ref['total']}")
        logger.info(f"  Snap threshold: {ref['threshold']*1000:.0f}ms")
    print()

    # Compare results
    logger.info("=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    logger.info(f"Total hits delta: {refined_stats['total_hits'] - baseline_stats['total_hits']:+d}")
    logger.info(f"Baseline MIDI: {baseline_midi}")
    logger.info(f"Refined MIDI: {refined_midi}")
    print()
    logger.info("Listen to both MIDI files to compare timing accuracy!")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test onset refinement feature')
    parser.add_argument('audio_path', help='Path to test audio file (isolated drums recommended)')
    args = parser.parse_args()

    test_onset_refinement(args.audio_path)
