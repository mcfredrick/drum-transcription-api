"""Onset refinement using audio-based onset detection.

This module provides functionality to refine model predictions by aligning them
to onsets detected directly from the audio signal using librosa's onset detection.
"""

import librosa
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def detect_onsets(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    backtrack: bool = True,
    delta: float = 0.05,
) -> np.ndarray:
    """
    Detect onsets in audio using librosa's spectral flux onset detection.

    Args:
        audio_path: Path to audio file
        sr: Sample rate for loading audio
        hop_length: Hop length for onset detection (default 512 = ~23ms frames)
        backtrack: Use onset backtracking to refine onset times (default True)
        delta: Threshold parameter for peak picking (default 0.05)

    Returns:
        Array of onset times in seconds, sorted chronologically
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr)

        # Detect onsets using spectral flux
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            hop_length=hop_length,
            backtrack=backtrack,
            delta=delta,
            units='frames'
        )

        # Convert to time
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)

        logger.info(f"Detected {len(onset_times)} onsets in audio")
        return onset_times

    except Exception as e:
        logger.error(f"Failed to detect onsets: {e}")
        raise


def refine_onsets_with_detected(
    predictions: List[Tuple[float, str, int]],
    detected_onsets: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[List[Tuple[float, str, int]], Dict]:
    """
    Refine model predictions by snapping to detected audio onsets.

    Args:
        predictions: List of (time, drum_name, velocity) tuples from model
        detected_onsets: Array of onset times from librosa (seconds)
        threshold: Maximum distance (seconds) to snap to onset (default 50ms)

    Returns:
        Tuple of:
            - List of refined (time, drum_name, velocity) tuples
            - Dictionary with refinement statistics
    """
    if len(predictions) == 0:
        return [], {'snapped': 0, 'kept': 0, 'total': 0}

    if len(detected_onsets) == 0:
        logger.warning("No onsets detected in audio, keeping original predictions")
        return predictions, {'snapped': 0, 'kept': len(predictions), 'total': len(predictions)}

    refined = []
    snapped_count = 0
    kept_count = 0

    for pred_time, drum, velocity in predictions:
        # Find nearest detected onset
        distances = np.abs(detected_onsets - pred_time)
        nearest_idx = np.argmin(distances)
        nearest_onset = detected_onsets[nearest_idx]
        distance = distances[nearest_idx]

        if distance <= threshold:
            # Snap to detected onset
            refined.append((nearest_onset, drum, velocity))
            snapped_count += 1
        else:
            # Keep original prediction (no nearby onset detected)
            refined.append((pred_time, drum, velocity))
            kept_count += 1

    # Sort by time
    refined.sort(key=lambda x: x[0])

    stats = {
        'snapped': snapped_count,
        'kept': kept_count,
        'total': len(predictions),
        'snap_rate': snapped_count / len(predictions) if predictions else 0.0,
        'threshold': threshold,
        'detected_onsets': len(detected_onsets),
    }

    logger.info(
        f"Onset refinement: {snapped_count}/{len(predictions)} predictions snapped "
        f"({stats['snap_rate']*100:.1f}%), {kept_count} kept"
    )

    return refined, stats


def refine_predictions(
    audio_path: str,
    predictions: List[Tuple[float, str, int]],
    snap_threshold: float = 0.05,
    onset_delta: float = 0.05,
) -> Tuple[List[Tuple[float, str, int]], Dict]:
    """
    Complete refinement pipeline: detect onsets and refine predictions.

    Args:
        audio_path: Path to audio file used for transcription
        predictions: List of (time, drum_name, velocity) from model
        snap_threshold: Maximum distance to snap predictions to onsets (default 50ms)
        onset_delta: Onset detection sensitivity parameter (default 0.05)

    Returns:
        Tuple of:
            - Refined predictions as list of (time, drum_name, velocity)
            - Statistics dictionary with refinement info
    """
    # Detect onsets
    detected_onsets = detect_onsets(audio_path, delta=onset_delta)

    # Refine predictions
    refined, stats = refine_onsets_with_detected(
        predictions, detected_onsets, threshold=snap_threshold
    )

    return refined, stats
