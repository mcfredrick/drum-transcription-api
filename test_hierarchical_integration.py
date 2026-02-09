#!/usr/bin/env python3
"""Test script to verify hierarchical model integration."""

import sys
import os
import numpy as np

# Add the API module to path
sys.path.insert(0, "/home/matt/Documents/drum-tranxn/drum-transcription-api")

from drum_transcription_api.transcription import DrumTranscriber
from drum_transcription_api.hierarchical_utils import convert_hierarchical_to_flat, apply_activation_to_hierarchical

def test_conversion_logic():
    """Test the hierarchical to flat conversion with synthetic data."""
    print("\n=== Testing Hierarchical to Flat Conversion ===")

    # Create synthetic hierarchical predictions (already probabilities)
    batch, time = 1, 100

    hierarchical_preds = {
        'kick': np.random.rand(batch, time, 1).astype(np.float32),
        'snare': np.random.rand(batch, time, 3).astype(np.float32),
        'tom': {
            'primary': np.random.rand(batch, time, 2).astype(np.float32),
            'variation': np.random.rand(batch, time, 3).astype(np.float32),
        },
        'cymbal': {
            'primary': np.random.rand(batch, time, 3).astype(np.float32),
            'hihat_variation': np.random.rand(batch, time, 2).astype(np.float32),
            'ride_variation': np.random.rand(batch, time, 2).astype(np.float32),
        },
        'crash': np.random.rand(batch, time, 1).astype(np.float32),
    }

    # Normalize multi-class predictions to sum to 1 (simulate softmax)
    hierarchical_preds['snare'] = hierarchical_preds['snare'] / hierarchical_preds['snare'].sum(axis=-1, keepdims=True)
    hierarchical_preds['tom']['primary'] = hierarchical_preds['tom']['primary'] / hierarchical_preds['tom']['primary'].sum(axis=-1, keepdims=True)
    hierarchical_preds['tom']['variation'] = hierarchical_preds['tom']['variation'] / hierarchical_preds['tom']['variation'].sum(axis=-1, keepdims=True)
    hierarchical_preds['cymbal']['primary'] = hierarchical_preds['cymbal']['primary'] / hierarchical_preds['cymbal']['primary'].sum(axis=-1, keepdims=True)
    hierarchical_preds['cymbal']['hihat_variation'] = hierarchical_preds['cymbal']['hihat_variation'] / hierarchical_preds['cymbal']['hihat_variation'].sum(axis=-1, keepdims=True)
    hierarchical_preds['cymbal']['ride_variation'] = hierarchical_preds['cymbal']['ride_variation'] / hierarchical_preds['cymbal']['ride_variation'].sum(axis=-1, keepdims=True)

    # Convert to flat
    flat_preds = convert_hierarchical_to_flat(hierarchical_preds)

    print(f"✓ Conversion successful")
    print(f"  Input: hierarchical dictionary")
    print(f"  Output shape: {flat_preds.shape}")
    print(f"  Expected shape: ({batch}, {time}, 12)")
    assert flat_preds.shape == (batch, time, 12), f"Shape mismatch: {flat_preds.shape}"
    print(f"  Value range: [{flat_preds.min():.3f}, {flat_preds.max():.3f}]")
    assert flat_preds.min() >= 0 and flat_preds.max() <= 1, "Values out of [0, 1] range"

    # Check each class
    class_names = ['kick', 'snare_head', 'snare_rim', 'side_stick', 'hihat_pedal',
                   'hihat_closed', 'hihat_open', 'floor_tom', 'high_mid_tom',
                   'ride', 'ride_bell', 'crash']

    print(f"\n  Per-class statistics:")
    for i, name in enumerate(class_names):
        active_frames = (flat_preds[0, :, i] > 0).sum()
        print(f"    {i:2d}. {name:15s}: {active_frames:3d} active frames")

    print("\n✓ All conversion tests passed!")
    return True


def test_model_loading():
    """Test loading the hierarchical model."""
    print("\n=== Testing Model Loading ===")

    checkpoint_path = "/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=01-val_kick_roc_auc=0.973.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        transcriber = DrumTranscriber(checkpoint_path, model_type="auto")

        print(f"✓ Model loaded successfully")
        print(f"  Model type: {transcriber.model_type}")
        print(f"  Device: {transcriber.device}")

        assert transcriber.model_type == "hierarchical", f"Expected hierarchical, got {transcriber.model_type}"
        print("\n✓ Model loading test passed!")
        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_forward_pass():
    """Test running inference with the hierarchical model."""
    print("\n=== Testing Model Forward Pass ===")

    checkpoint_path = "/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=01-val_kick_roc_auc=0.973.ckpt"

    try:
        transcriber = DrumTranscriber(checkpoint_path, model_type="auto")

        # Create synthetic audio spectrogram (128, time)
        time_frames = 200
        spec = np.random.randn(128, time_frames).astype(np.float32)

        # Save to temporary file
        import tempfile
        import soundfile as sf

        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        # Create dummy audio (1 second at 22050 Hz)
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        sf.write(temp_audio.name, audio, 22050)

        print(f"Running inference on synthetic audio...")
        predictions = transcriber.transcribe_drums(temp_audio.name)

        print(f"✓ Forward pass successful")
        print(f"  Output shape: {predictions.shape}")
        print(f"  Expected format: (time, 12)")
        assert predictions.shape[1] == 12, f"Expected 12 classes, got {predictions.shape[1]}"
        print(f"  Value range: [{predictions.min():.3f}, {predictions.max():.3f}]")

        # Check per-class statistics
        class_names = ['kick', 'snare_head', 'snare_rim', 'side_stick', 'hihat_pedal',
                       'hihat_closed', 'hihat_open', 'floor_tom', 'high_mid_tom',
                       'ride', 'ride_bell', 'crash']

        print(f"\n  Per-class prediction statistics:")
        for i, name in enumerate(class_names):
            mean_prob = predictions[:, i].mean()
            max_prob = predictions[:, i].max()
            above_threshold = (predictions[:, i] > 0.5).sum()
            print(f"    {i:2d}. {name:15s}: mean={mean_prob:.3f}, max={max_prob:.3f}, >0.5={above_threshold:3d}")

        # Cleanup
        os.unlink(temp_audio.name)

        print("\n✓ Forward pass test passed!")
        return True

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("HIERARCHICAL MODEL INTEGRATION TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: Conversion logic
    results.append(("Conversion Logic", test_conversion_logic()))

    # Test 2: Model loading
    results.append(("Model Loading", test_model_loading()))

    # Test 3: Forward pass
    results.append(("Forward Pass", test_model_forward_pass()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
