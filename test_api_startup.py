#!/usr/bin/env python3
"""Test API startup with hierarchical model."""

import os
import sys

# Test loading the transcriber as the API would
checkpoint_path = "/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=01-val_kick_roc_auc=0.973.ckpt"

# Fallback to flat model if hierarchical not found
if not os.path.exists(checkpoint_path):
    print(f"WARNING: Hierarchical model not found: {checkpoint_path}")
    print("Falling back to flat model")
    checkpoint_path = "/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt"

if not os.path.exists(checkpoint_path):
    print(f"ERROR: No checkpoint found at {checkpoint_path}")
    sys.exit(1)

# Import after path check
sys.path.insert(0, "/home/matt/Documents/drum-tranxn/drum-transcription-api")
from drum_transcription_api.transcription import DrumTranscriber, DRUM_NAMES, DRUM_MAP

print("Testing API startup with hierarchical model...")
print(f"Checkpoint: {checkpoint_path}")

try:
    # Force CPU to avoid CUDA OOM during training
    transcriber = DrumTranscriber(checkpoint_path, device="cpu", model_type="auto")
    print(f"✓ Model loaded successfully")
    print(f"  Model type: {transcriber.model_type}")
    print(f"  Device: {transcriber.device}")
    print(f"  Number of drum classes: {len(DRUM_NAMES)}")
    print(f"\nDrum classes:")
    for i, name in enumerate(DRUM_NAMES):
        midi_note = DRUM_MAP.get(name, "?")
        print(f"  {i:2d}. {name:15s} -> MIDI {midi_note}")

    print("\n✓ API would start successfully with this configuration")

except Exception as e:
    print(f"✗ Failed to initialize transcriber: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
