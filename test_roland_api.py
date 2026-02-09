"""Test the updated drum transcription API with Roland model."""

import sys
sys.path.append('/home/matt/Documents/drum-tranxn/drum_transcription')

from drum_transcription_api.transcription import DrumTranscriber
import librosa
import numpy as np
from pathlib import Path

# Test parameters
model_checkpoint = "/mnt/hdd/drum-tranxn/checkpoints_roland/roland-full-training-epoch=24-val_loss=0.0721.ckpt"
audio_path = "/home/matt/Documents/drum-tranxn/drum-transcription-api/test_audio/32 Soft Pink Glow.mp3"
output_midi = "/tmp/test_roland_output.mid"

print("=" * 70)
print("Drum Transcription API - Roland Model Test")
print("=" * 70)

# Initialize transcriber
print(f"\nInitializing transcriber with model: {Path(model_checkpoint).name}")
try:
    transcriber = DrumTranscriber(model_checkpoint, device='cpu')
    print(f"✓ Transcriber initialized")
    print(f"  - Device: {transcriber.device}")
except Exception as e:
    print(f"✗ Failed to initialize: {e}")
    sys.exit(1)

# Run full transcription pipeline (first 30 seconds for speed)
print(f"\nTranscribing audio file...")
print(f"  - Input: {Path(audio_path).name}")
print(f"  - Output: {output_midi}")

try:
    statistics = transcriber.transcribe_to_midi(
        audio_path=audio_path,
        output_midi_path=output_midi,
        threshold=0.5,
        min_interval=0.05,
        tempo=120,
        export_predictions=False
    )
    
    print(f"✓ Transcription successful!")
    print(f"\nStatistics:")
    print(f"  - Total hits detected: {statistics['total_hits']}")
    print(f"  - Audio duration: {statistics['duration']:.2f} seconds")
    print(f"\nPer-drum breakdown:")
    for drum_name, count in statistics['per_drum'].items():
        if count > 0:
            print(f"  - {drum_name}: {count} hits")
    
    # Verify MIDI file was created
    if Path(output_midi).exists():
        file_size = Path(output_midi).stat().st_size
        print(f"\n✓ MIDI file created: {file_size} bytes")
    else:
        print(f"\n✗ MIDI file not found!")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ API TEST PASSED - Roland model is working correctly!")
print("=" * 70)
