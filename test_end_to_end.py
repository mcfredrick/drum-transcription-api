#!/usr/bin/env python3
"""End-to-end test with real audio file."""

import sys
import os
import tempfile

sys.path.insert(0, "/home/matt/Documents/drum-tranxn/drum-transcription-api")

from drum_transcription_api.transcription import DrumTranscriber

def test_end_to_end():
    """Test complete transcription pipeline with real audio."""
    print("=" * 60)
    print("END-TO-END HIERARCHICAL MODEL TEST")
    print("=" * 60)

    # Test audio file
    audio_file = "/home/matt/Documents/drum-tranxn/drum-transcription-api/test_audio/32 Soft Pink Glow.mp3"

    if not os.path.exists(audio_file):
        print(f"✗ Test audio not found: {audio_file}")
        return False

    # Hierarchical checkpoint
    checkpoint_path = "/mnt/hdd/drum-tranxn/checkpoints/hierarchical/hierarchical-epoch=01-val_kick_roc_auc=0.973.ckpt"

    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return False

    print(f"\nTest Configuration:")
    print(f"  Audio: {os.path.basename(audio_file)}")
    print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"  Device: CPU (avoiding CUDA OOM)")

    try:
        # Initialize transcriber
        print(f"\n1. Loading model...")
        transcriber = DrumTranscriber(checkpoint_path, device="cpu", model_type="auto")
        print(f"   ✓ Model loaded: {transcriber.model_type}")

        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            output_midi = tmp.name

        # Run full pipeline
        print(f"\n2. Running transcription pipeline...")
        print(f"   - Extracting mel spectrogram...")
        print(f"   - Running model inference...")
        print(f"   - Detecting onsets...")
        print(f"   - Creating MIDI file...")

        stats = transcriber.transcribe_to_midi(
            audio_path=audio_file,
            output_midi_path=output_midi,
            threshold=0.5,
            min_interval=0.05,
            tempo=120,
            export_predictions=False
        )

        print(f"   ✓ Transcription completed")

        # Display statistics
        print(f"\n3. Results:")
        print(f"   Total hits: {stats['total_hits']}")
        print(f"   Duration: {stats['duration']:.2f} seconds")
        print(f"   MIDI file: {output_midi}")
        print(f"   File size: {os.path.getsize(output_midi)} bytes")

        print(f"\n   Per-drum statistics:")
        for drum, count in stats['per_drum'].items():
            if count > 0:
                print(f"     {drum:15s}: {count:3d} hits")

        # Verify MIDI file was created
        if os.path.exists(output_midi) and os.path.getsize(output_midi) > 0:
            print(f"\n✓ MIDI file created successfully")
        else:
            print(f"\n✗ MIDI file creation failed")
            return False

        # Cleanup
        os.unlink(output_midi)

        print(f"\n" + "=" * 60)
        print(f"✓ END-TO-END TEST PASSED")
        print(f"=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
