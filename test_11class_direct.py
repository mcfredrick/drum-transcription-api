#!/usr/bin/env python3
"""Direct test of the 11-class model without running the server."""

import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

from drum_transcription_api.transcription import DrumTranscriber


def test_with_audio():
    """Test the transcription with the audio file."""
    print("=" * 80)
    print("TESTING 11-CLASS MODEL WITH AUDIO FILE")
    print("=" * 80)

    # Audio file
    audio_path = "/home/matt/Documents/drum-tranxn/drum-transcription-api/test_audio/32 Soft Pink Glow.mp3"

    if not Path(audio_path).exists():
        print(f"✗ Audio file not found: {audio_path}")
        return False

    print(f"\n✓ Audio file found: {audio_path}")
    print(f"  Size: {Path(audio_path).stat().st_size / 1024 / 1024:.2f} MB")

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    model_checkpoint = (
        "/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt"
    )

    if not Path(model_checkpoint).exists():
        print(f"✗ Model checkpoint not found: {model_checkpoint}")
        return False

    print(f"Loading model: {model_checkpoint}")

    try:
        transcriber = DrumTranscriber(model_checkpoint)
        print(f"✓ Model loaded successfully")
        print(f"  Classes: {transcriber.model.n_classes}")
        print(f"  Device: {transcriber.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Transcribe
    print("\n" + "=" * 80)
    print("TRANSCRIBING AUDIO")
    print("=" * 80)

    try:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            output_path = tmp.name

        print(f"Input:  {audio_path}")
        print(f"Output: {output_path}")
        print("\nProcessing... (this may take a moment)")

        result = transcriber.transcribe_to_midi(
            audio_path=audio_path,
            output_midi_path=output_path,
            threshold=0.5,
            min_interval=0.05,
            tempo=120,
        )

        # Extract stats from result
        stats = result if isinstance(result, dict) else {}

        print(f"✓ Transcription completed")
        print(f"\n  Duration: {stats.get('duration', 'N/A')} seconds")
        print(f"  Total onsets: {stats.get('num_onsets', 0)}")

        # Parse results
        print("\n" + "=" * 80)
        print("DETECTION RESULTS")
        print("=" * 80)

        import pretty_midi

        midi = pretty_midi.PrettyMIDI(output_path)

        class_counts = {}
        total_notes = 0

        if midi.instruments:
            from drum_transcription_api.transcription import DRUM_MAP

            # Create reverse mapping MIDI note -> drum name
            midi_to_drum = {v: k for k, v in DRUM_MAP.items()}

            for note in midi.instruments[0].notes:
                total_notes += 1
                # Find the drum class name
                if note.pitch in midi_to_drum:
                    drum_name = midi_to_drum[note.pitch]
                    class_counts[drum_name] = class_counts.get(drum_name, 0) + 1

        print(f"\nTotal onsets detected: {total_notes}")

        if total_notes > 0:
            print("\nBreakdown by drum class:")
            sorted_classes = sorted(
                class_counts.items(), key=lambda x: x[1], reverse=True
            )
            for i, (drum_name, count) in enumerate(sorted_classes, 1):
                if count > 0:
                    percentage = count / total_notes * 100
                    print(f"  {i:2d}. {drum_name:20s}: {count:3d} ({percentage:5.1f}%)")

        print("\n" + "=" * 80)
        print("✓ TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"✗ Error during transcription: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_audio()
    sys.exit(0 if success else 1)
