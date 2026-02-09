#!/usr/bin/env python3
"""Test API against E-GMD dataset ground truth."""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from drum_transcription_api.transcription import DrumTranscriber, DRUM_MAP
import pretty_midi


def compare_midi_files(original_path, transcribed_path):
    """Compare original and transcribed MIDI files."""
    print("\n" + "=" * 80)
    print("COMPARING MIDI FILES")
    print("=" * 80)

    # Load both files
    original = pretty_midi.PrettyMIDI(original_path)
    transcribed = pretty_midi.PrettyMIDI(transcribed_path)

    print(f"\nOriginal:    {Path(original_path).name}")
    print(f"Transcribed: {Path(transcribed_path).name}")

    # Extract notes from both
    midi_to_drum = {v: k for k, v in DRUM_MAP.items()}

    original_notes = []
    transcribed_notes = []

    if original.instruments:
        for note in original.instruments[0].notes:
            if note.pitch in midi_to_drum:
                original_notes.append(
                    (note.start, note.pitch, midi_to_drum[note.pitch])
                )

    if transcribed.instruments:
        for note in transcribed.instruments[0].notes:
            if note.pitch in midi_to_drum:
                transcribed_notes.append(
                    (note.start, note.pitch, midi_to_drum[note.pitch])
                )

    # Sort by time
    original_notes.sort(key=lambda x: x[0])
    transcribed_notes.sort(key=lambda x: x[0])

    print(f"\nOriginal Notes: {len(original_notes)}")
    print(f"Transcribed Notes: {len(transcribed_notes)}")

    # Analyze accuracy
    print("\n" + "-" * 80)
    print("ORIGINAL MIDI (Ground Truth):")
    print("-" * 80)

    # Count classes in original
    original_counts = {}
    for _, pitch, drum_name in original_notes:
        original_counts[drum_name] = original_counts.get(drum_name, 0) + 1

    print(f"\nDrum Class Distribution:")
    for drum_name in sorted(original_counts.keys()):
        count = original_counts[drum_name]
        print(f"  {drum_name:20s}: {count:3d}")

    print("\nFirst 20 Onsets:")
    for i, (time, pitch, drum_name) in enumerate(original_notes[:20], 1):
        print(f"  {i:2d}. {time:7.3f}s | {drum_name}")

    print("\n" + "-" * 80)
    print("TRANSCRIBED MIDI (Model Output):")
    print("-" * 80)

    # Count classes in transcribed
    transcribed_counts = {}
    for _, pitch, drum_name in transcribed_notes:
        transcribed_counts[drum_name] = transcribed_counts.get(drum_name, 0) + 1

    print(f"\nDrum Class Distribution:")
    for drum_name in sorted(transcribed_counts.keys()):
        count = transcribed_counts[drum_name]
        print(f"  {drum_name:20s}: {count:3d}")

    print("\nFirst 20 Onsets:")
    for i, (time, pitch, drum_name) in enumerate(transcribed_notes[:20], 1):
        print(f"  {i:2d}. {time:7.3f}s | {drum_name}")

    # Accuracy metrics
    print("\n" + "=" * 80)
    print("ACCURACY METRICS")
    print("=" * 80)

    # Count total onsets
    print(f"\nOnset Detection:")
    print(f"  Original:    {len(original_notes):3d} onsets")
    print(f"  Transcribed: {len(transcribed_notes):3d} onsets")

    if len(original_notes) > 0:
        recall = len(transcribed_notes) / len(original_notes) * 100
        print(f"  Recall:      {recall:.1f}%")

    # Class-wise comparison
    print(f"\nClass-wise Comparison:")
    print(f"{'Class':<20} {'Original':<10} {'Transcribed':<12} {'Match %':<10}")
    print("-" * 52)

    all_classes = set(original_counts.keys()) | set(transcribed_counts.keys())

    for drum_name in sorted(all_classes):
        orig_count = original_counts.get(drum_name, 0)
        trans_count = transcribed_counts.get(drum_name, 0)

        if orig_count > 0:
            match_pct = min(orig_count, trans_count) / orig_count * 100
        else:
            match_pct = 0

        print(f"{drum_name:<20} {orig_count:<10} {trans_count:<12} {match_pct:>6.1f}%")

    # Temporal accuracy (onset timing)
    print(f"\nTiming Accuracy:")

    # Find matching onsets (within 0.1 second tolerance)
    matching_onsets = 0
    time_errors = []

    for trans_time, trans_pitch, trans_drum in transcribed_notes:
        for orig_time, orig_pitch, orig_drum in original_notes:
            time_diff = abs(trans_time - orig_time)
            if time_diff < 0.1:  # 100ms tolerance
                matching_onsets += 1
                time_errors.append(time_diff)
                break

    print(f"  Matching onsets (±100ms): {matching_onsets}/{len(original_notes)}")

    if time_errors:
        avg_error = sum(time_errors) / len(time_errors)
        max_error = max(time_errors)
        print(f"  Average timing error: {avg_error * 1000:.1f} ms")
        print(f"  Max timing error: {max_error * 1000:.1f} ms")

    # Overall assessment
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    if len(transcribed_notes) > 0 and len(original_notes) > 0:
        onset_recall = len(transcribed_notes) / len(original_notes) * 100

        if onset_recall > 80:
            print("✅ EXCELLENT - Model correctly detected most onsets")
        elif onset_recall > 60:
            print("✓ GOOD - Model detected majority of onsets")
        elif onset_recall > 40:
            print("⚠ FAIR - Model detected roughly half of onsets")
        else:
            print("❌ POOR - Model detection rate is low")

    print("\n" + "=" * 80)


def main():
    """Main test function."""
    print("=" * 80)
    print("E-GMD DATASET VALIDATION TEST")
    print("=" * 80)

    # Paths
    audio_file = "test_dataset/10_soul-groove10_102_beat_4-4_10.wav"
    original_midi = "test_dataset/10_soul-groove10_102_beat_4-4_10.midi"
    model_checkpoint = (
        "/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt"
    )

    # Verify files exist
    for file_path in [audio_file, original_midi]:
        if not Path(file_path).exists():
            print(f"✗ File not found: {file_path}")
            return False

    print(f"\n✓ Audio file: {audio_file}")
    print(f"✓ Original MIDI: {original_midi}")
    print(f"✓ Model: {model_checkpoint}")

    # Load model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)

    try:
        transcriber = DrumTranscriber(model_checkpoint)
        print(f"✓ Model loaded")
        print(f"  Classes: {transcriber.model.n_classes}")
        print(f"  Device: {transcriber.device}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

    # Transcribe
    print("\n" + "=" * 80)
    print("TRANSCRIBING AUDIO")
    print("=" * 80)

    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            transcribed_path = tmp.name

        print(f"Processing: {audio_file}")
        print("(This may take a moment)")

        result = transcriber.transcribe_to_midi(
            audio_path=audio_file,
            output_midi_path=transcribed_path,
            threshold=0.5,
            min_interval=0.05,
            tempo=120,
        )

        print(f"✓ Transcription completed")
        print(f"  Output: {transcribed_path}")

        # Compare results
        compare_midi_files(original_midi, transcribed_path)

        # Save transcribed MIDI
        output_midi = "test_dataset/10_soul-groove10_102_beat_4-4_10_transcribed.mid"
        import shutil

        shutil.copy(transcribed_path, output_midi)
        print(f"\n✓ Transcribed MIDI saved to: {output_midi}")

        return True

    except Exception as e:
        print(f"✗ Error during transcription: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()

    print("\n" + "=" * 80)
    if success:
        print("✅ VALIDATION TEST COMPLETED")
    else:
        print("❌ VALIDATION TEST FAILED")
    print("=" * 80 + "\n")

    sys.exit(0 if success else 1)
