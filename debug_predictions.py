#!/usr/bin/env python3
"""Debug script to examine model predictions over the entire audio file."""

import sys
import numpy as np
import json
from pathlib import Path

# Add the API path
sys.path.append(".")
from drum_transcription_api.transcription import DrumTranscriber


def main():
    audio_file = "test_audio/32 Soft Pink Glow.mp3"
    model_checkpoint = (
        "/mnt/hdd/drum-tranxn/checkpoints/drum-11class-epoch=99-val_loss=0.0872.ckpt"
    )

    print(f"Loading model from {model_checkpoint}")
    transcriber = DrumTranscriber(model_checkpoint)

    print(f"Processing audio file: {audio_file}")

    # Get the raw spectrogram
    print("Extracting spectrogram...")
    spec = transcriber.extract_log_mel_spectrogram(audio_file)
    print(f"Spectrogram shape: {spec.shape}")

    # Calculate actual audio duration from spectrogram
    frame_rate = 22050 / 512  # ~43 FPS
    spec_duration = spec.shape[1] / frame_rate
    print(f"Spectrogram duration: {spec_duration:.2f} seconds")

    # Get raw model predictions
    print("Running model inference...")
    predictions = transcriber.transcribe_drums(audio_file)
    print(f"Raw predictions shape: {predictions.shape}")

    # Calculate duration from predictions
    pred_duration = predictions.shape[0] / frame_rate
    print(f"Predictions duration: {pred_duration:.2f} seconds")

    # Save raw predictions to file
    output_dir = Path("test_audio")
    predictions_file = output_dir / "raw_predictions.npy"
    np.save(predictions_file, predictions)
    print(f"Raw predictions saved to: {predictions_file}")

    # Save predictions as JSON for easier inspection
    drum_names = [
        "kick",
        "snare_head",
        "snare_rim",
        "side_stick",
        "hihat_pedal",
        "hihat_closed",
        "hihat_open",
        "floor_tom",
        "high_mid_tom",
        "ride",
        "ride_bell",
    ]

    # Convert to more readable format
    predictions_data = {
        "shape": predictions.shape,
        "frame_rate": frame_rate,
        "duration_seconds": pred_duration,
        "drum_names": drum_names,
        "predictions_sample": [],
    }

    # Save first 100 frames as sample
    sample_size = min(100, predictions.shape[0])
    for t in range(sample_size):
        time_sec = t / frame_rate
        frame_data = {"time": time_sec, "frame": t, "predictions": {}}
        for i, drum in enumerate(drum_names):
            frame_data["predictions"][drum] = float(predictions[t, i])
        predictions_data["predictions_sample"].append(frame_data)

    # Save as JSON
    json_file = output_dir / "raw_predictions.json"
    with open(json_file, "w") as f:
        json.dump(predictions_data, f, indent=2)
    print(f"Sample predictions saved to: {json_file}")

    # Print some statistics
    print("\n=== PREDICTION STATISTICS ===")
    for i, drum in enumerate(drum_names):
        class_preds = predictions[:, i]
        print(
            f"{drum:8s}: min={class_preds.min():.3f}, max={class_preds.max():.3f}, mean={class_preds.mean():.3f}"
        )
        print(
            f"         Frames > 0.5: {(class_preds > 0.5).sum()}, > 0.7: {(class_preds > 0.7).sum()}, > 0.9: {(class_preds > 0.9).sum()}"
        )

    print(f"\nTotal frames: {predictions.shape[0]}")
    print(f"Duration: {pred_duration:.2f} seconds")
    print(f"Frames per second: {frame_rate:.1f}")


if __name__ == "__main__":
    main()
