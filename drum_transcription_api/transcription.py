"""Drum transcription pipeline using the trained CRNN model."""

import os
import tempfile
import torch
import librosa
import numpy as np
import pretty_midi
from scipy.signal import find_peaks
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
import soundfile as sf

logger = logging.getLogger(__name__)

# 12-class standard drum kit MIDI drum mapping
DRUM_MAP = {
    "kick": 36,  # 0: Kick
    "snare_head": 38,  # 1: Snare Head
    "snare_rim": 40,  # 2: Snare Rim
    "side_stick": 37,  # 3: Side Stick
    "hihat_pedal": 44,  # 4: Hi-Hat Pedal
    "hihat_closed": 42,  # 5: Hi-Hat Closed
    "hihat_open": 46,  # 6: Hi-Hat Open
    "floor_tom": 43,  # 7: Floor Tom
    "high_mid_tom": 48,  # 8: High-Mid Tom
    "ride": 51,  # 9: Ride
    "ride_bell": 53,  # 10: Ride Bell
    "crash": 49,  # 11: Crash
}

# 12-class drum class names (in order, corresponding to model output indices 0-11)
DRUM_NAMES = [
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
    "crash",
]


class DrumTranscriber:
    """Drum transcription pipeline using trained CRNN model."""

    def __init__(self, model_checkpoint: str, device: str = "auto", model_type: str = "auto"):
        """
        Initialize the transcriber.

        Args:
            model_checkpoint: Path to the trained model checkpoint
            device: 'auto', 'cuda', or 'cpu'
            model_type: 'auto', 'flat', or 'hierarchical'
        """
        self.model_checkpoint = model_checkpoint
        self.device = self._setup_device(device)
        self.model = None
        self.model_type = model_type
        if model_type == "auto":
            self.model_type = self._detect_model_type()
        self._load_model()

    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _detect_model_type(self) -> str:
        """
        Detect whether checkpoint is for flat or hierarchical model.

        Returns:
            'flat' or 'hierarchical'
        """
        try:
            checkpoint = torch.load(self.model_checkpoint, map_location='cpu')

            # Check for hierarchical indicators in hyperparameters
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                if 'branch_weights' in hparams:
                    logger.info("Detected hierarchical model (branch_weights in hyperparameters)")
                    return "hierarchical"

            # Check for branch-specific layers in state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                hierarchical_keys = [
                    'kick_branch.',
                    'snare_branch.',
                    'tom_branch.',
                    'cymbal_branch.',
                    'crash_branch.'
                ]
                for key in state_dict.keys():
                    if any(h_key in key for h_key in hierarchical_keys):
                        logger.info(f"Detected hierarchical model (branch layers in state_dict)")
                        return "hierarchical"

            logger.info("Detected flat model (no hierarchical indicators)")
            return "flat"

        except Exception as e:
            logger.warning(f"Failed to detect model type: {e}. Defaulting to flat.")
            return "flat"

    def _load_model(self):
        """Load the trained model based on detected type."""
        try:
            # Import here to avoid dependency issues if model isn't available
            import sys

            sys.path.append("/home/matt/Documents/drum-tranxn/drum_transcription")

            logger.info(f"Loading {self.model_type} model from {self.model_checkpoint}")

            # Map location for loading checkpoint
            map_location = self.device

            if self.model_type == 'hierarchical':
                from src.models.hierarchical_crnn import HierarchicalDrumCRNN
                self.model = HierarchicalDrumCRNN.load_from_checkpoint(
                    self.model_checkpoint,
                    map_location=map_location
                )
            else:
                from src.models.crnn import DrumTranscriptionCRNN
                self.model = DrumTranscriptionCRNN.load_from_checkpoint(
                    self.model_checkpoint,
                    map_location=map_location
                )

            self.model.eval()
            self.model.to(self.device)
            logger.info(f"{self.model_type.capitalize()} model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _convert_hierarchical_to_flat(self, hierarchical_logits: Dict) -> np.ndarray:
        """
        Convert hierarchical model output to flat 12-class format.

        Args:
            hierarchical_logits: Raw hierarchical model output (dictionary of logits)

        Returns:
            Flat predictions of shape (time, 12) with probabilities in [0, 1]
        """
        from .hierarchical_utils import apply_activation_to_hierarchical, convert_hierarchical_to_flat

        # Apply sigmoid/softmax to get probabilities
        hierarchical_probs = apply_activation_to_hierarchical(hierarchical_logits, self.device)

        # Convert to flat format (batch, time, 12)
        flat_preds = convert_hierarchical_to_flat(hierarchical_probs)

        return flat_preds

    def _separate_drums(self, audio_path: str) -> str:
        """
        Separate drums from full mix using Demucs.

        Args:
            audio_path: Path to full mix audio file

        Returns:
            Path to separated drums audio file (in temp directory)

        Raises:
            Exception: If separation fails
        """
        try:
            import torch
            from demucs.apply import apply_model
            from demucs.pretrained import get_model
            from demucs.audio import AudioFile

            logger.info(f"Separating drums from full mix using Demucs...")

            # Load model (cache after first load)
            if not hasattr(self, '_demucs_model'):
                logger.info("Loading Demucs model (first time only)...")
                self._demucs_model = get_model('htdemucs')
                self._demucs_model.cpu()
                self._demucs_model.eval()
                logger.info("✓ Demucs model loaded")

            # Load audio
            wav = AudioFile(audio_path).read(seek_time=0, duration=None, streams=0)
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            # Separate (drums is index 0 in htdemucs)
            with torch.no_grad():
                sources = apply_model(self._demucs_model, wav[None], device='cpu', progress=False)[0]

            drums = sources[0].numpy()  # (channels, samples)

            # Save to temporary file
            temp_dir = tempfile.mkdtemp(prefix='demucs_')
            drums_path = os.path.join(temp_dir, 'drums.wav')
            sf.write(drums_path, drums.T, 44100, subtype='PCM_16')

            logger.info(f"✓ Drums separated and saved to {drums_path}")
            return drums_path

        except Exception as e:
            logger.error(f"Drum separation failed: {e}")
            raise

    def extract_log_mel_spectrogram(self, audio_path: str) -> np.ndarray:
        """
        Extract log-mel spectrogram from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Log-mel spectrogram (128, time_frames)
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)

            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=22050,
                n_fft=2048,
                hop_length=512,  # ~23ms frames (43 FPS)
                n_mels=128,
                fmin=30,  # Captures kick fundamentals
                fmax=11025,  # Nyquist
            )

            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            return log_mel_spec  # Shape: (128, time_frames)
        except Exception as e:
            logger.error(f"Failed to extract spectrogram from {audio_path}: {e}")
            raise

    @torch.no_grad()
    def transcribe_drums(self, audio_path: str) -> np.ndarray:
        """
        Run inference on audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Model predictions (time_frames, 12) with probabilities
        """
        try:
            # Extract features
            spec = self.extract_log_mel_spectrogram(audio_path)

            # Convert to tensor: (1, 1, 128, time)
            spec_tensor = (
                torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)
            )

            # Run model
            predictions = self.model(spec_tensor)

            if self.model_type == 'hierarchical':
                # Hierarchical model returns dictionary
                # Convert to flat format (batch, time, 12)
                flat_predictions = self._convert_hierarchical_to_flat(predictions)
                # Remove batch dimension: (time, 12)
                return flat_predictions[0]
            else:
                # Flat model returns tensor (1, time, n_classes)
                predictions = predictions[0].cpu().numpy()  # (time, n_classes)
                # Apply sigmoid to get probabilities
                predictions = torch.sigmoid(torch.FloatTensor(predictions)).numpy()
                return predictions

        except Exception as e:
            logger.error(f"Failed to transcribe drums from {audio_path}: {e}")
            raise

    def extract_onsets(
        self,
        predictions: np.ndarray,
        threshold: float = 0.5,
        min_interval: float = 0.05,
    ) -> List[Tuple[float, str, int]]:
        """
        Convert frame-level predictions to discrete drum onsets.

        Args:
            predictions: (time, 12) array of probabilities for 12-class drum mapping
            threshold: Minimum probability for detection (0-1)
            min_interval: Minimum time between onsets in seconds

        Returns:
            List of (time, drum_name, velocity) tuples
        """
        onsets = []
        # Calculate frame rate based on model type
        input_frame_rate = 22050 / 512  # ~43.07 FPS (input spectrogram)

        if self.model_type == 'hierarchical':
            # Hierarchical model: no temporal reduction (pools frequency only)
            frame_rate = input_frame_rate  # ~43.07 FPS
        else:
            # Flat model: 16x temporal reduction (4 pooling layers, pool_size=2)
            pooling_reduction = 2**4
            frame_rate = input_frame_rate / pooling_reduction  # ~2.69 FPS

        min_distance_frames = int(min_interval * frame_rate)

        for class_idx, drum_name in enumerate(DRUM_NAMES):
            # Get predictions for this drum
            class_preds = predictions[:, class_idx]

            # Find peaks above threshold
            peaks, _ = find_peaks(
                class_preds, height=threshold, distance=max(1, min_distance_frames)
            )

            # Convert to onsets
            for peak_idx in peaks:
                onset_time = peak_idx / frame_rate
                velocity = int(min(127, max(1, class_preds[peak_idx] * 127)))
                onsets.append((onset_time, drum_name, velocity))

        # Sort by time
        onsets.sort(key=lambda x: x[0])
        return onsets

    def create_midi(
        self, onsets: List[Tuple[float, str, int]], output_path: str, tempo: int = 120
    ):
        """
        Create MIDI file from drum onsets using 12-class drum mapping.

        Args:
            onsets: List of (time, drum_name, velocity) tuples
            output_path: Path to save MIDI file
            tempo: BPM (default 120)
        """
        try:
            # Create MIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

            # Create drum instrument (channel 9 = channel 10 in GM)
            drums = pretty_midi.Instrument(program=0, is_drum=True)

            # Add notes
            for time, drum_name, velocity in onsets:
                # Look up MIDI note from drum name
                if drum_name not in DRUM_MAP:
                    logger.warning(f"Unmapped drum: {drum_name}, using kick")
                    note_number = DRUM_MAP["kick"]
                else:
                    note_number = DRUM_MAP[drum_name]

                # Use appropriate drum hit duration (short, percussive)
                # Drums are typically short hits, not sustained notes
                duration = 0.05  # 50ms - typical drum hit duration

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_number,
                    start=time,
                    end=time + duration,
                )
                drums.notes.append(note)

            # Add to MIDI and save
            midi.instruments.append(drums)
            midi.write(output_path)
            logger.info(f"MIDI file saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to create MIDI file: {e}")
            raise

    def transcribe_to_midi(
        self,
        audio_path: str,
        output_midi_path: str,
        threshold: float = 0.5,
        min_interval: float = 0.05,
        tempo: int = 120,
        export_predictions: bool = False,
        separate_stems: bool = True,
    ) -> Dict:
        """
        Complete pipeline: audio file → MIDI file using 12-class drum model.

        Args:
            audio_path: Path to input audio
            output_midi_path: Path to save MIDI file
            threshold: Onset detection threshold (0-1)
            min_interval: Minimum time between onsets (seconds)
            tempo: BPM for output MIDI
            export_predictions: Export raw model predictions for debugging
            separate_stems: Whether to separate drums before transcription (default: True)

        Returns:
            Dictionary with statistics
        """
        drums_path = None
        temp_dir = None

        try:
            # Separate drums if requested
            if separate_stems:
                logger.info("Stem separation enabled")
                drums_path = self._separate_drums(audio_path)
                audio_to_transcribe = drums_path
            else:
                logger.info("Stem separation disabled, transcribing full mix")
                audio_to_transcribe = audio_path

            # Run inference on isolated drums (or full mix)
            predictions = self.transcribe_drums(audio_to_transcribe)

            # Export predictions for debugging if requested
            if export_predictions:
                self._export_predictions(audio_to_transcribe, predictions)

            # Extract onsets
            onsets = self.extract_onsets(predictions, threshold, min_interval)

            # Export to MIDI
            self.create_midi(onsets, output_midi_path, tempo)

            # Return statistics with 12-class drum names
            stats = {name: 0 for name in DRUM_NAMES}
            for _, drum_name, _ in onsets:
                if drum_name in stats:
                    stats[drum_name] += 1

            # Calculate duration based on model type
            input_frame_rate = 22050 / 512
            if self.model_type == 'hierarchical':
                output_frame_rate = input_frame_rate
            else:
                time_reduction = 2**4
                output_frame_rate = input_frame_rate / time_reduction

            duration = predictions.shape[0] / output_frame_rate

            return {
                "total_hits": len(onsets),
                "per_drum": stats,
                "duration": duration,
                "onsets": onsets,
                "stem_separation_used": separate_stems,
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

        finally:
            # Cleanup temporary separated stems
            if drums_path and os.path.exists(drums_path):
                try:
                    temp_dir = os.path.dirname(drums_path)
                    import shutil
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary stems: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary files: {e}")

    def _export_predictions(self, audio_path: str, predictions: np.ndarray):
        """
        Export raw model predictions for debugging.

        Args:
            audio_path: Path to input audio file
            predictions: Raw model predictions (time, 12) for 12-class drum mapping
        """
        try:
            import json
            from pathlib import Path

            # Create output directory
            audio_file = Path(audio_path)
            output_dir = audio_file.parent
            base_name = audio_file.stem

            # Save raw predictions as numpy
            predictions_file = output_dir / f"{base_name}_predictions.npy"
            np.save(predictions_file, predictions)

            # Save as JSON for easier inspection
            # Calculate frame rate based on model type
            input_frame_rate = 22050 / 512  # ~43.07 FPS
            if self.model_type == 'hierarchical':
                # Hierarchical: no temporal reduction
                frame_rate = input_frame_rate
            else:
                # Flat: 16x temporal reduction (4 pooling layers)
                frame_rate = input_frame_rate / 16

            predictions_data = {
                "audio_file": str(audio_path),
                "predictions_shape": predictions.shape,
                "frame_rate": frame_rate,
                "duration_seconds": predictions.shape[0] / frame_rate,
                "drum_names": DRUM_NAMES,
                "model_type": self.model_type,
                "statistics": {},
            }

            # Add statistics for each drum
            for i, drum in enumerate(DRUM_NAMES):
                class_preds = predictions[:, i]
                predictions_data["statistics"][drum] = {
                    "min": float(class_preds.min()),
                    "max": float(class_preds.max()),
                    "mean": float(class_preds.mean()),
                    "frames_above_0.5": int((class_preds > 0.5).sum()),
                    "frames_above_0.7": int((class_preds > 0.7).sum()),
                    "frames_above_0.9": int((class_preds > 0.9).sum()),
                }

            # Save sample of predictions (first 100 frames)
            predictions_data["sample_predictions"] = []
            sample_size = min(100, predictions.shape[0])
            for t in range(sample_size):
                time_sec = t / frame_rate
                frame_data = {"time": time_sec, "frame": t, "predictions": {}}
                for i, drum in enumerate(DRUM_NAMES):
                    frame_data["predictions"][drum] = float(predictions[t, i])
                predictions_data["sample_predictions"].append(frame_data)

            # Save JSON
            json_file = output_dir / f"{base_name}_predictions.json"
            with open(json_file, "w") as f:
                json.dump(predictions_data, f, indent=2)

            logger.info(f"Predictions exported: {predictions_file}, {json_file}")

        except Exception as e:
            logger.warning(f"Failed to export predictions: {e}")
