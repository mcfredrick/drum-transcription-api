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

logger = logging.getLogger(__name__)

# MIDI drum mapping for app compatibility
GM_DRUM_MAP = {
    'kick': [35, 36],      # Bass Drum 1, Bass Drum 2
    'snare': [37, 38, 40], # Side Stick, Acoustic Snare, Electric Snare
    'hihat': [42, 44, 46], # Closed Hi-Hat, Pedal Hi-Hat, Open Hi-Hat
    'hi_tom': [48, 50],    # Hi-Mid Tom, High Tom
    'mid_tom': [45, 47],   # Low Tom, Low-Mid Tom
    'low_tom': [41, 43],   # Floor Tom (low), Floor Tom (high)
    'crash': [49, 52, 55, 57], # Crash Cymbal 1, China Cymbal, Splash Cymbal, Crash Cymbal 2
    'ride': [51, 53, 59],  # Ride Cymbal 1, Ride Bell, Ride Cymbal 2
}

# Default to first note in each list
DEFAULT_DRUM_MAP = {name: notes[0] for name, notes in GM_DRUM_MAP.items()}


class DrumTranscriber:
    """Drum transcription pipeline using trained CRNN model."""
    
    def __init__(self, model_checkpoint: str, device: str = 'auto'):
        """
        Initialize the transcriber.
        
        Args:
            model_checkpoint: Path to the trained model checkpoint
            device: 'auto', 'cuda', or 'cpu'
        """
        self.model_checkpoint = model_checkpoint
        self.device = self._setup_device(device)
        self.model = None
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup computation device."""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self):
        """Load the trained model."""
        try:
            # Import here to avoid dependency issues if model isn't available
            import sys
            sys.path.append('/home/matt/Documents/drum-tranxn/drum_transcription')
            from src.models.crnn import DrumTranscriptionCRNN
            
            logger.info(f"Loading model from {self.model_checkpoint}")
            self.model = DrumTranscriptionCRNN.load_from_checkpoint(self.model_checkpoint)
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
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
                hop_length=512,      # ~23ms frames (43 FPS)
                n_mels=128,
                fmin=30,             # Captures kick fundamentals
                fmax=11025           # Nyquist
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
            Model predictions (time_frames, 8)
        """
        try:
            # Extract features
            spec = self.extract_log_mel_spectrogram(audio_path)
            
            # Convert to tensor: (1, 1, 128, time)
            spec_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Run model
            predictions = self.model(spec_tensor)  # (1, time, 8)
            predictions = predictions[0].cpu().numpy()  # (time, 8)
            
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
        min_interval: float = 0.05
    ) -> List[Tuple[float, str, int]]:
        """
        Convert frame-level predictions to discrete drum onsets.
        
        Args:
            predictions: (time, 8) array of probabilities
            threshold: Minimum probability for detection (0-1)
            min_interval: Minimum time between onsets in seconds
        
        Returns:
            List of (time, drum_name, velocity) tuples
        """
        onsets = []
        # Model output frame rate accounts for 8x pooling reduction
        input_frame_rate = 22050 / 512  # ~43 FPS (input spectrogram)
        pooling_reduction = 8  # 2^3 from three pooling layers
        frame_rate = input_frame_rate / pooling_reduction  # ~5.38 FPS (model output)
        min_distance_frames = int(min_interval * frame_rate)
        
        drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
        
        for class_idx, drum_name in enumerate(drum_names):
            # Get predictions for this drum
            class_preds = predictions[:, class_idx]
            
            # Find peaks above threshold
            peaks, _ = find_peaks(
                class_preds,
                height=threshold,
                distance=max(1, min_distance_frames)
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
        self, 
        onsets: List[Tuple[float, str, int]], 
        output_path: str,
        tempo: int = 120,
        use_alternative_notes: bool = False
    ):
        """
        Create MIDI file from drum onsets.
        
        Args:
            onsets: List of (time, drum_name, velocity) tuples
            output_path: Path to save MIDI file
            tempo: BPM (default 120)
            use_alternative_notes: If True, randomly select from available notes
        """
        try:
            # Create MIDI object
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            
            # Create drum instrument (channel 9 = channel 10 in GM)
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            
            # Add notes
            for i, (time, drum_name, velocity) in enumerate(onsets):
                if use_alternative_notes:
                    # Randomly select from available notes for variety
                    import random
                    note_number = random.choice(GM_DRUM_MAP[drum_name])
                else:
                    # Use default (first) note
                    note_number = DEFAULT_DRUM_MAP[drum_name]
                
                # Use appropriate drum hit duration (short, percussive)
                # Drums are typically short hits, not sustained notes
                duration = 0.05  # 50ms - typical drum hit duration
                
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_number,
                    start=time,
                    end=time + duration
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
        use_alternative_notes: bool = False,
        export_predictions: bool = False
    ) -> Dict:
        """
        Complete pipeline: audio file → MIDI file.
        
        Args:
            audio_path: Path to input audio
            output_midi_path: Path to save MIDI file
            threshold: Onset detection threshold (0-1)
            min_interval: Minimum time between onsets (seconds)
            tempo: BPM for output MIDI
            use_alternative_notes: Use alternative MIDI notes for variety
            export_predictions: Export raw model predictions for debugging
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Run inference
            predictions = self.transcribe_drums(audio_path)
            
            # Export predictions for debugging if requested
            if export_predictions:
                self._export_predictions(audio_path, predictions)
            
            # Extract onsets
            onsets = self.extract_onsets(predictions, threshold, min_interval)
            
            # Export to MIDI
            self.create_midi(onsets, output_midi_path, tempo, use_alternative_notes)
            
            # Return statistics
            drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
            stats = {name: 0 for name in drum_names}
            for _, drum_name, _ in onsets:
                stats[drum_name] += 1
            
            return {
                'total_hits': len(onsets),
                'per_drum': stats,
                'duration': predictions.shape[0] / ((22050 / 512) / 8),  # Use output frame rate
                'onsets': onsets
            }
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _export_predictions(self, audio_path: str, predictions: np.ndarray):
        """
        Export raw model predictions for debugging.
        
        Args:
            audio_path: Path to input audio file
            predictions: Raw model predictions (time, n_classes)
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
            drum_names = ['kick', 'snare', 'hihat', 'hi_tom', 'mid_tom', 'low_tom', 'crash', 'ride']
            frame_rate = (22050 / 512) / 8  # Output frame rate
            
            predictions_data = {
                'audio_file': str(audio_path),
                'predictions_shape': predictions.shape,
                'frame_rate': frame_rate,
                'duration_seconds': predictions.shape[0] / frame_rate,
                'drum_names': drum_names,
                'statistics': {}
            }
            
            # Add statistics for each drum
            for i, drum in enumerate(drum_names):
                class_preds = predictions[:, i]
                predictions_data['statistics'][drum] = {
                    'min': float(class_preds.min()),
                    'max': float(class_preds.max()),
                    'mean': float(class_preds.mean()),
                    'frames_above_0.5': int((class_preds > 0.5).sum()),
                    'frames_above_0.7': int((class_preds > 0.7).sum()),
                    'frames_above_0.9': int((class_preds > 0.9).sum())
                }
            
            # Save sample of predictions (first 100 frames)
            predictions_data['sample_predictions'] = []
            sample_size = min(100, predictions.shape[0])
            for t in range(sample_size):
                time_sec = t / frame_rate
                frame_data = {
                    'time': time_sec,
                    'frame': t,
                    'predictions': {}
                }
                for i, drum in enumerate(drum_names):
                    frame_data['predictions'][drum] = float(predictions[t, i])
                predictions_data['sample_predictions'].append(frame_data)
            
            # Save JSON
            json_file = output_dir / f"{base_name}_predictions.json"
            with open(json_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            logger.info(f"Predictions exported: {predictions_file}, {json_file}")
            
        except Exception as e:
            logger.warning(f"Failed to export predictions: {e}")
