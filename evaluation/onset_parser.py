"""Parse MIDI files to extract drum onsets."""

import pretty_midi
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path

# TD-17 to 12-class mapping (from E-GMD dataset)
TD17_TO_12CLASS = {
    36: 'kick',           # Kick
    38: 'snare_head',     # Snare Head
    40: 'snare_rim',      # Snare Rim
    37: 'side_stick',     # Side Stick (cross stick)
    44: 'hihat_pedal',    # Hi-Hat Pedal
    42: 'hihat_closed',   # Hi-Hat Closed
    46: 'hihat_open',     # Hi-Hat Open
    43: 'floor_tom',      # Floor Tom (or Tom 3)
    48: 'high_mid_tom',   # High-Mid Tom (Tom 1)
    45: 'high_mid_tom',   # Tom 2 → merge with high_mid
    51: 'ride',           # Ride
    53: 'ride_bell',      # Ride Bell
    49: 'crash',          # Crash
    55: 'crash',          # Crash edge → merge with crash
    57: 'crash',          # Crash 2 → merge with crash
}


class OnsetParser:
    """Parse onsets from MIDI and prediction files."""

    @staticmethod
    def parse_midi(midi_path: str, note_mapping: Dict[int, str] = None) -> List[Tuple[float, str]]:
        """
        Parse MIDI file and extract drum onsets.

        Args:
            midi_path: Path to MIDI file
            note_mapping: Optional custom MIDI note to drum name mapping

        Returns:
            List of (time, drum_name) tuples, sorted by time
        """
        if note_mapping is None:
            note_mapping = TD17_TO_12CLASS

        midi = pretty_midi.PrettyMIDI(midi_path)
        onsets = []

        # Get drum track (usually the only instrument, or first instrument)
        for instrument in midi.instruments:
            if instrument.is_drum:
                for note in instrument.notes:
                    drum_name = note_mapping.get(note.pitch)
                    if drum_name:
                        onsets.append((note.start, drum_name))

        # Sort by time
        onsets.sort(key=lambda x: x[0])
        return onsets

    @staticmethod
    def parse_predictions(predictions: List[Tuple[float, str, int]]) -> List[Tuple[float, str]]:
        """
        Parse model predictions (from transcription output).

        Args:
            predictions: List of (time, drum_name, velocity) tuples

        Returns:
            List of (time, drum_name) tuples, sorted by time
        """
        # Remove velocity, keep time and drum_name
        onsets = [(time, drum) for time, drum, _ in predictions]
        onsets.sort(key=lambda x: x[0])
        return onsets
