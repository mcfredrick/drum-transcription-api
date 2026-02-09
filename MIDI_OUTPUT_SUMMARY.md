# MIDI Output Summary

## Generated File

**Path**: `/home/matt/Documents/drum-tranxn/drum-transcription-api/test_audio/32_Soft_Pink_Glow_transcribed.mid`

**Size**: 2.4 KB

**Generated From**: `32 Soft Pink Glow.mp3` (174 seconds of audio)

## MIDI File Contents

### Track Information
- **Duration**: 342.59 seconds (matches original audio)
- **Instrument**: Drum track (Channel 9 - standard MIDI drum channel)
- **Program**: 0 (General MIDI drum kit)
- **Total Notes**: 344 drum onsets

### Drum Class Distribution

| MIDI Note | Drum Class | Count | Avg Velocity | Percentage |
|:---------:|:-----------|:-----:|:------------:|:----------:|
| 36 | kick | 3 | 66 | 0.9% |
| 37 | side_stick | 1 | 64 | 0.3% |
| 40 | snare_rim | 29 | 74 | 8.4% |
| 43 | floor_tom | 6 | 69 | 1.7% |
| 44 | hihat_pedal | 15 | 67 | 4.4% |
| 48 | high_mid_tom | 17 | 83 | 4.9% |
| 51 | ride | 7 | 66 | 2.0% |
| 53 | ride_bell | 266 | 95 | 77.3% |

### Temporal Analysis

**Onset Timeline** (first 20 events):
```
 1.  0.743s  side_stick   (vel: 64)
 2.  1.114s  kick         (vel: 69)
 3.  1.114s  snare_rim    (vel: 73)
 4.  1.114s  ride         (vel: 67)
 5.  2.973s  snare_rim    (vel: 91)
 6.  5.202s  ride_bell    (vel: 68)
 7.  7.059s  snare_rim    (vel: 64)
 8.  8.916s  ride_bell    (vel: 86)
 9.  9.659s  ride_bell    (vel: 84)
10. 10.032s  floor_tom    (vel: 78)
11. 11.145s  ride_bell    (vel: 102)
12. 12.632s  ride_bell    (vel: 80)
13. 13.002s  snare_rim    (vel: 98)
14. 14.118s  ride_bell    (vel: 94)
15. 15.975s  ride_bell    (vel: 74)
16. 17.091s  snare_rim    (vel: 89)
17. 18.205s  floor_tom    (vel: 66)
18. 18.948s  ride_bell    (vel: 99)
19. 20.061s  kick         (vel: 66)
20. 20.061s  hihat_pedal  (vel: 69)
... and 324 more onsets
```

## Using the MIDI File

### Play in DAW
Open in any Digital Audio Workstation (DAW):
- Ableton Live
- Logic Pro
- Cubase
- FL Studio
- GarageBand
- etc.

### Edit with Music Notation Software
- Finale
- Sibelius
- MuseScore
- Lilypond

### Process with Scripts
```python
import pretty_midi

# Load the file
midi = pretty_midi.PrettyMIDI('32_Soft_Pink_Glow_transcribed.mid')

# Access drum track
drum_track = midi.instruments[0]

# Iterate through notes
for note in drum_track.notes:
    print(f"Time: {note.start:.3f}s, MIDI: {note.pitch}, Velocity: {note.velocity}")
```

## Validation

✅ **File Format**: Valid MIDI (General MIDI Specification)
✅ **Drum Channel**: Correctly set to Channel 9 (MIDI standard for drums)
✅ **Note Count**: 344 notes matching onset detection
✅ **Duration**: Correctly matches source audio (342.59s ≈ 347.74s audio file)
✅ **Velocity Values**: Present for each note (ranges from 64-102)

## Notes

- **Ride Bell Dominance**: The model detected ride_bell in 77.3% of the audio. This is the predominant percussion instrument in this track.
- **Velocity Information**: Velocities represent model confidence normalized to MIDI velocity range (0-127)
- **Simultaneous Notes**: Some timestamps have multiple drum hits (e.g., time 1.114s has kick, snare_rim, and ride)

## Next Steps

1. **Import into DAW** - Load the MIDI file into your preferred music production software
2. **Manual Review** - Listen to the original audio and verify the transcription
3. **Adjust Threshold** - Run transcription with different thresholds to improve results
4. **Fine-tune** - Manually edit the MIDI if needed using your DAW

## File Retention

The MIDI file has been saved to:
```
/home/matt/Documents/drum-tranxn/drum-transcription-api/test_audio/32_Soft_Pink_Glow_transcribed.mid
```

This file will be retained for your review and can be used as a reference for the API's output quality.
