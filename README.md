# Drum Transcription API

FastAPI server for transcribing drum audio to MIDI using a trained CRNN model.

## Documentation

- **[API Reference](docs/API_REFERENCE.md)** - Complete API documentation
- **[Usage Examples](USAGE_EXAMPLES.md)** - Detailed examples and code samples
- **[Development Guide](docs/DEVELOPMENT.md)** - Setup and contribution guidelines
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment instructions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project
- **[Security Policy](SECURITY.md)** - Security information and vulnerability reporting
- **[Changelog](docs/CHANGELOG.md)** - Version history and changes

## Quick Start

```bash
# Install dependencies with uv
uv sync

# Run the server
uv run python -m drum_transcription_api.main

# Or with uvicorn directly
uv run uvicorn drum_transcription_api.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### POST /transcribe
Upload an audio file and receive MIDI transcription.

**Request:** 
- multipart/form-data with audio file

**Response:**
- JSON with transcription results and MIDI file download URL

### GET /health
Health check endpoint.

## Model Configuration

The API uses the best checkpoint from training:
```
/mnt/hdd/drum-tranxn/checkpoints/full-training-epoch=99-val_loss=0.0528.ckpt
```

## Model Output

The trained CRNN model outputs 8 drum classes with the following characteristics:

### Drum Classes
- **Class 0**: Kick
- **Class 1**: Snare  
- **Class 2**: Hi-Hat (includes closed, pedal, and open variants)
- **Class 3**: High Tom
- **Class 4**: Mid Tom
- **Class 5**: Low Tom (includes floor tom variants)
- **Class 6**: Crash Cymbal
- **Class 7**: Ride Cymbal

### Output Format
- **Frame rate**: ~43 FPS (hop_length=512, sr=22050)
- **Activation threshold**: 0.5 (configurable)
- **Post-processing**: Median filtering and minimum duration constraints

## MIDI Note Mapping

The API converts model outputs to MIDI files using the following note mappings:

- **Kick**: Notes 35, 36 (Bass Drum 1, Bass Drum 2)
- **Snare**: Notes 37, 38, 40 (Side Stick, Acoustic Snare, Electric Snare)  
- **Hi-Hat**: Notes 42, 44, 46 (Closed, Pedal, Open Hi-Hat)
- **High Tom**: Notes 48, 50 (Hi-Mid Tom, High Tom)
- **Mid Tom**: Notes 45, 47 (Low Tom, Low-Mid Tom)
- **Low Tom**: Notes 41, 43 (Floor Tom variants)
- **Crash**: Notes 49, 52, 55, 57 (Crash Cymbal variants)
- **Ride**: Notes 51, 53, 59 (Ride Cymbal variants)

All notes are on MIDI channel 10 (channel 9 in zero-indexed terms) with default velocity 80.

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run ruff check .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 **Documentation**: See the [docs/](docs/) directory for comprehensive guides
- 🐛 **Bug Reports**: Open an issue on GitHub
- 💡 **Feature Requests**: Open an issue with the "enhancement" label
- 🔒 **Security**: Email security@your-domain.com for security issues

## Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Model trained with [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)
- Audio processing with [Librosa](https://librosa.org/)
- MIDI generation with [Pretty MIDI](https://craffel.github.io/pretty-midi/)
