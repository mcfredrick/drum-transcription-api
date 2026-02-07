# Changelog

All notable changes to the Drum Transcription API will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial API implementation
- FastAPI server with Pydantic validation
- Drum transcription pipeline using CRNN model
- MIDI file generation with correct note mapping
- File upload handling and validation
- Health check endpoint
- Comprehensive documentation

### Features
- Audio file transcription to MIDI
- Configurable parameters (threshold, interval, tempo)
- Alternative MIDI note selection
- Real-time processing statistics
- Error handling and validation
- Automatic API documentation

## [0.1.0] - 2026-02-07

### Added
- FastAPI-based REST API server
- Drum transcription using trained CRNN model
- Support for MP3, WAV, M4A, FLAC audio formats
- MIDI output with General MIDI drum mapping
- Pydantic models for request/response validation
- File upload with size and format validation
- Health check and status endpoints
- Automatic OpenAPI documentation
- UV dependency management
- Docker deployment configuration
- Comprehensive test suite
- Development and deployment documentation

### Technical Details
- **Model**: CRNN (Convolutional Recurrent Neural Network)
- **Input**: Log-mel spectrogram (128 bins, 22050 Hz)
- **Output**: 8-class drum classification
- **MIDI Mapping**: Compatible with target application
  - Kick: Notes 35, 36
  - Snare: Notes 37, 38, 40
  - Hi-Hat: Notes 42, 44, 46
  - High Tom: Notes 48, 50
  - Mid Tom: Notes 45, 47
  - Floor Tom: Notes 41, 43
  - Crash: Notes 49, 52, 55, 57
  - Ride: Notes 51, 53, 59

### API Endpoints
- `GET /` - API information and configuration
- `GET /health` - Health check and model status
- `POST /transcribe` - Audio file transcription
- `GET /download/{filename}` - MIDI file download

### Dependencies
- FastAPI 0.104.0+
- PyTorch 2.4.0+
- Lightning 2.4.0+
- Librosa 0.10.2+
- Pretty MIDI 0.2.10+
- UV package manager

### Documentation
- API Reference with detailed endpoint documentation
- Development guide with setup and contribution guidelines
- Deployment guide for production environments
- Usage examples with curl, Python, and JavaScript
- Comprehensive README with quick start guide

## [Future Plans]

### Planned Features
- [ ] Batch processing for multiple files
- [ ] Real-time streaming transcription
- [ ] Additional audio format support
- [ ] Custom MIDI mapping configuration
- [ ] User authentication and API keys
- [ ] Rate limiting and quota management
- [ ] Performance monitoring and metrics
- [ ] Model versioning and A/B testing
- [ ] Audio preprocessing options
- [ ] Export to other formats (MusicXML, etc.)

### Improvements
- [ ] Enhanced error handling and recovery
- [ ] Caching for repeated requests
- [ ] GPU memory optimization
- [ ] Asynchronous processing queue
- [ ] Webhook support for completed jobs
- [ ] Audio quality analysis
- [ ] Transcription confidence scores
- [ ] Drum pattern recognition
- [ ] Tempo detection and adjustment

### Infrastructure
- [ ] Kubernetes deployment templates
- [ ] CI/CD pipeline setup
- [ ] Automated testing pipeline
- [ ] Performance benchmarking
- [ ] Load testing suite
- [ ] Security audit and hardening
- [ ] Monitoring and alerting setup
- [ ] Backup and disaster recovery

---

## Version History

### Version 0.1.0 (Initial Release)
- **Release Date**: February 7, 2026
- **Status**: Production Ready
- **Compatibility**: Python 3.10+, CUDA 11.8+
- **Model**: Full training checkpoint (epoch 99, val_loss 0.0528)

### Breaking Changes
None in current version.

### Upgrade Notes
- Ensure model checkpoint path is correctly configured
- Install UV package manager for dependency management
- Verify CUDA availability for GPU acceleration

### Migration Guide
No migration needed for initial release.

---

## Support and Contributing

### Reporting Issues
- Use GitHub Issues for bug reports
- Include audio file examples for transcription issues
- Provide system information (OS, Python version, GPU)

### Contributing
- Follow the development guide in `docs/DEVELOPMENT.md`
- Ensure all tests pass before submitting PRs
- Follow the established code style and documentation standards

### Security
- Report security vulnerabilities privately
- Follow responsible disclosure practices
- Keep dependencies updated regularly
