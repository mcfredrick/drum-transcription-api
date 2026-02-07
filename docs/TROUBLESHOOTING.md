# Troubleshooting Guide

## Common Issues and Solutions

### Installation and Setup

#### UV Sync Fails
**Problem**: `uv sync` command fails with dependency conflicts

**Solutions**:
```bash
# Clear UV cache
uv cache clean

# Force reinstall
uv sync --reinstall

# Check Python version compatibility
python --version  # Should be 3.10+

# Update UV to latest version
pip install --upgrade uv
```

#### Model Loading Errors
**Problem**: "Model checkpoint not found" or "Failed to load model"

**Solutions**:
1. **Check checkpoint path**:
   ```python
   # In drum_transcription_api/main.py, verify:
   model_checkpoint = "/mnt/hdd/drum-tranxn/checkpoints/full-training-epoch=99-val_loss=0.0528.ckpt"
   ```

2. **Verify file exists**:
   ```bash
   ls -la /mnt/hdd/drum-tranxn/checkpoints/full-training-epoch=99-val_loss=0.0528.ckpt
   ```

3. **Check permissions**:
   ```bash
   chmod 644 /path/to/checkpoint.ckpt
   ```

4. **Update path in main.py**:
   ```python
   model_checkpoint = "/correct/path/to/checkpoint.ckpt"
   ```

#### CUDA Issues
**Problem**: CUDA out of memory or CUDA not available

**Solutions**:
1. **Check CUDA availability**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")
   ```

2. **Force CPU usage**:
   ```python
   # In main.py, change device initialization
   transcriber = DrumTranscriber(model_checkpoint, device="cpu")
   ```

3. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Runtime Issues

#### File Upload Fails
**Problem**: "Unsupported file type" or file upload errors

**Solutions**:
1. **Check supported formats**:
   - MP3: `audio/mpeg`, `audio/mp3`
   - WAV: `audio/wav`, `audio/x-wav`
   - M4A: `audio/m4a`
   - FLAC: `audio/flac`

2. **Verify file size** (default 50MB limit):
   ```python
   # Check file size before upload
   import os
   file_size = os.path.getsize("your_file.mp3") / (1024 * 1024)  # MB
   print(f"File size: {file_size:.2f} MB")
   ```

3. **Convert audio format**:
   ```bash
   # Using ffmpeg
   ffmpeg -i input.wav -codec:a mp3 output.mp3
   ffmpeg -i input.m4a -codec:a wav output.wav
   ```

#### Transcription Fails
**Problem**: "Transcription failed" error

**Solutions**:
1. **Check audio quality**:
   ```python
   import librosa
   y, sr = librosa.load("your_file.mp3")
   print(f"Duration: {len(y)/sr:.2f} seconds")
   print(f"Sample rate: {sr}")
   ```

2. **Adjust parameters**:
   ```bash
   # Try with lower threshold
   curl -X POST "http://localhost:8000/transcribe" \
     -F "file=@your_file.mp3" \
     -F "threshold=0.3" \
     -F "min_interval=0.1"
   ```

3. **Check logs**:
   ```bash
   # View server logs
   tail -f /var/log/drum-transcription-api.log
   ```

#### MIDI Download Fails
**Problem**: MIDI file not found or download fails

**Solutions**:
1. **File expiration**: MIDI files are temporary, re-run transcription
2. **Check URL**: Ensure correct download URL from response
3. **Verify file creation**:
   ```python
   # Check if MIDI file was created
   import os
   midi_path = "/tmp/drum-transcription/output.mid"
   if os.path.exists(midi_path):
       print(f"MIDI file size: {os.path.getsize(midi_path)} bytes")
   ```

### Performance Issues

#### Slow Response Times
**Problem**: Transcription takes too long

**Solutions**:
1. **Use GPU acceleration**:
   ```python
   # Ensure CUDA is available and being used
   transcriber = DrumTranscriber(model_checkpoint, device="cuda")
   ```

2. **Optimize audio length**:
   ```bash
   # Split long files into chunks
   ffmpeg -i long_track.mp3 -t 300 first_5min.mp3
   ```

3. **Monitor system resources**:
   ```bash
   # Check GPU usage
   nvidia-smi
   
   # Check CPU and memory
   htop
   ```

#### Memory Issues
**Problem**: Out of memory errors

**Solutions**:
1. **Reduce concurrent requests**:
   ```python
   # Limit concurrent processing
   import asyncio
   semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests
   ```

2. **Clear temporary files**:
   ```bash
   # Clean temp directory
   rm -rf /tmp/drum-transcription/*
   ```

3. **Monitor memory usage**:
   ```python
   import psutil
   memory = psutil.virtual_memory()
   print(f"Memory usage: {memory.percent}%")
   ```

### API Issues

#### Connection Refused
**Problem**: Cannot connect to API server

**Solutions**:
1. **Check if server is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check port availability**:
   ```bash
   netstat -tlnp | grep 8000
   ```

3. **Restart server**:
   ```bash
   ./start_server.sh
   # or
   uv run uvicorn drum_transcription_api.main:app --host 0.0.0.0 --port 8000
   ```

#### CORS Issues
**Problem**: Browser blocks API requests

**Solutions**:
1. **Add CORS middleware**:
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Be specific in production
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Use browser extension for testing** (development only)

### Docker Issues

#### Container Fails to Start
**Problem**: Docker container exits immediately

**Solutions**:
1. **Check logs**:
   ```bash
   docker logs drum-transcription-api
   ```

2. **Verify model volume**:
   ```bash
   # Ensure model files are accessible
   docker run --rm -v /path/to/models:/models alpine ls -la /models
   ```

3. **Check GPU support**:
   ```bash
   # Install nvidia-docker2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   ```

#### Build Fails
**Problem**: Docker build fails during dependency installation

**Solutions**:
1. **Use multi-stage build**:
   ```dockerfile
   FROM python:3.11-slim as builder
   WORKDIR /app
   COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
   COPY pyproject.toml uv.lock ./
   RUN uv sync --frozen
   
   FROM python:3.11-slim
   WORKDIR /app
   COPY --from=builder /app/.venv /app/.venv
   COPY . .
   CMD ["/app/.venv/bin/python", "-m", "drum_transcription_api.main"]
   ```

2. **Clear Docker cache**:
   ```bash
   docker system prune -a
   ```

### Testing Issues

#### Tests Fail
**Problem**: Unit or integration tests failing

**Solutions**:
1. **Check test dependencies**:
   ```bash
   uv sync --dev
   ```

2. **Run specific test**:
   ```bash
   uv run pytest tests/test_transcription.py::test_specific_function -v
   ```

3. **Debug with pdb**:
   ```python
   import pdb; pdb.set_trace()
   ```

#### API Test Failures
**Problem**: `test_api.py` cannot connect to server

**Solutions**:
1. **Start server first**:
   ```bash
   # Terminal 1: Start server
   ./start_server.sh
   
   # Terminal 2: Run tests
   uv run python test_api.py
   ```

2. **Check server health**:
   ```bash
   curl http://localhost:8000/health
   ```

3. **Use test audio file**:
   ```bash
   # Download test audio
   wget https://example.com/test-drums.mp3
   uv run python test_api.py test-drums.mp3
   ```

### Debugging Tools

#### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or for specific module
logging.getLogger('drum_transcription_api').setLevel(logging.DEBUG)
```

#### Monitor Resources
```python
import psutil
import torch

def print_system_info():
    print(f"CPU: {psutil.cpu_percent()}%")
    print(f"Memory: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

#### Profile Performance
```python
import cProfile
import pstats

def profile_transcription():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run transcription
    result = transcribe_audio("test.mp3")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

### Getting Help

#### Collect Debug Information
```bash
# System info
uname -a
python --version
uv --version

# GPU info
nvidia-smi
torch.version.cuda

# API status
curl -s http://localhost:8000/health | jq .

# Logs
journalctl -u drum-transcription-api -f
```

#### Report Issues
Include the following in bug reports:
1. **System information**: OS, Python version, GPU details
2. **Error messages**: Full traceback and logs
3. **Audio file details**: Format, duration, size
4. **API parameters**: Threshold, interval, etc.
5. **Expected vs actual behavior**: What you expected vs what happened

#### Community Support
- Check existing GitHub issues
- Review documentation in `docs/`
- Run the test suite to verify installation
- Try with different audio files to isolate the issue
