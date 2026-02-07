# Development Guide

## Project Structure

```
drum-transcription-api/
├── drum_transcription_api/          # Main package
│   ├── __init__.py                  # Package initialization
│   ├── main.py                      # FastAPI application
│   └── transcription.py             # Core transcription logic
├── docs/                            # Documentation
│   ├── API_REFERENCE.md             # API documentation
│   ├── DEVELOPMENT.md               # Development guide
│   └── DEPLOYMENT.md                # Deployment instructions
├── tests/                           # Test files
├── pyproject.toml                   # UV project configuration
├── README.md                        # Project overview
├── USAGE_EXAMPLES.md                # Usage examples
├── test_api.py                      # API test script
└── start_server.sh                  # Startup script
```

## Setup

### Prerequisites

- Python 3.10+
- UV package manager
- CUDA (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd drum-transcription-api

# Install dependencies with UV
uv sync

# Install development dependencies
uv sync --dev
```

### Configuration

#### Model Checkpoint

Update the model checkpoint path in `drum_transcription_api/main.py`:

```python
model_checkpoint = "/path/to/your/checkpoint.ckpt"
```

#### Environment Variables

Create a `.env` file for configuration:

```bash
# Model settings
MODEL_CHECKPOINT="/path/to/model.ckpt"
DEVICE="auto"  # auto, cuda, cpu

# Server settings
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"

# File settings
MAX_FILE_SIZE="52428800"  # 50MB in bytes
TEMP_DIR="/tmp/drum-transcription"
```

## Code Architecture

### Core Components

#### 1. FastAPI Application (`main.py`)

- **Purpose**: HTTP server and request handling
- **Key Features**: 
  - File upload handling
  - Parameter validation
  - Response formatting
  - Error handling

#### 2. Transcription Pipeline (`transcription.py`)

- **Purpose**: Audio processing and MIDI generation
- **Key Classes**:
  - `DrumTranscriber`: Main transcription class
- **Key Methods**:
  - `extract_log_mel_spectrogram()`: Audio feature extraction
  - `transcribe_drums()`: Model inference
  - `extract_onsets()`: Peak detection
  - `create_midi()`: MIDI file generation

### Data Flow

```
Audio Upload → Validation → Feature Extraction → Model Inference → 
Peak Detection → MIDI Generation → File Download
```

## Development Workflow

### 1. Making Changes

```bash
# Start development server with auto-reload
uv run uvicorn drum_transcription_api.main:app --reload

# Or use the startup script
./start_server.sh
```

### 2. Testing

```bash
# Run unit tests
uv run pytest

# Run integration tests
uv run python test_api.py

# Test with coverage
uv run pytest --cov=drum_transcription_api
```

### 3. Code Quality

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

## Adding Features

### New Endpoints

1. Define Pydantic models for request/response
2. Add endpoint function with proper validation
3. Update API documentation
4. Add tests

Example:

```python
from pydantic import BaseModel

class NewRequest(BaseModel):
    parameter: str = Field(..., min_length=1, max_length=100)

class NewResponse(BaseModel):
    result: str
    timestamp: datetime

@app.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    # Implementation
    return NewResponse(result="success", timestamp=datetime.now())
```

### Model Updates

When updating the transcription model:

1. Update model loading logic in `DrumTranscriber`
2. Modify feature extraction if needed
3. Update MIDI mapping if classes change
4. Add new parameters to API endpoints
5. Update documentation

## Testing

### Unit Tests

Create test files in the `tests/` directory:

```python
# tests/test_transcription.py
import pytest
from drum_transcription_api.transcription import DrumTranscriber

def test_drum_transcriber_initialization():
    transcriber = DrumTranscriber("test_checkpoint.ckpt", device="cpu")
    assert transcriber.device == "cpu"
```

### Integration Tests

Test the full API:

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from drum_transcription_api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
```

### API Testing

Use the provided test script:

```bash
# Test with actual audio file
uv run python test_api.py /path/to/test.mp3
```

## Performance Optimization

### GPU Acceleration

Ensure CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Caching

Implement caching for repeated requests:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_spectrogram(audio_path_hash: str):
    # Extract and cache spectrogram
    pass
```

### Batch Processing

For multiple files:

```python
async def transcribe_batch(files: List[UploadFile]):
    tasks = [transcribe_single(file) for file in files]
    results = await asyncio.gather(*tasks)
    return results
```

## Debugging

### Logging

Configure logging levels:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("Processing file: %s", filename)
logger.debug("Spectrogram shape: %s", spec.shape)
```

### Common Issues

1. **Model Loading Errors**
   - Check checkpoint path
   - Verify model compatibility
   - Check CUDA availability

2. **Audio Processing Errors**
   - Validate file format
   - Check audio duration
   - Verify sample rate

3. **Memory Issues**
   - Reduce batch size
   - Use CPU instead of GPU
   - Implement file size limits

## Contributing

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and commit
git add .
git commit -m "Add new feature"

# Push and create PR
git push origin feature/new-feature
```

### Code Style

- Follow PEP 8
- Use Black for formatting
- Use Ruff for linting
- Add type hints
- Write docstrings

### Documentation

- Update API reference for new endpoints
- Add usage examples
- Update README.md
- Document configuration options

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for production deployment instructions.
