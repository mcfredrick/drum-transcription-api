# Contributing to Drum Transcription API

Thank you for your interest in contributing to the Drum Transcription API! This guide will help you get started with contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct.html). Please read and follow these guidelines to ensure a welcoming environment for all contributors.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- UV package manager
- Git
- Basic knowledge of FastAPI, PyTorch, and audio processing

### Setup

1. **Fork the repository**:
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/drum-transcription-api.git
   cd drum-transcription-api
   ```

2. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/drum-transcription-api.git
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   uv sync --dev  # For development dependencies
   ```

4. **Verify setup**:
   ```bash
   uv run python -c "from drum_transcription_api.main import app; print('Setup successful')"
   ```

### Development Environment

1. **Create a virtual environment** (if not using UV):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```

3. **Start development server**:
   ```bash
   uv run uvicorn drum_transcription_api.main:app --reload
   ```

## Development Workflow

### 1. Create a Branch

```bash
# Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Follow the coding standards outlined below
- Write tests for new functionality
- Update documentation as needed
- Keep changes focused and atomic

### 3. Test Your Changes

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=drum_transcription_api

# Run specific test
uv run pytest tests/test_transcription.py -v

# Test API manually
uv run python test_api.py /path/to/test.mp3
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add real-time transcription endpoint"

# Or for bug fixes
git commit -m "fix: resolve memory leak in audio processing"
```

### 5. Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

## Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with the following tools:

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

### Type Hints

All functions should include type hints:

```python
from typing import List, Optional, Dict, Any
import numpy as np

def extract_onsets(
    predictions: np.ndarray, 
    threshold: float = 0.5, 
    min_interval: float = 0.05
) -> List[Tuple[float, str, int]]:
    """Extract drum onsets from model predictions."""
    # Implementation
    pass
```

### Documentation

All public functions and classes should have docstrings:

```python
class DrumTranscriber:
    """Drum transcription pipeline using trained CRNN model.
    
    Args:
        model_checkpoint: Path to the trained model checkpoint
        device: Computation device ('auto', 'cuda', 'cpu')
    
    Attributes:
        model: Loaded PyTorch model
        device: Active computation device
    """
    
    def transcribe_to_midi(
        self,
        audio_path: str,
        output_midi_path: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Transcribe audio file to MIDI.
        
        Args:
            audio_path: Path to input audio file
            output_midi_path: Path to save output MIDI file
            threshold: Detection threshold (0.0-1.0)
        
        Returns:
            Dictionary with transcription statistics
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio format is unsupported
        """
        pass
```

### Error Handling

Use specific exceptions and proper error messages:

```python
import logging

logger = logging.getLogger(__name__)

def process_audio(file_path: str) -> np.ndarray:
    """Process audio file with proper error handling."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Check file format
        if not file_path.lower().endswith(('.mp3', '.wav', '.m4a', '.flac')):
            raise ValueError(f"Unsupported audio format: {file_path}")
        
        # Process audio
        audio, sr = librosa.load(file_path, sr=22050)
        logger.info(f"Loaded audio: {len(audio)/sr:.2f}s at {sr}Hz")
        
        return audio
        
    except Exception as e:
        logger.error(f"Failed to process audio {file_path}: {e}")
        raise
```

## Testing

### Test Structure

```
tests/
├── test_transcription.py      # Core transcription tests
├── test_api.py               # API endpoint tests
├── test_models.py            # Pydantic model tests
└── conftest.py               # Test configuration
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch
from drum_transcription_api.transcription import DrumTranscriber

class TestDrumTranscriber:
    """Test cases for DrumTranscriber class."""
    
    @pytest.fixture
    def mock_transcriber(self):
        """Create a mock transcriber for testing."""
        with patch('drum_transcription_api.transcription.DrumTranscriptionCRNN'):
            return DrumTranscriber("fake_checkpoint.ckpt", device="cpu")
    
    def test_extract_spectrogram(self, mock_transcriber):
        """Test spectrogram extraction."""
        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            # Save test audio
            test_audio = np.random.randn(22050)  # 1 second of audio
            sf.write(f.name, test_audio, 22050)
            
            # Test extraction
            spec = mock_transcriber.extract_log_mel_spectrogram(f.name)
            
            # Assertions
            assert spec.shape[0] == 128  # Number of mel bins
            assert spec.shape[1] > 0     # Time frames
            assert np.all(spec <= 0)     # Log mel should be <= 0 dB
```

### Test Coverage

Maintain high test coverage:

```bash
# Run coverage report
uv run pytest --cov=drum_transcription_api --cov-report=html

# View coverage in browser
open htmlcov/index.html
```

### Integration Tests

Test API endpoints:

```python
import pytest
from fastapi.testclient import TestClient
from drum_transcription_api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_transcribe_endpoint():
    """Test transcription endpoint."""
    # Create test audio file
    with open("test_audio.mp3", "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("test.mp3", f, "audio/mpeg")},
            data={"threshold": 0.5}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "midi_file_url" in data
```

## Documentation

### API Documentation

- Update `docs/API_REFERENCE.md` for new endpoints
- Add Pydantic models for request/response validation
- Include examples in docstrings

### Code Documentation

- Use clear, descriptive variable names
- Add inline comments for complex logic
- Document configuration options

### README Updates

Update README.md for:
- New features
- Breaking changes
- Installation instructions
- Usage examples

## Submitting Changes

### Pull Request Template

```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Added new tests for new functionality

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Commit Message Format

Use [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(api): add batch transcription endpoint
fix(transcription): resolve memory leak in audio processing
docs(readme): update installation instructions
test(api): add integration tests for new endpoints
```

## Review Process

### Review Guidelines

1. **Code Review Checklist**:
   - [ ] Code follows style guidelines
   - [ ] Tests are comprehensive
   - [ ] Documentation is updated
   - [ ] No security vulnerabilities
   - [ ] Performance considerations addressed

2. **Review Focus Areas**:
   - Correctness and logic
   - Performance implications
   - Security considerations
   - API design consistency
   - Documentation clarity

### Approval Process

- At least one maintainer approval required
- All CI checks must pass
- Documentation must be updated for API changes
- Breaking changes require maintainer discussion

### Merge Process

1. **Squash and merge** for feature branches
2. **Create release tag** for breaking changes
3. **Update CHANGELOG.md**
4. **Deploy to staging** for final verification

## Getting Help

### Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Librosa Documentation](https://librosa.org/doc/)
- [Project Documentation](docs/)

### Communication

- Create GitHub issues for bugs and feature requests
- Use discussions for questions and ideas
- Join our community channels (if available)

### Mentorship

If you're new to contributing:
- Start with good first issues
- Ask for help in discussions
- Review existing pull requests to learn patterns

Thank you for contributing to the Drum Transcription API! 🎵
