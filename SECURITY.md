# Security Policy

## Supported Versions

| Version | Supported          | Security Updates |
|---------|--------------------|------------------|
| 0.1.x   | :white_check_mark: | :white_check_mark: |
| < 0.1   | :x:                | :x:               |

## Reporting a Vulnerability

### Private Disclosure

We take security seriously and appreciate your efforts to responsibly disclose vulnerabilities. 

**Please do NOT open a public issue for security vulnerabilities.**

Instead, please send an email to: **security@your-domain.com**

Include the following information in your report:
- Type of vulnerability
- Steps to reproduce the issue
- Potential impact
- Any proof-of-concept code or screenshots
- Your name (optional, for credit in acknowledgments)

### Response Timeline

- **Initial response**: Within 48 hours
- **Detailed assessment**: Within 7 days
- **Patch release**: Within 14 days (depending on complexity)
- **Public disclosure**: After patch is released (or with coordinated disclosure)

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report within 48 hours
2. **Validation**: We'll validate and reproduce the vulnerability
3. **Assessment**: We'll determine the severity and potential impact
4. **Remediation**: We'll develop and test a fix
5. **Release**: We'll release a security patch
6. **Credit**: We'll credit you in the release notes (with your permission)

## Security Features

### Input Validation

The API implements comprehensive input validation:

```python
# File type validation
allowed_types = ['audio/mpeg', 'audio/wav', 'audio/x-wav', 'audio/mp3', 'audio/m4a', 'audio/flac']

# Parameter validation with Pydantic
class TranscriptionRequest(BaseModel):
    threshold: float = Field(ge=0.1, le=1.0)
    min_interval: float = Field(ge=0.01, le=0.5)
    tempo: int = Field(ge=60, le=200)
```

### File Upload Security

- **File size limits**: Maximum 50MB (configurable)
- **File type validation**: MIME type checking
- **Temporary file handling**: Automatic cleanup
- **Path traversal prevention**: Secure file naming

### API Security

#### Authentication (Optional)

For production deployments, implement API key authentication:

```python
from fastapi import Security, HTTPBearer, HTTPException

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    api_key = os.getenv("API_KEY")
    if credentials.credentials != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials
```

#### Rate Limiting

Implement rate limiting to prevent abuse:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(request: Request, file: UploadFile):
    # Implementation
    pass
```

#### CORS Configuration

Configure CORS properly for production:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Be specific!
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

## Security Best Practices

### Deployment Security

#### Docker Security

```dockerfile
# Use non-root user
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Minimal base image
FROM python:3.11-slim

# Remove unnecessary packages
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
```

#### Environment Variables

Never commit sensitive data:

```bash
# .env (never commit)
API_KEY=your-secret-key
MODEL_CHECKPOINT=/secure/path/to/model.ckpt
```

#### SSL/TLS

Always use HTTPS in production:

```nginx
server {
    listen 443 ssl http2;
    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;
    
    # Strong SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;
}
```

### Code Security

#### Dependency Management

Regularly update dependencies:

```bash
# Check for vulnerabilities
uv pip-audit

# Update dependencies
uv sync --upgrade
```

#### Input Sanitization

Always validate and sanitize inputs:

```python
import os
from pathlib import Path

def safe_filename(filename: str) -> str:
    """Generate safe filename from user input."""
    # Remove path components
    name = Path(filename).name
    
    # Remove dangerous characters
    safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
    safe_name = ''.join(c for c in name if c in safe_chars)
    
    # Limit length
    return safe_name[:100] or "upload"
```

#### Error Handling

Don't expose sensitive information in errors:

```python
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
```

### Infrastructure Security

#### Network Security

- Use firewalls to restrict access
- Implement VPN for administrative access
- Monitor network traffic for anomalies

#### Access Control

- Principle of least privilege
- Regular access reviews
- Multi-factor authentication for admin access

#### Monitoring and Logging

```python
import structlog

logger = structlog.get_logger()

@app.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile):
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    
    logger.info(
        "transcription_request",
        client_ip=client_ip,
        user_agent=user_agent,
        filename=file.filename,
        file_size=file.size
    )
```

## Vulnerability Management

### Common Vulnerabilities

#### 1. File Upload Vulnerabilities

**Risk**: Malicious file uploads, path traversal

**Mitigation**:
- Validate file types and sizes
- Use secure file naming
- Store uploads outside web root
- Scan uploaded files for malware

#### 2. Denial of Service (DoS)

**Risk**: Resource exhaustion through large requests

**Mitigation**:
- Implement rate limiting
- Set request size limits
- Use connection timeouts
- Monitor resource usage

#### 3. Injection Attacks

**Risk**: Command injection through file processing

**Mitigation**:
- Use parameterized queries
- Validate all inputs
- Avoid shell commands with user input
- Use subprocess with proper arguments

#### 4. Information Disclosure

**Risk**: Exposing sensitive system information

**Mitigation**:
- Sanitize error messages
- Remove debug information in production
- Implement proper logging levels
- Secure configuration files

### Security Testing

#### Automated Testing

```python
import pytest
from fastapi.testclient import TestClient

def test_file_upload_validation(client: TestClient):
    """Test file upload security controls."""
    
    # Test oversized file
    large_file = b"x" * (100 * 1024 * 1024)  # 100MB
    response = client.post(
        "/transcribe",
        files={"file": ("large.mp3", large_file, "audio/mpeg")}
    )
    assert response.status_code == 413
    
    # Test malicious file type
    malicious_file = b"<?php system($_GET['cmd']); ?>"
    response = client.post(
        "/transcribe",
        files={"file": ("malicious.php", malicious_file, "application/x-php")}
    )
    assert response.status_code == 400

def test_rate_limiting(client: TestClient):
    """Test rate limiting functionality."""
    responses = []
    for _ in range(15):  # Exceed rate limit of 10/minute
        response = client.get("/health")
        responses.append(response.status_code)
    
    assert any(status == 429 for status in responses)
```

#### Security Scanning

```bash
# Dependency vulnerability scan
uv pip-audit

# Container security scan
docker scan drum-transcription-api

# Web application security scan
# (use tools like OWASP ZAP or Burp Suite)
```

## Incident Response

### Security Incident Process

1. **Detection**: Monitor logs and alerts for suspicious activity
2. **Assessment**: Evaluate the scope and impact
3. **Containment**: Isolate affected systems
4. **Eradication**: Remove the threat
5. **Recovery**: Restore services
6. **Lessons Learned**: Update security measures

### Contact Information

For security matters:
- **Email**: security@your-domain.com
- **PGP Key**: Available on request
- **Response Time**: Within 48 hours

## Security Updates

### Patch Management

- Regular security updates
- Vulnerability scanning
- Dependency updates
- Security advisories

### Changelog

Security updates will be documented in the changelog with:

```markdown
### Security
- Fixed file upload vulnerability (CVE-XXXX-XXXX)
- Updated dependencies to address security issues
- Added rate limiting to prevent DoS attacks
```

## Compliance

### Data Protection

- No personal data stored permanently
- Temporary files automatically cleaned
- GDPR compliance considerations
- Data retention policies

### Industry Standards

- OWASP Top 10 compliance
- Security best practices
- Regular security assessments
- Third-party security audits

---

Thank you for helping keep the Drum Transcription API secure! 🔒
