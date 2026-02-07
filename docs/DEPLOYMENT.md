# Deployment Guide

## Production Deployment

### Docker Deployment

#### 1. Create Dockerfile

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY drum_transcription_api/ ./drum_transcription_api/

# Install dependencies
RUN uv sync --frozen

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV MODEL_CHECKPOINT="/models/model.ckpt"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start server
CMD ["uv", "run", "uvicorn", "drum_transcription_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Create docker-compose.yml

```yaml
version: '3.8'

services:
  drum-transcription-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_CHECKPOINT=/models/full-training-epoch=99-val_loss=0.0528.ckpt
      - DEVICE=cuda
    volumes:
      - ./models:/models:ro
      - ./temp:/tmp
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - drum-transcription-api
    restart: unless-stopped
```

#### 3. Nginx Configuration

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream drum_api {
        server drum-transcription-api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # File upload size limit
        client_max_body_size 100M;

        location / {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://drum_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout for large files
            proxy_read_timeout 300s;
            proxy_send_timeout 300s;
        }
    }
}
```

### Kubernetes Deployment

#### 1. Deployment Manifest

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: drum-transcription-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: drum-transcription-api
  template:
    metadata:
      labels:
        app: drum-transcription-api
    spec:
      containers:
      - name: api
        image: drum-transcription-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_CHECKPOINT
          value: "/models/model.ckpt"
        - name: DEVICE
          value: "cuda"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-volume
          mountPath: /models
          readOnly: true
        - name: temp-volume
          mountPath: /tmp
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: model-volume
        persistentVolumeClaim:
          claimName: model-pvc
      - name: temp-volume
        emptyDir: {}
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

#### 2. Service Manifest

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: drum-transcription-api-service
spec:
  selector:
    app: drum-transcription-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

#### 3. Ingress Manifest

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: drum-transcription-api-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/rate-limit: "10"
spec:
  tls:
  - hosts:
    - api.your-domain.com
    secretName: api-tls
  rules:
  - host: api.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: drum-transcription-api-service
            port:
              number: 80
```

### Cloud Deployment

#### AWS ECS

```json
{
  "family": "drum-transcription-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "drum-transcription-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/drum-transcription-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MODEL_CHECKPOINT",
          "value": "/models/model.ckpt"
        }
      ],
      "mountPoints": [
        {
          "sourceVolume": "model-volume",
          "containerPath": "/models"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/drum-transcription-api",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ],
  "volumes": [
    {
      "name": "model-volume",
      "efsVolumeConfiguration": {
        "fileSystemId": "fs-12345678",
        "rootDirectory": "/models"
      }
    }
  ]
}
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/PROJECT-ID/drum-transcription-api

gcloud run deploy drum-transcription-api \
  --image gcr.io/PROJECT-ID/drum-transcription-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --gpu 1 \
  --set-env-vars MODEL_CHECKPOINT=/models/model.ckpt \
  --mount-volume name=model-volume,type=cloud-storage,bucket=models-bucket \
  --mount-path /models
```

## Monitoring and Logging

### Prometheus Metrics

Add metrics to the application:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('drum_transcription_requests_total', 'Total requests')
REQUEST_DURATION = Histogram('drum_transcription_duration_seconds', 'Request duration')
TRANSCRIPTION_COUNT = Counter('drum_transcriptions_total', 'Total transcriptions')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    REQUEST_COUNT.inc()
    
    response = await call_next(request)
    
    REQUEST_DURATION.observe(time.time() - start_time)
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    logger.info("transcription_started", filename=file.filename, size=file.size)
    
    try:
        result = await transcribe_file(file)
        logger.info("transcription_completed", 
                   filename=file.filename, 
                   hits=result['total_hits'])
        return result
    except Exception as e:
        logger.error("transcription_failed", 
                    filename=file.filename, 
                    error=str(e))
        raise
```

## Security

### API Key Authentication

```python
from fastapi import Security, HTTPBearer, HTTPException

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    # Implementation
    pass
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(request: Request, file: UploadFile):
    # Implementation
    pass
```

## Performance Optimization

### Caching with Redis

```python
import redis
import pickle

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_spectrogram(audio_hash: str, spectrogram: np.ndarray):
    redis_client.setex(f"spec:{audio_hash}", 3600, pickle.dumps(spectrogram))

def get_cached_spectrogram(audio_hash: str) -> Optional[np.ndarray]:
    data = redis_client.get(f"spec:{audio_hash}")
    return pickle.loads(data) if data else None
```

### Load Balancing

Use multiple instances behind a load balancer:

```yaml
# docker-compose.scale.yml
version: '3.8'
services:
  drum-transcription-api:
    build: .
    scale: 3
    # ... other config
```

## Backup and Recovery

### Model Backup

```bash
#!/bin/bash
# backup_models.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/models_$DATE"

mkdir -p $BACKUP_DIR
cp /models/*.ckpt $BACKUP_DIR/
tar -czf "$BACKUP_DIR.tar.gz" $BACKUP_DIR/
rm -rf $BACKUP_DIR

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR.tar.gz" s3://backup-bucket/models/
```

### Database Backup (if using)

```bash
#!/bin/bash
# backup_db.sh

pg_dump drum_transcription_db > backup_$(date +%Y%m%d).sql
aws s3 cp backup_$(date +%Y%m%d).sql s3://backup-bucket/database/
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Implement request queuing
   - Add memory limits
   - Use streaming for large files

2. **GPU Memory Issues**
   - Reduce batch size
   - Implement model unloading
   - Add memory monitoring

3. **Slow Response Times**
   - Add caching
   - Optimize model loading
   - Use connection pooling

### Health Checks

```python
@app.get("/health/detailed")
async def detailed_health():
    checks = {
        "model": transcriber is not None,
        "gpu": torch.cuda.is_available(),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage('/').percent
    }
    return {
        "status": "healthy" if all(checks.values()) else "unhealthy",
        "checks": checks
    }
```
