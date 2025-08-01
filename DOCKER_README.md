# Docker Deployment Guide for Bajaj RAG

This guide will help you deploy the Bajaj RAG application using Docker.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose (usually comes with Docker Desktop)
- API keys for the services you want to use

## Quick Start

### 1. Automatic Deployment (Windows)

```bash
# Run the deployment script
deploy.bat
```

### 2. Manual Deployment

```bash
# Build the Docker image
docker build -t bajaj-rag:latest .

# Run with docker-compose
docker-compose up -d
```

## Configuration

### Environment Variables

Create a `.env` file in the project root with your API keys:

```env
# API Keys
MISTRAL_OCR_API_KEY=your_mistral_api_key_here
HUGGING_FACE_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Application Settings
FLASK_ENV=production
PYTHONPATH=/app
```

### Required API Keys

- **Mistral OCR API Key**: For document OCR processing
- **Hugging Face Token**: For downloading models
- **OpenAI API Key**: For LLM completions (optional, can use local models)

## Usage

### Health Check
```bash
curl http://localhost:5000/health
```

### API Endpoint
```bash
curl -X POST http://localhost:5000/hackrx/run \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_token" \
  -d '{
    "documents": ["https://example.com/document.pdf"],
    "questions": ["What is the main topic?"]
  }'
```

## Docker Commands

### Build Image
```bash
docker build -t bajaj-rag:latest .
```

### Run Container
```bash
docker run -p 5000:5000 --env-file .env bajaj-rag:latest
```

### Using Docker Compose
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using port 5000
   netstat -ano | findstr :5000
   ```

2. **Docker build fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   ```

3. **API key issues**
   - Ensure all required API keys are set in `.env`
   - Check API key permissions and quotas

4. **Memory issues**
   - Increase Docker memory limit in Docker Desktop settings
   - Recommended: 8GB+ RAM for ML models

### Logs and Debugging

```bash
# View application logs
docker-compose logs -f bajaj-rag

# Access container shell
docker exec -it bajaj-rag-app bash

# Check container status
docker ps
```

## Production Deployment

### Using Docker Compose with Production Settings

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  bajaj-rag:
    build: .
    restart: always
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data
    deploy:
      resources:
        limits:
          memory: 8G
        reservations:
          memory: 4G
```

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml bajaj-rag
```

## Performance Optimization

### Multi-stage Build (Optional)

For smaller images, you can use a multi-stage build:

```dockerfile
# Build stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

### Resource Limits

Add to your `docker-compose.yml`:

```yaml
services:
  bajaj-rag:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

## Security Considerations

1. **Never commit API keys** to version control
2. **Use secrets management** in production
3. **Run container as non-root user** in production
4. **Enable Docker security scanning**

## Monitoring

### Health Checks
The application includes a health check endpoint at `/health`

### Metrics (Optional)
Add Prometheus metrics endpoint for monitoring:

```python
from prometheus_client import Counter, Histogram
import time

# Add to your Flask app
request_count = Counter('http_requests_total', 'Total HTTP requests')
request_latency = Histogram('http_request_duration_seconds', 'HTTP request latency')

@app.before_request
def before_request():
    request_count.inc()

@app.after_request
def after_request(response):
    request_latency.observe(time.time() - request.start_time)
    return response
```

## Support

For issues and questions:
1. Check the logs: `docker-compose logs -f`
2. Verify environment variables are set correctly
3. Ensure Docker has sufficient resources allocated
4. Check API key permissions and quotas 