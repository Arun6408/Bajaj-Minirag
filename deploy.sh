#!/bin/bash

# Bajaj RAG Docker Deployment Script

echo "🚀 Building and deploying Bajaj RAG application..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file template..."
    cat > .env << EOF
# API Keys
MISTRAL_OCR_API_KEY=your_mistral_api_key_here
HUGGING_FACE_TOKEN=your_huggingface_token_here
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1

# Application Settings
FLASK_ENV=production
PYTHONPATH=/app
EOF
    echo "⚠️  Please edit .env file with your actual API keys before running the application."
fi

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t bajaj-rag:latest .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully!"
    
    # Run with docker-compose
    echo "🚀 Starting application with docker-compose..."
    docker-compose up -d
    
    echo "✅ Application is starting up!"
    echo "📊 Health check: http://localhost:5000/health"
    echo "🔗 API endpoint: http://localhost:5000/hackrx/run"
    echo ""
    echo "📋 To view logs: docker-compose logs -f"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Docker build failed!"
    exit 1
fi 