@echo off
REM Bajaj RAG Docker Deployment Script for Windows

echo 🚀 Building and deploying Bajaj RAG application...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Create .env file if it doesn't exist
if not exist .env (
    echo 📝 Creating .env file template...
    (
        echo # API Keys
        echo MISTRAL_OCR_API_KEY=your_mistral_api_key_here
        echo HUGGING_FACE_TOKEN=your_huggingface_token_here
        echo OPENAI_API_KEY=your_openai_api_key_here
        echo OPENAI_API_BASE=https://api.openai.com/v1
        echo.
        echo # Application Settings
        echo FLASK_ENV=production
        echo PYTHONPATH=/app
    ) > .env
    echo ⚠️  Please edit .env file with your actual API keys before running the application.
    pause
)

REM Build the Docker image
echo 🔨 Building Docker image...
docker build -t bajaj-rag:latest .

if errorlevel 1 (
    echo ❌ Docker build failed!
    pause
    exit /b 1
)

echo ✅ Docker image built successfully!

REM Run with docker-compose
echo 🚀 Starting application with docker-compose...
docker-compose up -d

echo ✅ Application is starting up!
echo 📊 Health check: http://localhost:5000/health
echo 🔗 API endpoint: http://localhost:5000/hackrx/run
echo.
echo 📋 To view logs: docker-compose logs -f
echo 🛑 To stop: docker-compose down
echo.
pause 