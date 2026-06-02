# Dockerfile — Credit Risk Intelligence Platform
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Create directories for data and models (will be mounted or populated)
RUN mkdir -p /data /app/models /app/logs

# Expose Streamlit port
EXPOSE 8501

# Entrypoint: train if model doesn't exist, then launch UI
CMD ["sh", "-c", "\
  echo '=== NeoStats Credit Risk Platform ===' && \
  if [ -f /app/data/application_train.csv ] && [ ! -f /app/models/model.pkl ]; then \
    echo '>>> Training model...' && python src/ml/train.py; \
  fi && \
  echo '>>> Starting Streamlit...' && \
  streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
"]
