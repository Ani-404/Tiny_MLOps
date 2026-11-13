FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, data, and static assets
COPY app.py train.py data_generator.py ./
COPY data/ data/
COPY static/ static/

# Train the model at build time so the container starts instantly
RUN python train.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
