FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./
RUN mkdir -p models data

EXPOSE 5000
CMD ["python", "app.py"]