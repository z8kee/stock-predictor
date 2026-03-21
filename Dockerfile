FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

# 1. Keep the root working directory clean
WORKDIR /app

# 2. Copy and install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# 3. Copy the project (This cleanly puts your code at /app/src/app.py)
COPY . .

EXPOSE 5000

# 4. Use Gunicorn's built-in --chdir flag to step into the src folder before running
CMD ["gunicorn", "--workers", "3", "--threads", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "--chdir", "src", "app:app"]