# Tri-Net v2 — CPU image for the CLI and inference.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e .

# Mount trained weights at runtime: -v $(pwd)/outputs:/app/outputs
COPY . .

ENTRYPOINT ["trinet"]
CMD ["--help"]
