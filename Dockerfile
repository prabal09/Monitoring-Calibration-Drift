FROM python:3.11-slim-bookworm

# System libs required by Open3D and image processing wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libusb-1.0-0 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy only what's needed to resolve and install deps, so the
# expensive `pip install` layer is cached unless pyproject changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --upgrade pip && \
    pip install -e ".[dev]"

# Project source is bind-mounted at /workspace by the dev container,
# so edits on the host are live inside the container.
CMD ["bash"]
