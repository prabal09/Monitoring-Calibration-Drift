# syntax=docker/dockerfile:1
FROM python:3.11-slim-bookworm

ARG REPO_URL
ARG REPO_REF=main

# System libs for Open3D / OpenCV, plus git + openssh-client so the image can
# clone over SSH at build time and the user can `git pull` from inside the
# running container.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libgomp1 \
        libglib2.0-0 \
        libusb-1.0-0 \
        git \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Trust github.com's host key so SSH clone does not hang on the unknown-host
# prompt during a non-interactive build.
RUN mkdir -p -m 0700 ~/.ssh && \
    ssh-keyscan -t rsa,ed25519 github.com >> ~/.ssh/known_hosts

# Clone via SSH using BuildKit's --ssh forwarding. The key stays on the host;
# only the agent socket is mounted into the build. REPO_URL must be in SSH
# form, e.g. git@github.com:<user>/<repo>.git
RUN --mount=type=ssh git clone --branch ${REPO_REF} ${REPO_URL} .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

CMD ["bash"]
