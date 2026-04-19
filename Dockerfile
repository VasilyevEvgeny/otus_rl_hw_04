FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-dev \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /workspace/otus_rl_hw_04
ENV HOME=/workspace/otus_rl_hw_04

COPY project/ .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN uv sync

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
