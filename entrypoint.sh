#!/bin/sh
set -e
cd /workspace/otus_rl_hw_04 && uv sync
exec "$@"
