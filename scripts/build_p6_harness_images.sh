#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

docker build \
  --file "$ROOT/configs/p6/docker/requests-harness-v1.Dockerfile" \
  --tag beliefkv/sweb.eval.x86_64.psf_1776_requests-5414:harness-v1 \
  "$ROOT/configs/p6/docker"
