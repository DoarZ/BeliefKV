#!/usr/bin/env bash
set -euo pipefail

NETWORK_NAME="${QWEN_PROXY_NETWORK_NAME:-qwen-code-sandbox-proxy}"
SUBNET="${QWEN_PROXY_SUBNET:-172.20.0.0/16}"
GATEWAY="${QWEN_PROXY_GATEWAY:-172.20.0.1}"

if docker network inspect "${NETWORK_NAME}" >/dev/null 2>&1; then
  actual_gateway="$(
    docker network inspect \
      -f '{{(index .IPAM.Config 0).Gateway}}' "${NETWORK_NAME}"
  )"
  actual_internal="$(
    docker network inspect -f '{{.Internal}}' "${NETWORK_NAME}"
  )"
  if [[ "${actual_gateway}" != "${GATEWAY}" || "${actual_internal}" != "false" ]]; then
    printf 'Unexpected network configuration for %s: gateway=%s internal=%s\n' \
      "${NETWORK_NAME}" "${actual_gateway}" "${actual_internal}" >&2
    exit 2
  fi
else
  docker network create \
    --driver bridge \
    --subnet "${SUBNET}" \
    --gateway "${GATEWAY}" \
    "${NETWORK_NAME}" >/dev/null
fi

printf '%s\n' "${GATEWAY}"
