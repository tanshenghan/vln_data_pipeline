#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="${VLN_UV:-.venv/bin/uv}"
isaac_env="${VLN_ISAAC_ENV:-data_pipeline/.venv/isaac}"
if [[ ! -x "$isaac_env/bin/python" ]]; then
  "$uv_bin" venv --python 3.11 "$isaac_env"
fi
"$uv_bin" pip install --python "$isaac_env/bin/python" \
  'isaacsim[all,extscache]==5.1.0' 'websockets>=14,<16' --extra-index-url https://pypi.nvidia.com
# Installation alone does not mark the workstation ready. A real capture,
# annotation, LeRobot export and independent validation must pass first.
