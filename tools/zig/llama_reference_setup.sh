#!/usr/bin/env bash
set -euo pipefail
exl_env=${1:-$HOME/exl3-local}
uv venv "$exl_env" --python 3.12
uv pip install --python "$exl_env/bin/python" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$exl_env/bin/python" \
  'https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/exllamav3-1.6.5%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl' \
  transformers tabulate pyyaml safetensors
"$exl_env/bin/python" -c 'import importlib.metadata as m; print("torch", m.version("torch"), "exllamav3", m.version("exllamav3"))'
