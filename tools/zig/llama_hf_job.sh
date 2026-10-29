#!/bin/bash
set -euo pipefail
: "${ENGINE_COMMIT:?set the reviewed engine commit} ${BASELINE_COMMIT:?set the P1 baseline commit}"
: "${OUT_BUCKET:?hf://buckets/namespace/output/run}"
START_SECONDS=$SECONDS
INPUTS=${INPUTS:-/inputs}
W=${W:-/workspace}
OUT=${OUT:-$W/receipts}
WORK=${WORK:-$W/work}
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$W" "$OUT" "$WORK"
exec > >(tee -a "$OUT/container.log") 2>&1
apt-get update -qq
apt-get install -y -qq git curl ca-certificates build-essential python3 python3-venv xz-utils
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv "$W/exl3" --python 3.12
uv pip install --python "$W/exl3/bin/python" torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$W/exl3/bin/python" numpy transformers==4.57.1 sentencepiece safetensors \
    'huggingface_hub[hf_xet]' fastapi uvicorn regex pyyaml \
    'https://github.com/turboderp-org/exllamav3/releases/download/v1.6.5/exllamav3-1.6.5%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl'
HF=(uvx --from huggingface_hub hf)
if [[ ${RESUME:-0} == 1 ]]; then
    "${HF[@]}" buckets sync "$OUT_BUCKET/receipts" "$OUT"
    "${HF[@]}" buckets sync "$OUT_BUCKET/work" "$WORK"
fi
sync_out() {
    rm -rf "$W/snapshot"
    cp -a "$OUT" "$W/snapshot"
    "${HF[@]}" buckets sync "$WORK" "$OUT_BUCKET/work"
    "${HF[@]}" buckets sync "$W/snapshot" "$OUT_BUCKET/receipts"
}
(while sleep 600; do sync_out || echo 'Periodic sync failed; final sync will retry'; done) &
SYNC_PID=$!
finish() {
    rc=$?
    trap - EXIT INT TERM
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
    sync_out || rc=1
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
(cd "$INPUTS" && sha256sum -c SHA256SUMS)
mkdir -p "$W/zig"
tar -xf "$INPUTS/zig-0.17.tar.xz" -C "$W/zig" --strip-components=1
ZIG=$W/zig/zig
"$ZIG" version | tee "$OUT/zig-version.txt"
[[ $("$ZIG" version) == 0.17.* ]]
[[ -d $W/engine ]] || git clone "$INPUTS/engine.bundle" "$W/engine"
[[ -d $W/baseline ]] || git clone "$INPUTS/baseline.bundle" "$W/baseline"
git -C "$W/engine" checkout --detach "$ENGINE_COMMIT"
git -C "$W/baseline" checkout --detach "$BASELINE_COMMIT"
SM=${SM:-89}
for source in baseline engine; do
    targets=(native tf-llama-test)
    [[ $source != engine ]] || targets+=(tf-llama-wire-test)
    (cd "$W/$source" && "$ZIG" build "${targets[@]}" \
        -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm="$SM" -j4 --summary all) > "$OUT/build-$source.log" 2>&1
done
cd "$W/engine"
GATE_MINUTES=$(( (9000 - (SECONDS - START_SECONDS) - 120) / 60 ))
(( GATE_MINUTES > 0 )) || { echo 'Setup exhausted the 150 minute container budget'; exit 1; }
export EXL3_INT8_GEMV=0
"$W/exl3/bin/python" tools/zig/llama_full_gate.py --model "$INPUTS/model" --draft "$INPUTS/draft" \
    --histories "$INPUTS/histories.json" --pod "$INPUTS/serve_exl3.py" --smoke "$INPUTS/smoke_serve.py" \
    --baseline-runner "$W/baseline/zig-out/bin/tf-llama-test" --reference-python "$W/exl3/bin/python" \
    --sm "$SM" --work "$WORK" --out "$OUT" --budget-minutes "$GATE_MINUTES" --timeout 1800 \
    "$@"
