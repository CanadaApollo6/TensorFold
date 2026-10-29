#!/bin/bash
set -euo pipefail
if [[ ${LLAMA_JOB_BOUNDED:-0} != 1 ]]; then
    (( ${JOB_SECONDS:-12600} > 20 && ${JOB_SECONDS:-12600} <= 12600 )) || exit 2
    export GATE_STARTED=$(date +%s)
    export GATE_DEADLINE=$((GATE_STARTED + ${JOB_SECONDS:-12600}))
    export LLAMA_JOB_BOUNDED=1
    exec timeout --signal=TERM --kill-after=20 "$(( ${JOB_SECONDS:-12600} - 20 ))" bash "$0" "$@"
fi
INPUTS=${INPUTS:-/inputs}
W=${W:-/workspace}
OUT_MOUNT=${OUT_MOUNT:-/out}
OUT=${OUT:-$OUT_MOUNT/receipts}
WORK=${WORK:-$W/work}
mkdir -p "$W" "$OUT" "$WORK"
exec >> "$OUT/container.log" 2>&1
printf 'CONTAINER_START %s DEADLINE %s\n' "$GATE_STARTED" "$GATE_DEADLINE"
HF=("$W/exl3/bin/hf")
SYNC_PID=
FAILED=$W/sync-failed
rm -f "$FAILED"
run_sync() {
    local limit=$((GATE_DEADLINE - $(date +%s) - 25))
    (( limit > 0 )) || return 1
    timeout --kill-after=5 "$limit" "$@"
}
sync_out() {
    local snapshot rc=0
    snapshot=$(mktemp -d "$W/snapshot.XXXXXX") || return 1
    run_sync cp -a "$OUT/." "$snapshot/" || rc=1
    if [[ ${LOCAL_DRY:-0} == 1 ]]; then
        [[ ${FAIL_SYNC:-} != copy ]] || rc=1
        [[ ${FAIL_SYNC:-} != upload ]] || rc=1
    else
        run_sync "${HF[@]}" buckets sync "$WORK" "$OUT_BUCKET/work" || rc=1
        if [[ $rc != 0 ]]; then echo FAIL > "$OUT/status.txt"; cp "$OUT/status.txt" "$snapshot/status.txt" || rc=1; fi
        run_sync "${HF[@]}" buckets sync "$snapshot" "$OUT_BUCKET/receipts" || rc=1
    fi
    if [[ $rc != 0 ]]; then
        echo FAIL > "$OUT/status.txt"
        echo FAIL > "$snapshot/status.txt"
    fi
    run_sync rm -rf "$snapshot" || rc=1
    return "$rc"
}
finish() {
    local rc=$? remaining
    trap - EXIT INT TERM
    if [[ -n $SYNC_PID ]]; then
        kill "$SYNC_PID" 2>/dev/null || true
        wait "$SYNC_PID" 2>/dev/null || true
    fi
    [[ ! -f $FAILED ]] || rc=1
    remaining=$((GATE_DEADLINE - $(date +%s) - 20))
    (( remaining > 0 )) || rc=1
    printf '%s\n' "$([[ $rc == 0 ]] && echo PASS || echo FAIL)" > "$OUT/status.txt"
    if (( remaining > 0 )); then
        # The final copy is bounded too; the mounted status remains durable if the upload fails.
        sync_out || rc=1
    fi
    if [[ $rc != 0 ]]; then echo FAIL > "$OUT/status.txt"; fi
    printf 'CONTAINER_END exit=%s\n' "$rc"
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ ${LOCAL_DRY:-0} == 1 ]]; then
    [[ ${DRY:-0} == 1 ]] || exit 2
    [[ ${FAIL_WORK:-0} == 0 ]] || { echo WORK_FAILED; exit 23; }
    echo CPU_DRY_PASS
    exit 0
fi
: "${ENGINE_COMMIT:?set the reviewed engine commit} ${OUT_BUCKET:?set the output bucket}"
mountpoint -q "$OUT_MOUNT" || { echo 'A writable output bucket must be mounted from container start'; exit 1; }
(cd "$INPUTS" && sha256sum -c SHA256SUMS)
mapfile -d '' -t debs < <(find "$INPUTS/debs" -type f -name '*.deb' -print0)
(( ${#debs[@]} > 0 )) || exit 1
for file in "${debs[@]}" "$INPUTS"/bootstrap.tar.xz "$INPUTS"/image.txt "$INPUTS"/llama-gate.lock "$INPUTS"/llama-hf.lock; do
    actual=$(cd "$INPUTS" && sha256sum "${file#"$INPUTS"/}")
    grep -Fxq "$actual" "$INPUTS/SHA256SUMS" || { echo 'Unpinned setup input'; exit 1; }
done
[[ $(cat "$INPUTS/image.txt") == "${CONTAINER_IMAGE:?supply the pinned image digest}" ]]
[[ $CONTAINER_IMAGE == *@sha256:* ]]
export DEBIAN_FRONTEND=noninteractive
dpkg --install "${debs[@]}"
mkdir -p "$W/bootstrap"
tar -xf "$INPUTS/bootstrap.tar.xz" -C "$W/bootstrap"
UV=$W/bootstrap/uv
PYTHON=$W/bootstrap/python/bin/python3.12
"$UV" --version
"$PYTHON" --version
"$UV" venv "$W/exl3" --python "$PYTHON"
"$UV" pip sync --python "$W/exl3/bin/python" --require-hashes "$INPUTS/llama-gate.lock"
"$UV" venv "$W/hf" --python "$PYTHON"
"$UV" pip sync --python "$W/hf/bin/python" --require-hashes "$INPUTS/llama-hf.lock"
HF=("$W/hf/bin/hf")
if [[ ${RESUME:-0} == 1 ]]; then
    "${HF[@]}" buckets sync "$OUT_BUCKET/receipts" "$W/prior"
    cp "$OUT/container.log" "$W/startup.log"
    cp -a "$W/prior/." "$OUT/"
    cat "$W/startup.log" >> "$OUT/container.log"
    "${HF[@]}" buckets sync "$OUT_BUCKET/work" "$WORK"
fi
sync_out || { touch "$FAILED"; exit 1; }
(while sleep 60; do sync_out || { touch "$FAILED"; exit 1; }; done) &
SYNC_PID=$!
if [[ ${DRY:-0} == 1 ]]; then
    echo CPU_DRY_PASS
    exit 0
fi
mkdir -p "$W/zig"
tar -xf "$INPUTS/zig-0.17.tar.xz" -C "$W/zig" --strip-components=1
ZIG=$W/zig/zig
"$ZIG" version > "$OUT/zig-version.txt"
[[ $("$ZIG" version) == 0.17.* ]]
[[ -d $W/engine ]] || git clone "$INPUTS/engine.bundle" "$W/engine"
git -C "$W/engine" checkout --detach "$ENGINE_COMMIT"
SM=${SM:-89}
(cd "$W/engine" && "$ZIG" build native tf-llama-test tf-llama-wire-test tf-llama-replay \
    -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm="$SM" -j4 --summary all) > "$OUT/build-engine.log" 2>&1
cd "$W/engine"
export EXL3_INT8_GEMV=0
"$W/exl3/bin/python" tools/zig/llama_full_gate.py --model "$INPUTS/model" --draft "$INPUTS/draft" \
    --histories "$INPUTS/histories.json" --pod "$INPUTS/serve_exl3.py" --smoke "$INPUTS/smoke_serve.py" \
    --baseline-runner "$W/engine/zig-out/bin/tf-llama-test" --reference-python "$W/exl3/bin/python" \
    --sm "$SM" --work "$WORK" --out "$OUT" --budget-minutes 210 --deadline "$GATE_DEADLINE" --timeout "${GATE_JOB_TIMEOUT:-1800}" \
    "$@"
