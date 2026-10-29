"""Run one bounded GPU check with a desktop reserve and process VRAM telemetry."""
import argparse
import json
import subprocess
import time
from pathlib import Path


def smi(query, group="gpu"):
    return subprocess.check_output([
        "nvidia-smi", f"--query-{group}={query}", "--format=csv,noheader,nounits", "--id=0"
    ], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--need-mib", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.need_mib <= 0:
        parser.error("a command and positive peak estimate are required")
    free = int(smi("memory.free"))
    if args.need_mib + 2048 > free:
        raise SystemExit(f"BLOCKED need {args.need_mib} MiB plus 2048 MiB desktop reserve; free {free} MiB")
    start = time.monotonic()
    peak = 0
    minimum_free = free
    child = subprocess.Popen(command)
    expired = False
    reserve_breached = False
    try:
        while child.poll() is None:
            for row in smi("pid,used_gpu_memory", "compute-apps").splitlines():
                pid, used = (v.strip() for v in row.split(","))
                if int(pid) == child.pid and used.isdigit():
                    peak = max(peak, int(used))
            minimum_free = min(minimum_free, int(smi("memory.free")))
            if minimum_free < 2048:
                reserve_breached = True
                child.terminate()
                break
            if time.monotonic() - start > args.timeout:
                expired = True
                child.terminate()
                break
            time.sleep(0.2)
        child.wait(timeout=15)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    receipt = {"command": command, "pid": child.pid, "exit_code": child.returncode,
               "seconds": time.monotonic() - start, "timeout": expired,
               "reserve_breached": reserve_breached, "estimate_exceeded": peak > args.need_mib,
               "need_mib": args.need_mib, "initial_free_mib": free, "minimum_free_mib": minimum_free,
               "peak_process_vram_mib": peak, "sampling_ms": 200}
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    raise SystemExit(child.returncode or int(expired or reserve_breached or peak > args.need_mib))


if __name__ == "__main__":
    main()
