"""Select real review-7 pilot token prefixes without loading a model or using CUDA."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--length", type=int, default=20000)
    args = p.parse_args()
    prompts = []
    names = []
    with (args.review / "expected.jsonl").open() as f:
        for line in f:
            item = json.loads(line)
            ids = item.get("prompt", [])
            if ids:
                prompts.append(ids)
                names.append(item["name"])
    if len(prompts) != 55 or any(ids[0] != 166100 for ids in prompts):
        raise ValueError("expected 55 review-7 prompts with Nanbeige BOS")
    eligible = [i for i, ids in enumerate(prompts) if len(ids) >= args.length]
    if not eligible:
        raise ValueError("no prompt reaches the requested length")
    chosen = sorted(set([eligible[0], eligible[len(eligible) // 2], eligible[-1]]))
    args.out.mkdir(parents=True, exist_ok=True)
    for folder, length in (("long", args.length), ("short", 128), ("partial", 2048)):
        d = args.out / folder
        d.mkdir(exist_ok=True)
        sequences = [prompts[i][:length] for i in chosen]
        (d / "sequences.json").write_text(json.dumps(sequences) + "\n")
    manifest = {"source": str(args.review / "expected.jsonl"),
                "source_sha256": hashlib.file_digest((args.review / "expected.jsonl").open("rb"), "sha256").hexdigest(),
                "selected": [{"history": i, "name": names[i], "original_tokens": len(prompts[i]),
                              "tested_prefix": args.length} for i in chosen],
                "all_history_lengths": [len(ids) for ids in prompts]}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
