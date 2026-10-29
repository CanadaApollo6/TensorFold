"""CPU checks for paid-run admission, matrix coverage and interleaved timing contracts."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import llama_draft_full
import llama_gpu_run
import llama_full_gate
import llama_kv_full
import llama_kv_prepare


class Gates(unittest.TestCase):
    def invoke(self, module, args):
        with patch.object(sys, "argv", [module.__name__, *map(str, args)]), contextlib.redirect_stdout(io.StringIO()):
            module.main()

    def test_full_plan_has_every_cell_and_no_gpu_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            histories = d / "histories.json"
            histories.write_text(json.dumps([[166100, i] + [i] * 99998 for i in [1, 2, 3]]))
            args = ["--histories", histories, "--work", d / "work", "--out", d / "out", "--sm", "89", "--plan-only"]
            for name in ["model", "draft", "pod", "smoke", "baseline-runner"]:
                args += ["--" + name, d / name]
            with patch.object(llama_full_gate.subprocess, "run", side_effect=AssertionError("GPU launch in plan")), \
                    patch.object(llama_full_gate.subprocess, "check_output", return_value="test-head\n"):
                self.invoke(llama_full_gate, args)
            result = json.loads((d / "out/summary.json").read_text())
            jobs = result["jobs"]
            self.assertEqual(len(jobs), 22)
            self.assertNotIn("--prepare", jobs[0]["command"])
            self.assertEqual(sum(j["label"].startswith("p4-") for j in jobs), 9)
            self.assertEqual(sum(j["label"].startswith("p3-") for j in jobs), 8)
            self.assertTrue(all(j["state"] == "NOT RUN" for j in jobs))
            for job in jobs:
                cmd = job["command"]
                if cmd[1].endswith("llama_gpu_run.py"):
                    self.assertIn("llama_gpu_run.py", cmd[1])
                    self.assertGreaterEqual(int(cmd[cmd.index("--need-mib") + 1]), 17000)

    def test_prepare_refuses_identical_inputs_then_writes_distinct_short_suffixes(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            source = d / "expected.jsonl"
            records = [dict(name=str(i), prompt=[166100] + [1] * 2100) for i in range(55)]
            source.write_text("\n".join(map(json.dumps, records)))
            args = ["--review", d, "--out", d / "out", "--length", 2000]
            with self.assertRaisesRegex(ValueError, "distinct"):
                self.invoke(llama_kv_prepare, args)
            for i, record in enumerate(records):
                record["prompt"][1600:] = [i] * 501
            source.write_text("\n".join(map(json.dumps, records)))
            self.invoke(llama_kv_prepare, args)
            for folder in ["short", "partial", "long"]:
                seqs = json.loads((d / "out" / folder / "sequences.json").read_text())
                self.assertEqual(len({tuple(s) for s in seqs}), 3)
                self.assertTrue(all(s[0] == 166100 for s in seqs))

    def test_kv_pairs_interleave_and_only_quality_pair_scores_512(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            prompts = d / "prompts.json"
            prompts.write_text(json.dumps([[166100] + [i] * 512 for i in [1, 2, 3]]))
            launches = []
            class Result:
                returncode = 0
                stdout = ""
                stderr = ""
            def run(cmd, **kwargs):
                cmd = list(map(str, cmd))
                result = Result()
                if "llama-capacity" in cmd:
                    result.stderr = "\n".join(f"NEED mode={m} context=641 bytes=1 need_mib=17000" for m in ["bf16", "q4"])
                elif "llama-kv-dump" in cmd:
                    launches.append(cmd)
                    out = Path(cmd[-2])
                    out.mkdir(parents=True)
                    (out / "greedy-0.json").write_text(json.dumps([1] * 128))
                    result.stdout = "QUALITY decode_tok_s=20.0\n"
                elif "llama_kv_compare.py" in cmd[1]:
                    Path(cmd[-1]).write_text('{"passes_distribution_bar":true,"greedy_equal":true}')
                else:
                    raise AssertionError(cmd)
                return result
            with patch.object(llama_kv_full.subprocess, "run", side_effect=run):
                self.invoke(llama_kv_full, ["--model", d, "--prompts", prompts, "--work", d / "work", "--receipts", d / "out"])
            self.assertEqual(len(launches), 10)
            self.assertEqual([c[-3] for c in launches], ["16", "4", "4", "16", "16", "4", "4", "16", "16", "4"])
            self.assertEqual([c[-1] for c in launches], ["512", "512"] + ["0"] * 8)
            self.assertTrue(all("llama_gpu_run.py" in c[1] for c in launches))
            result = json.loads((d / "out/summary.json").read_text())
            self.assertEqual(result["paired_speed_ratio"]["median"], 1)

    def test_guard_stops_only_its_child_when_reserve_is_breached(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            class Child:
                pid = 123
                returncode = None
                terminated = False
                def poll(self):
                    return self.returncode
                def terminate(self):
                    self.terminated = True
                    self.returncode = -15
                def wait(self, **kwargs):
                    return self.returncode
                def kill(self):
                    raise AssertionError("unexpected kill")
            child = Child()
            with patch.object(llama_gpu_run.subprocess, "Popen", return_value=child), \
                    patch.object(llama_gpu_run, "smi", side_effect=["4000", "999, 2000\n123, 900", "1500"]):
                with self.assertRaises(SystemExit) as exit:
                    self.invoke(llama_gpu_run, ["--need-mib", "1000", "--receipt", d / "vram.json", "--", "child"])
            self.assertNotEqual(exit.exception.code, 0)
            self.assertTrue(child.terminated)
            receipt = json.loads((d / "vram.json").read_text())
            self.assertTrue(receipt["reserve_breached"])
            self.assertEqual(receipt["peak_process_vram_mib"], 900)

    def test_p3_refuses_zero_acceptance_despite_proposals(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            prompts = d / "prompts.json"
            prompts.write_text("[[1],[2],[3]]")
            class Result:
                returncode = 0
                stdout = ""
                stderr = "NEED draft=bf16 kv=bf16 bytes=1 need_mib=17000"
            def run(cmd, **kwargs):
                if "llama-draft-full" in list(map(str, cmd)):
                    records = []
                    for prompt in range(3):
                        for seed in [None, *range(32)]:
                            for variant in [0 if seed is None else seed % 4]:
                                for rep in range(5 if seed is None else 1):
                                    side = dict(tokens=[1] * 256, seconds=1, rounds=255, drafted=1000, accepted=0, stopped=False)
                                    records.append(dict(prompt=prompt, seed=seed, variant=variant, rep=rep, plain=side, drafted=side))
                    kwargs["stdout"].write("FORCED full_accept_then_reject\n" * 3)
                    (d / "out/pairs.json").write_text(json.dumps(records))
                return Result()
            with patch.object(llama_draft_full.subprocess, "run", side_effect=run), \
                    patch.object(llama_draft_full.subprocess, "check_output", return_value="49152"):
                with self.assertRaisesRegex(RuntimeError, "mean acceptance"):
                    self.invoke(llama_draft_full, ["--model", d, "--draft", d, "--prompts", prompts,
                                                  "--out", d / "out", "--format", "bf16", "--kv", "16"])


if __name__ == "__main__":
    unittest.main()
