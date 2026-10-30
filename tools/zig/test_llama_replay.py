"""CPU replay must apply penalties to the cached cell's committed history."""
from pathlib import Path
import unittest

from llama_sample_gate import fixtures, replay
from llama_sample_oracle import evaluate


class ReplayTests(unittest.TestCase):
    def test_penalized_conditional_history(self):
        runner = Path(__file__).resolve().parents[2] / 'zig-out/bin/tf-llama-replay'
        self.assertTrue(runner.is_file(), 'build tf-llama-replay before running the CPU regressions')
        cases = [dict(fixtures()[-1], history=history) for history in
                 [[0, 0, 0], [1, 2, 3], [9, 9, 9], [1, 2, 3, 4, 5, 6]]]
        for case, record in zip(cases, replay(runner, cases), strict=True):
            result = evaluate(case, record)
            with self.subTest(history=case['history']):
                self.assertEqual(result['state'], 'PASS', result)


if __name__ == '__main__':
    unittest.main()
