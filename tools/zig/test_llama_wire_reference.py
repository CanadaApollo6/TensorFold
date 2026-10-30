"""The pod reference and served SSE must request the same usage frames."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import llama_draft_wire as wire


class WireReferenceTests(unittest.TestCase):
    def test_reference_requests_served_usage_frame(self):
        fixture = next(c for c in json.loads((wire.ROOT / 'zig/src/server/fixtures/llama/responses.json').read_text())
                       if c['name'] == 'usage')
        actual = [fixture['json_bytes'].encode(), fixture['sse_bytes'].encode()]

        def oracle(*args):
            async def label(case):
                case['request'] = case['extra']
                case['json_bytes'] = fixture['json_bytes']
                frames = fixture['sse_bytes'].split('\n\n')
                if not case['extra'].get('stream_options', {}).get('include_usage'):
                    frames = [f for f in frames if '"usage"' not in f]
                case['sse_bytes'] = '\n\n'.join(frames)
            return label

        def replies(args, drafting):
            self.assertFalse(drafting)
            self.assertEqual(args.reference[1], actual)
            return actual

        hf = SimpleNamespace(decode=lambda *args, **kwargs: 'Plan.</think>answer')
        transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: hf))
        with tempfile.TemporaryDirectory() as tmp:
            ids = Path(tmp) / 'ids.json'
            ids.write_text('[1, 2, 3, 4]')
            argv = ['llama_draft_wire.py', '--model', tmp, '--pod', str(Path(tmp) / 'pod.py'),
                    '--reference-ids', str(ids), '--need-mib', '1', '--out', tmp]
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, transformers=transformers), \
                    patch('llama_serve_reference.pod_functions', return_value={}), \
                    patch('llama_serve_envelopes.oracle', side_effect=oracle), \
                    patch.object(wire, 'replies', side_effect=replies), patch.dict(wire.os.environ):
                wire.main()
            self.assertTrue(json.loads((Path(tmp) / 'summary.json').read_text())['pod_reference_equal'])


if __name__ == '__main__':
    unittest.main()
