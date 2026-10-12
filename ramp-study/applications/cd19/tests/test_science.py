import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, EXP, MARKERS, STORY


class FrozenScience(unittest.TestCase):
    def test_permanent_mask_hook_all_markers(self):
        path = HERE / 'frozen_20261011/code' / EXP / 'scripts/cd3_nn_joint_validation.py'
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'mask_hook']
        self.assertEqual(len(nodes), 1)
        for j in range(14):
            namespace = {'J': j}
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
            x = torch.randn(9, 28); altered = x.clone(); altered[:, j] = 1e6
            a = namespace['mask_hook'](None, (x,))[0]
            b = namespace['mask_hook'](None, (altered,))[0]
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.equal(a[:, j], torch.zeros(9)))
            self.assertTrue(torch.equal(a[:, j + 14], torch.ones(9)))
            self.assertTrue(torch.equal(x[:, j + 14], altered[:, j + 14]))

    def test_reference_only_targets(self):
        path = HERE / 'frozen_20261011/code' / STORY / 'prediction/masked_marker_model.py'
        tree = ast.parse(path.read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'allowed_targets']
        ns = {'np': np, 'CODEX_ID': 0, 'TRIANA_ID': 1, 'CYTOF_ID': 2,
              'CD19_INDEX': 14, 'NORMAL_MARKERS': MARKERS}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        self.assertEqual(list(ns['allowed_targets'](0)), list(range(14)))
        for platform in (1, 2):
            self.assertEqual(list(ns['allowed_targets'](platform)), list(range(15)))

    def test_dataset_order_is_explicit(self):
        p = HERE / 'frozen_20261011/code' / STORY / 'prediction/masked_marker_model.py'
        nodes = ast.parse(p.read_text()).body
        found = {}
        for n in nodes:
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == 'DATASETS':
                        found[t.id] = ast.literal_eval(n.value)
        self.assertEqual(found['DATASETS'], ('BM CODEX', 'Triana AbSeq', 'BM CyTOF'))

    def test_real_directory_competition(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp).resolve() / 'out'
            code = 'import sys;sys.path.insert(0,sys.argv[1]);from common import reserve;reserve(sys.argv[2],[])'
            processes = [subprocess.Popen([sys.executable, '-B', '-c', code, str(HERE), str(out)], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for _ in range(2)]
            results = []
            for process in processes:
                process.communicate(); results.append(process.returncode)
            self.assertEqual(sorted(results), [0, 1])
            self.assertEqual(list(out.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
