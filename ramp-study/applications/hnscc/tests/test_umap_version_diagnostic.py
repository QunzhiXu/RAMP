"""No scientific fitting: identity, binding, status and runtime regression checks."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import umap_version_diagnostic as d


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='hnscc_umap_diagnostic_test_')
        self.root = Path(self.tmp.name)
        self.a, self.b = self.root/'a.csv.gz', self.root/'b.csv.gz'
        self.frame = pd.DataFrame({'cell_id':[f'c{i}' for i in range(94488)],
                                   'dataset':'test', 'UMAP1':np.arange(94488)/100., 'UMAP2':0.})

    def tearDown(self): self.tmp.cleanup()

    def pair(self, other=None):
        self.frame.to_csv(self.a,index=False)
        (self.frame if other is None else other).to_csv(self.b,index=False)
        return d.comparison(self.a,self.b)

    def test_exact_coordinates_and_decompressed_content(self):
        r=self.pair(); self.assertTrue(r['coordinates_exact']); self.assertTrue(r['gzip_decompressed_equal'])
        self.assertEqual(r['columns']['UMAP1']['max_abs_difference'],0)

    def test_coordinate_difference_reported_not_hidden(self):
        other=self.frame.copy(); other.loc[0,'UMAP1']=.01
        r=self.pair(other); self.assertFalse(r['coordinates_exact'])
        self.assertEqual(r['columns']['UMAP1']['changed_values'],1)

    def test_row_order_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'): self.pair(self.frame.iloc[::-1])

    def test_changed_identity_rejected(self):
        other=self.frame.copy(); other.loc[0,'cell_id']='different'
        with self.assertRaisesRegex(ValueError,'identity'): self.pair(other)

    def test_duplicate_identity_rejected(self):
        self.frame.loc[0,'cell_id']='c1'
        with self.assertRaisesRegex(ValueError,'identity'): self.pair()

    def test_missing_row_rejected(self):
        with self.assertRaisesRegex(ValueError,'cohort'): self.pair(self.frame.iloc[:-1])

    def test_missing_coordinate_rejected(self):
        with self.assertRaisesRegex(ValueError,'schema'): self.pair(self.frame.drop(columns=['UMAP1']))

    def test_nonfinite_rejected(self):
        self.frame.loc[0,'UMAP1']=float('inf')
        with self.assertRaisesRegex(ValueError,'Nonfinite'): self.pair()

    def test_bound_file_change_rejected(self):
        path=self.root/'source'; path.write_text('before'); binding={'source':d.bound(path)}
        d.check_bindings(binding); path.write_text('after')
        with self.assertRaisesRegex(ValueError,'Bound source changed'): d.check_bindings(binding)

    def test_incomplete_rejected(self):
        with self.assertRaisesRegex(ValueError,'incomplete'): d.verify(self.root)

    def test_failed_rejected(self):
        (self.root/'FAILED.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'failed'): d.verify(self.root)

    def test_bad_completion_hash_rejected(self):
        (self.root/'MANIFEST.json').write_text('{}')
        (self.root/'COMPLETE.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'seal'): d.verify(self.root)

    def test_unknown_controller_rejected(self):
        (self.root/'REQUEST.json').write_text(json.dumps({'controller':{},'parameters':d.PARAMETERS}))
        with self.assertRaisesRegex(ValueError,'controller'): d.check_payload(self.root)

    def test_changed_seed_rejected(self):
        params={**d.PARAMETERS,'random_state':42}
        (self.root/'REQUEST.json').write_text(json.dumps({'controller':d.controller(),'parameters':params}))
        with self.assertRaisesRegex(ValueError,'parameters'): d.check_payload(self.root)

    def test_wrong_dependency_rejected(self):
        with patch.object(d.metadata,'version',return_value='unexpected'):
            with self.assertRaisesRegex(ValueError,'dependency'): d.probe('0.5.11')

    def test_existing_output_rejected(self):
        with self.assertRaises(FileExistsError): d.reserve(self.root,[])

    def test_protected_output_rejected_before_creation(self):
        out=self.root/'protected'/'new'; out.parent.mkdir()
        with self.assertRaisesRegex(ValueError,'overlaps'): d.reserve(out,[out.parent])
        self.assertFalse(out.exists())


if __name__=='__main__': unittest.main()
