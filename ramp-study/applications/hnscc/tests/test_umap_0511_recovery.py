"""Actual cell_uid schema and strict, single-failure recovery without fitting."""
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
import validate_umap_0511 as v


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='umap_recovery_test_')
        self.root=Path(self.tmp.name).resolve()
        self.a=self.root/'a.csv.gz';self.b=self.root/'b.csv.gz'
        self.frame=pd.DataFrame({'cell_uid':[f'u{i}' for i in range(94488)],'dataset':'test',
                                 'UMAP1':np.arange(94488)/100.,'UMAP2':1.})

    def tearDown(self):self.tmp.cleanup()

    def pair(self,other=None):
        self.frame.to_csv(self.a,index=False);(self.frame if other is None else other).to_csv(self.b,index=False)
        return v.compare_coordinates(self.a,self.b)

    def test_actual_identity_column_exact(self):
        r=self.pair();self.assertTrue(r['coordinates_exact']);self.assertTrue(r['gzip_decompressed_equal'])

    def test_wrong_cell_id_name_rejected(self):
        self.frame=self.frame.rename(columns={'cell_uid':'cell_id'})
        with self.assertRaisesRegex(ValueError,'schema'):self.pair()

    def test_duplicate_identity_rejected(self):
        self.frame.loc[0,'cell_uid']='u1'
        with self.assertRaisesRegex(ValueError,'identity'):self.pair()

    def test_missing_identity_rejected(self):
        self.frame.loc[0,'cell_uid']=None
        with self.assertRaisesRegex(ValueError,'identity'):self.pair()

    def test_row_order_rejected(self):
        with self.assertRaisesRegex(ValueError,'identity'):self.pair(self.frame.iloc[::-1])

    def test_metadata_changed_rejected(self):
        other=self.frame.copy();other.loc[0,'dataset']='different'
        with self.assertRaisesRegex(ValueError,'identity'):self.pair(other)

    def test_coordinate_changes_remain_visible(self):
        other=self.frame.copy();other.loc[0,'UMAP2']=2.
        r=self.pair(other);self.assertFalse(r['coordinates_exact']);self.assertEqual(r['columns']['UMAP2']['max_abs_difference'],1)

    def test_nonfinite_rejected(self):
        self.frame.loc[0,'UMAP2']=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):self.pair()

    def test_unknown_run_request_rejected(self):
        (self.root/'REQUEST.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'Unrecognized'):v.inspect(self.root)

    def test_other_failure_not_whitelisted(self):
        with patch.object(v,'sha',side_effect=[v.REQUEST_SHA,'0'*64]):
            with self.assertRaisesRegex(ValueError,'Unrecognized'):v.inspect(self.root)

    def test_completed_run_not_recovery_target(self):
        (self.root/'COMPLETE.json').write_text('{}')
        with patch.object(v,'sha',side_effect=[v.REQUEST_SHA,v.FAILED_SHA]):
            with self.assertRaisesRegex(ValueError,'Unrecognized'):v.inspect(self.root)


if __name__=='__main__':unittest.main()
