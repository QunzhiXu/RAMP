import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from compare import selected_tables,table

class SelectedTables(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.a=self.root/'a.csv'; self.b=self.root/'b.csv'
    def tearDown(self): self.tmp.cleanup()
    def test_only_fixed_fraction(self):
        pd.DataFrame({'fraction':[.2,.3],'value':[1,2]}).to_csv(self.a,index=False)
        pd.DataFrame({'fraction':[.3],'value':[2]}).to_csv(self.b,index=False)
        a,b,_=selected_tables(self.a,self.b,'group_fraction_scan_20260910/group_counts.csv')
        self.assertTrue(table(a,b)['exact_values']); self.assertEqual(len(a),1)
    def test_other_fraction_rejected(self):
        pd.DataFrame({'fraction':[.2,.3],'value':[1,2]}).to_csv(self.a,index=False)
        pd.DataFrame({'fraction':[.2],'value':[1]}).to_csv(self.b,index=False)
        with self.assertRaises(ValueError): selected_tables(self.a,self.b,'group_fraction_scan_20260910/group_counts.csv')
    def test_fixed_fraction_value_not_ignored(self):
        pd.DataFrame({'fraction':[.2,.3],'value':[1,2]}).to_csv(self.a,index=False)
        pd.DataFrame({'fraction':[.3],'value':[9]}).to_csv(self.b,index=False)
        a,b,_=selected_tables(self.a,self.b,'group_fraction_scan_20260910/group_counts.csv')
        self.assertFalse(table(a,b)['exact_values'])
    def test_display_identity_not_ignored(self):
        pd.DataFrame({'marker_raw':['TIGIT'],'donor':['a'],'association':[.3]}).to_csv(self.a,index=False)
        pd.DataFrame({'marker_raw':['CD39'],'donor':['a'],'association':[.3]}).to_csv(self.b,index=False)
        a,b,_=selected_tables(self.a,self.b,'panel_D/plotted_donor_values.csv')
        self.assertFalse(table(a,b)['exact_values'])
