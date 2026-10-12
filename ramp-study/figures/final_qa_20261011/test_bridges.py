import unittest
import numpy as np
import pandas as pd

from check_bridges import keyed_check, relative_profile, TOLERANCE


class BridgeContract(unittest.TestCase):
    def setUp(self):
        self.ref = pd.DataFrame({'case':['a','b'],'method':['m','m'],
                                 'value':[.25,.75],'label':['one','two']})

    def check(self, actual, expected=None, rows=2):
        return keyed_check(actual, self.ref if expected is None else expected,
                           ['case','method'],['value','label'],rows)

    def test_exact(self):
        self.assertEqual(self.check(self.ref)['max_abs_difference'],0)

    def test_order_independent_keys(self):
        self.check(self.ref.iloc[::-1])

    def test_one_ulp(self):
        d=self.ref.copy();d.loc[0,'value']=np.nextafter(.25,1)
        self.assertGreater(self.check(d)['max_abs_difference'],0)

    def test_changed_value(self):
        d=self.ref.copy();d.loc[0,'value']+=1e-8
        with self.assertRaisesRegex(ValueError,'Numeric'):self.check(d)

    def test_no_relative_tolerance(self):
        ref=self.ref.copy();ref.loc[0,'value']=1e8
        d=ref.copy();d.loc[0,'value']+=1e-5
        with self.assertRaisesRegex(ValueError,'Numeric'):self.check(d,ref)

    def test_nonfinite(self):
        for value in [np.nan,np.inf,-np.inf]:
            with self.subTest(value=value):
                d=self.ref.copy();d.loc[0,'value']=value
                with self.assertRaisesRegex(ValueError,'Non-finite'):self.check(d)

    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.check(self.ref.iloc[[0,0]])

    def test_missing(self):
        with self.assertRaisesRegex(ValueError,'Row count'):self.check(self.ref.iloc[:1])

    def test_extra(self):
        with self.assertRaisesRegex(ValueError,'Row count'):self.check(pd.concat([self.ref,self.ref.iloc[:1]]))

    def test_wrong_case(self):
        d=self.ref.copy();d.loc[0,'case']='foreign'
        with self.assertRaisesRegex(ValueError,'Identity set'):self.check(d)

    def test_missing_identity(self):
        d=self.ref.copy();d.loc[0,'case']=None
        with self.assertRaisesRegex(ValueError,'Missing identity'):self.check(d)

    def test_missing_value_column(self):
        with self.assertRaisesRegex(ValueError,'Missing columns'):self.check(self.ref.drop(columns='value'))

    def test_text_mismatch(self):
        d=self.ref.copy();d.loc[0,'label']='bad'
        with self.assertRaisesRegex(ValueError,'Text value'):self.check(d)

    def test_expected_duplicates_also_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.check(self.ref,self.ref.iloc[[0,0]])

    def test_no_zero_row_acceptance(self):
        with self.assertRaisesRegex(ValueError,'Invalid expected'):self.check(self.ref.iloc[:0],rows=0)

    def test_relative_profile_both_directions_and_ties(self):
        d=pd.DataFrame([{'benchmark':'a','method':m,'metric':metric,'family':'f','direction':direction,'mean':v}
                        for metric,direction,vals in [('x','higher',[0,2]),('y','lower',[2,0]),('z','higher',[1,1])]
                        for m,v in zip(['m1','m2'],vals)])
        result=relative_profile(d).set_index('method').relative_performance
        self.assertAlmostEqual(result['m1'],1/3)
        self.assertEqual(result['m2'],1)

    def test_relative_unknown_direction(self):
        d=pd.DataFrame([{'benchmark':'a','method':'m','metric':'x','family':'f','direction':'unknown','mean':0}])
        with self.assertRaisesRegex(ValueError,'Unknown metric'):relative_profile(d)


if __name__=='__main__':unittest.main()
