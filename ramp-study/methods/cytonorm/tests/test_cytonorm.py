"""CytoNorm-specific source, route, identity and environment contracts."""
import ast
import copy
import unittest
from unittest.mock import patch

from fixtures import io, inv, study
import test_contract


class CytoNormContract(unittest.TestCase):
    def test_story_helper_only_guard_diff(self):
        original=(inv.SNAPSHOT/'adapter/cytonorm_repro.R').read_bytes()
        for base in inv.STORIES.values():
            story=(inv.SNAPSHOT/'study_sources'/base/'scripts/cytonorm_repro.R').read_bytes()
            self.assertEqual(story,original.replace(inv.GUARD_OLD.encode(),inv.GUARD_STORY.encode()))

    def test_tnbc_current_cohort(self):
        s,_=study.plan('tnbc');self.assertEqual(s['cells'],62701)
        self.assertEqual(s['identity_check']['dataset_counts']['Meyer IMC'],22701)

    def test_bm_not_full_panel_clr(self):
        s,_=study.plan('bone_marrow')
        self.assertEqual([r['operator'] for r in s['preprocessing']],['asinh','asinh','log1p'])
        self.assertEqual([r.get('cofactor') for r in s['preprocessing']],[1,5,None])
        self.assertNotIn('CD19',s['markers'])

    def test_lung_not_other_method_scale(self):
        s,_=study.plan('lung_cancer')
        self.assertEqual([r['cofactor'] for r in s['preprocessing']],[5,1,150])

    def test_signed_and_zero_inputs_recorded(self):
        specs={s['case']:s for s in inv.verify_local()['branches']}
        self.assertLess(specs['hnscc_story']['identity_check']['input_ranges']['Haist_CODEX'][0],0)
        self.assertLess(specs['bone_marrow']['identity_check']['input_ranges']['BM CyTOF'][0],0)
        for c,n in [('melanoma',3),('lung_cancer',12),('lung_story',28),('melanoma_story',417)]:
            self.assertEqual(specs[c]['identity_check']['zero_l2_cells'],n)

    def test_lock_and_helper_in_every_bundle(self):
        for s in inv.verify_local()['branches']:
            _,files=study.plan(s['case']);paths={r['path'] for r in files}
            self.assertIn('environment/runtime_lock.csv',paths)
            self.assertIn('adapter/vendor/CytoNorm_2.0.6.tar.gz',paths)
            self.assertEqual('scripts/cytonorm_repro.R' in paths,s['kind']=='story')

    def test_no_fit_command(self):
        tree=ast.parse((inv.HERE/'study.py').read_text())
        names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertNotIn('fit',names)


class AdditionalInputTests(unittest.TestCase):
    setUp=test_contract.Bundle.setUp
    tearDown=test_contract.Bundle.tearDown
    prepare=test_contract.Bundle.prepare

    def test_wrong_preprocessing_scale(self):
        self.prepare();s=copy.deepcopy(self.spec);s['preprocessing'][0]['operator']='identity'
        with self.assertRaisesRegex(ValueError,'scale'):study.validate_inputs(self.bundle,s)

    def test_wrong_seed(self):
        self.prepare();orig=io.read_json
        def read(p):
            value=orig(p)
            if p==self.bundle/'inputs'/self.spec['config']:value['seed']=43
            return value
        with patch.object(io,'read_json',side_effect=read),self.assertRaisesRegex(ValueError,'parameters'):
            study.validate_inputs(self.bundle,self.spec)

    def test_input_marker_permutation(self):
        self.prepare();s=copy.deepcopy(self.spec);s['markers'].reverse()
        with self.assertRaisesRegex(ValueError,'marker order'):study.validate_inputs(self.bundle,s)


if __name__=='__main__':unittest.main()
