import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import prepare_manifest as p
import validate_manifest as v


class CandidateBinding(unittest.TestCase):
    def setUp(self):
        self.relative = 'runs/full_20261011/group_0000/runtime/results/job_03060/predictions.npz'
        rec = dict(bytes=7, sha256='a'*64)
        self.binding = dict(path=str(p.MODULE/self.relative), relative_to_module=self.relative,
            category='B2', origin='original', group='group_0000', local_job='job_03060',
            task_key=dict(task_type='J2',task='bone_marrow',method='ramp',family='knn',seed=20260919,
                          source='a',target='b',marker='CD117',id='shared-id'),
            payload='predictions.npz', group_lock=dict(path='/locked/group.json',bytes=100,sha256='b'*64),
            sealed_target=rec, canonical=dict(path=str(p.SOURCE/'locked/job/predictions.npz'),**rec))
        self.row = copy.deepcopy(self.binding) | dict(**rec, stat=dict(bytes=7),
            current_bytes_equal_canonical=True,
            proposed_action='unlink_only_after_separate_explicit_approval_and_fresh_checks',
            currently_authorized_action='read_only_review',recovery='canonical_copy_if_hash_still_matches',
            affected_checks=['full verify-run','compare_combined','historical file inventories'])
        self.row['canonical']['stat'] = dict(bytes=7)
        self.expected = {self.relative:self.binding}

    def test_valid_bound_prediction(self):
        v.check_candidate_binding(self.row,self.expected)

    def test_heads_code_cannot_replace_prediction_even_with_equal_hash(self):
        r=copy.deepcopy(self.row)
        r['relative_to_module']='runs/full_20261011/group_0000/runtime/src/trainability/heads.py'
        r['path']=str(p.MODULE/r['relative_to_module'])
        r['canonical']['path']=str(p.SOURCE/'cross_platform_trainability_v2_20260919/src/trainability/heads.py')
        with self.assertRaisesRegex(ValueError,'Candidate not allowed by fixed locks and routes'):
            v.check_candidate_binding(r,self.expected)

    def test_absolute_path_cannot_disagree_with_key(self):
        r=copy.deepcopy(self.row);r['path']=str(p.MODULE/'retained.py')
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_category_cannot_be_relabeled(self):
        r=copy.deepcopy(self.row);r['category']='A_failed'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_origin_cannot_be_relabeled(self):
        r=copy.deepcopy(self.row);r['origin']='continuation'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_group_cannot_be_swapped(self):
        r=copy.deepcopy(self.row);r['group']='group_0850'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_job_cannot_be_swapped(self):
        r=copy.deepcopy(self.row);r['local_job']='job_00001'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_payload_cannot_disagree_with_basename(self):
        r=copy.deepcopy(self.row);r['payload']='ledger.json'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_all_task_identity_fields_are_bound(self):
        for name in self.row['task_key']:
            with self.subTest(field=name):
                r=copy.deepcopy(self.row);r['task_key'][name]='changed'
                with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_unexpected_task_identity_field_rejected(self):
        r=copy.deepcopy(self.row);r['task_key']['injected']=True
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_canonical_path_is_not_self_reported(self):
        r=copy.deepcopy(self.row);r['canonical']['path']=str(p.SOURCE/'other/predictions.npz')
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_collaborative_hash_rewrite_cannot_override_lock(self):
        r=copy.deepcopy(self.row);r['sha256']='c'*64;r['canonical']['sha256']='c'*64;r['sealed_target']['sha256']='c'*64
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_group_lock_is_bound(self):
        r=copy.deepcopy(self.row);r['group_lock']['sha256']='c'*64
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_unknown_candidate_field_rejected(self):
        r=copy.deepcopy(self.row);r['recursive']=True
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_action_cannot_be_changed_to_execution(self):
        r=copy.deepcopy(self.row);r['currently_authorized_action']='delete'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_recovery_and_impact_cannot_be_hidden(self):
        for name,val in [('recovery','no_loss'),('affected_checks',[]),('current_bytes_equal_canonical',False)]:
            with self.subTest(field=name):
                r=copy.deepcopy(self.row);r[name]=val
                with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_stat_size_consistency(self):
        for part in ['target','source']:
            r=copy.deepcopy(self.row)
            (r['stat'] if part=='target' else r['canonical']['stat'])['bytes']=8
            with self.assertRaises(ValueError):v.check_candidate_binding(r,self.expected)

    def test_candidate_set_rejects_missing_extra_duplicate(self):
        v.check_candidate_grid([self.relative], self.expected)
        for values in [[],[self.relative,self.relative],[self.relative,'unknown']]:
            with self.assertRaises(ValueError):v.check_candidate_grid(values,self.expected)

    def test_unsealed_failure_metadata_is_not_falsely_restorable(self):
        b=copy.deepcopy(self.binding)
        b.update(category='A_failed',origin='failed',payload='RESULT.json',sealed_target=None,
                 path=str(p.MODULE/'runs/full_20261011/group_0845/runtime/results/job_1/RESULT.json'),
                 relative_to_module='runs/full_20261011/group_0845/runtime/results/job_1/RESULT.json')
        r=copy.deepcopy(self.row)|b
        r['canonical']=copy.deepcopy(b['canonical'])|dict(stat=dict(bytes=7))
        r.update(sha256='c'*64,current_bytes_equal_canonical=False,
                 recovery='not_byte_restorable_from_canonical_keep_disposition_hash_only',
                 affected_checks=['failed_or_smoke_payload_inventory','historical file inventories'])
        expected={b['relative_to_module']:b}
        v.check_candidate_binding(r,expected)
        r['recovery']='canonical_copy_if_hash_still_matches'
        with self.assertRaises(ValueError):v.check_candidate_binding(r,expected)

    def test_pinned_anchor_cannot_be_resealed_by_caller(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            root=Path(tmp);path=root/'anchor.json';path.write_text('{"changed":true}')
            with patch.object(p,'MODULE',root):
                with self.assertRaisesRegex(ValueError,'Pinned anchor changed'):
                    v.read_pinned('anchor.json','0'*64)


if __name__=='__main__':unittest.main()
