import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import inventory as inv


def race_claim(path, barrier, queue):
    barrier.wait(20)
    try:
        inv.reserve(path)
        queue.put('won')
    except FileExistsError:
        queue.put('refused')


def race_copy(source, destination, expected, barrier, queue):
    barrier.wait(20)
    try:
        inv.copy_new(Path(source), Path(destination), expected)
        queue.put('won')
    except FileExistsError:
        queue.put('refused')


class Safety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='cycombine_test_')
        self.root = Path(self.tmp.name).resolve()
        anchor = patch.object(inv, 'REGISTRY', inv.SNAPSHOT / 'accepted_registry.json')
        anchor.start()
        self.addCleanup(anchor.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self):
        source = self.root / 'source'; source.mkdir()
        payload = source / 'payload'; payload.write_bytes(b'12345')
        info = inv.record(payload)
        manifest = source / 'manifest.json'
        inv.create_json(manifest, {'raw': info['sha256'], 'note': info['sha256']})
        catalog = {'registry_sha256': inv.REGISTRY_SHA, 'cases': [],
                   'files': [{'source': 'payload', 'copy': 'copy/payload', **info},
                             {'source': 'manifest.json', 'copy': 'copy/manifest.json', **inv.record(manifest)}],
                   'links': [{'manifest': 'manifest.json', 'field': ['raw'], 'target': 'payload', 'sha256': info['sha256']}]}
        snap = self.root / 'snapshot'; snap.mkdir()
        for row in catalog['files']:
            inv.copy_new(source / row['source'], snap / row['copy'], row)
        shutil.copyfile(inv.REGISTRY, snap / 'accepted_registry.json')
        inv.create_json(snap / 'catalog.json', catalog)
        inv.create_json(snap / 'COMPLETE.json', {'status': 'complete', 'catalog_sha256': inv.sha(snap / 'catalog.json')})
        return source, snap

    def test_import_has_no_file_write_or_subprocess(self):
        with patch.object(Path, 'open', side_effect=AssertionError('Unexpected open')):
            spec = importlib.util.spec_from_file_location('inventory_import_probe', HERE / 'inventory.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

    def test_reserve_refuses_existing(self):
        inv.reserve(self.root / 'new')
        with self.assertRaises(FileExistsError): inv.reserve(self.root / 'new')

    def test_reserve_refuses_dangling_link(self):
        path = self.root / 'link'; path.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError): inv.reserve(path)

    def test_reserve_refuses_overlap(self):
        source = self.root / 'source'; source.mkdir()
        with self.assertRaises(ValueError): inv.reserve(source / 'new', [source])

    def test_reserve_refuses_parent_of_protected_future_path(self):
        with self.assertRaises(ValueError): inv.reserve(self.root / 'new', [self.root / 'new/file'])

    def test_within_refuses_escape(self):
        with self.assertRaises(ValueError): inv.within(self.root, '../file')
        with self.assertRaises(ValueError): inv.within(self.root, '/tmp/file')

    def test_within_refuses_external_symlink(self):
        parent = self.root / 'directory'; parent.mkdir()
        outside = self.root / 'outside'; outside.write_text('x')
        (parent / 'alias').symlink_to(outside)
        with self.assertRaises(ValueError): inv.within(parent, 'alias')

    def test_exclusive_file_creation(self):
        path = self.root / 'record.json'; inv.create_json(path, {'old': True})
        with self.assertRaises(FileExistsError): inv.create_json(path, {'new': True})
        self.assertEqual(inv.read_json(path), {'old': True})

    def test_copy_checks_bytes(self):
        source = self.root / 'input'; source.write_bytes(b'example')
        with self.assertRaises(ValueError): inv.copy_new(source, self.root / 'output', {'sha256': '0'*64, 'bytes': 7})
        self.assertEqual((self.root / 'output').read_bytes(), b'example')

    def test_copy_rejects_existing_hardlink(self):
        source = self.root / 'input'; source.write_bytes(b'original')
        dest = self.root / 'output'; dest.hardlink_to(source)
        with self.assertRaises(FileExistsError): inv.copy_new(source, dest, inv.record(source))
        self.assertEqual(source.read_bytes(), b'original')

    def test_true_process_claim_race(self):
        ctx = mp.get_context('spawn')
        for index in range(3):
            barrier = ctx.Barrier(2); queue = ctx.Queue()
            workers = [ctx.Process(target=race_claim, args=(str(self.root / str(index)), barrier, queue)) for _ in range(2)]
            for worker in workers: worker.start()
            result = [queue.get(timeout=30) for _ in workers]
            for worker in workers: worker.join(30); self.assertEqual(worker.exitcode, 0)
            self.assertEqual(sorted(result), ['refused', 'won'])

    def test_true_process_copy_race(self):
        ctx = mp.get_context('spawn'); barrier = ctx.Barrier(2); queue = ctx.Queue()
        source = self.root / 'input'; source.write_bytes(b'payload'*10000)
        target = self.root / 'output'
        workers = [ctx.Process(target=race_copy, args=(str(source), str(target), inv.record(source), barrier, queue)) for _ in range(2)]
        for worker in workers: worker.start()
        result = [queue.get(timeout=30) for _ in workers]
        for worker in workers: worker.join(30); self.assertEqual(worker.exitcode, 0)
        self.assertEqual(sorted(result), ['refused', 'won'])
        self.assertEqual(inv.record(source), inv.record(target))

    def test_local_and_source_verification(self):
        source, snapshot = self.fixture()
        self.assertEqual(inv.verify_sources(source, snapshot)['json_hash_links'], 1)

    def test_move_snapshot_works_without_source(self):
        source, snapshot = self.fixture()
        source.rename(self.root / 'unavailable')
        moved = self.root / 'moved'; snapshot.rename(moved)
        self.assertEqual(len(inv.verify_local(moved)['files']), 2)

    def test_wrong_complete_status(self):
        _, snapshot = self.fixture()
        p = snapshot / 'COMPLETE.json'; data = inv.read_json(p); data['status'] = 'partial'; p.write_text(json.dumps(data))
        with self.assertRaises(ValueError): inv.verify_local(snapshot)

    def test_missing_status(self):
        _, snapshot = self.fixture()
        p = snapshot / 'COMPLETE.json'; data = inv.read_json(p); del data['status']; p.write_text(json.dumps(data))
        with self.assertRaises(ValueError): inv.verify_local(snapshot)

    def test_changed_copy(self):
        _, snapshot = self.fixture(); (snapshot / 'copy/payload').write_bytes(b'new')
        with self.assertRaises(ValueError): inv.verify_local(snapshot)

    def test_changed_external_source(self):
        source, snapshot = self.fixture(); (source / 'payload').write_bytes(b'new')
        with self.assertRaises(ValueError): inv.verify_sources(source, snapshot)

    def test_extra_snapshot_file(self):
        _, snapshot = self.fixture(); (snapshot / 'extra').write_text('x')
        with self.assertRaises(ValueError): inv.verify_local(snapshot)

    def test_changed_catalog(self):
        _, snapshot = self.fixture(); (snapshot / 'catalog.json').write_text('{}')
        with self.assertRaises(ValueError): inv.verify_local(snapshot)

    def test_wrong_registry(self):
        _, snapshot = self.fixture(); (snapshot / 'accepted_registry.json').write_text('{}')
        with self.assertRaises(ValueError): inv.verify_local(snapshot)

    def test_precise_field_not_matching_note(self):
        source, snapshot = self.fixture()
        document = inv.read_json(source / 'manifest.json'); document['raw'] = '0'*64
        (source / 'manifest.json').write_text(json.dumps(document))
        # Bypass file identity here to isolate exact-field checking, not substring search.
        with patch.object(inv, 'check'):
            with self.assertRaises(ValueError): inv.verify_sources(source, snapshot)

    def test_historical_absolute_path_rebase(self):
        self.assertEqual(inv.original_relative('/old/host/final_xu/a/b', '.'), Path('a/b'))
        with self.assertRaises(ValueError): inv.original_relative('/arbitrary/host/a', '.')

    def test_failed_freeze_retains_partial_and_refuses_reuse(self):
        source, snapshot = self.fixture()
        catalog = inv.verify_local(snapshot)
        output = self.root / 'freeze'
        original = inv.copy_new
        def interrupted(*args):
            original(*args)
            raise RuntimeError('simulated interruption')
        with patch.object(inv, 'collect', return_value=catalog), patch.object(inv, 'copy_new', side_effect=interrupted):
            with self.assertRaises(RuntimeError): inv.freeze(source, output)
        self.assertTrue((output / 'FAILED.json').exists())
        self.assertFalse((output / 'COMPLETE.json').exists())
        self.assertTrue((output / 'copy/payload').exists())
        with self.assertRaises(FileExistsError): inv.freeze(source, output)

    def test_source_changed_during_freeze_rejected(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        original = inv.copy_new; output = self.root / 'freeze'
        def changed(src, dst, expected):
            original(src, dst, expected)
            if src == source / 'payload': src.write_bytes(b'changed')
        with patch.object(inv, 'collect', return_value=catalog), patch.object(inv, 'copy_new', side_effect=changed):
            with self.assertRaises(ValueError): inv.freeze(source, output)
        self.assertTrue((output / 'FAILED.json').exists())
        self.assertFalse((output / 'COMPLETE.json').exists())

    def isolated_registry(self):
        registry = self.root / 'registry.json'
        shutil.copyfile(inv.REGISTRY, registry)
        return registry

    def assert_failed_without_completion(self, output):
        self.assertTrue((output / 'FAILED.json').is_file())
        self.assertFalse((output / 'COMPLETE.json').exists())
        with self.assertRaises(FileNotFoundError): inv.verify_local(output)

    def test_registry_changed_after_collect_has_no_complete(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        registry = self.isolated_registry(); output = self.root / 'freeze'
        def changed(*args, **kwargs):
            registry.write_text('{}')
            return catalog
        with patch.object(inv, 'REGISTRY', registry), patch.object(inv, 'collect', side_effect=changed):
            with self.assertRaises(ValueError): inv.freeze(source, output)
        self.assert_failed_without_completion(output)

    def test_registry_changed_during_copy_has_no_complete(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        registry = self.isolated_registry(); output = self.root / 'freeze'
        original = inv.copy_new
        def changed(src, dst, expected):
            if src == registry: registry.write_text('{}')
            original(src, dst, expected)
        with patch.object(inv, 'REGISTRY', registry), patch.object(inv, 'collect', return_value=catalog), patch.object(inv, 'copy_new', side_effect=changed):
            with self.assertRaises(ValueError): inv.freeze(source, output)
        self.assert_failed_without_completion(output)

    def test_registry_changed_after_copy_has_no_complete(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        registry = self.isolated_registry(); output = self.root / 'freeze'
        original = inv.copy_new
        def changed(src, dst, expected):
            original(src, dst, expected)
            if src == registry: registry.write_text('{}')
        with patch.object(inv, 'REGISTRY', registry), patch.object(inv, 'collect', return_value=catalog), patch.object(inv, 'copy_new', side_effect=changed):
            with self.assertRaises(ValueError): inv.freeze(source, output)
        self.assert_failed_without_completion(output)

    def test_wrong_registry_before_collect_has_no_complete(self):
        source, _ = self.fixture(); registry = self.isolated_registry()
        registry.write_text('{}'); output = self.root / 'freeze'
        with patch.object(inv, 'REGISTRY', registry), patch.object(inv, 'collect') as collector:
            with self.assertRaises(ValueError): inv.freeze(source, output)
            collector.assert_not_called()
        self.assert_failed_without_completion(output)

    def test_final_payload_failure_has_no_complete(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        output = self.root / 'freeze'; original = inv.create_json
        def corrupt(path, value):
            original(path, value)
            if path == output / 'catalog.json': (output / 'copy/payload').write_bytes(b'tampered')
        with patch.object(inv, 'collect', return_value=catalog), patch.object(inv, 'create_json', side_effect=corrupt):
            with self.assertRaises(ValueError): inv.freeze(source, output)
        self.assert_failed_without_completion(output)

    def test_wrong_catalog_anchor_has_no_complete(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        catalog['registry_sha256'] = '0'*64; output = self.root / 'freeze'
        with patch.object(inv, 'collect', return_value=catalog):
            with self.assertRaises(ValueError): inv.freeze(source, output)
        self.assert_failed_without_completion(output)

    def test_success_marker_is_written_only_after_payload_check(self):
        source, snapshot = self.fixture(); catalog = inv.verify_local(snapshot)
        output = self.root / 'freeze'; original = inv._verify_payload; calls = []
        def checked(path, *, completed):
            self.assertFalse(completed)
            self.assertFalse((output / 'COMPLETE.json').exists())
            result = original(path, completed=completed)
            calls.append('checked')
            return result
        with patch.object(inv, 'collect', return_value=catalog), patch.object(inv, '_verify_payload', side_effect=checked):
            self.assertEqual(inv.freeze(source, output), output)
        self.assertEqual(calls, ['checked'])
        self.assertFalse((output / 'FAILED.json').exists())
        self.assertEqual(inv.verify_local(output), catalog)


class FrozenContract(unittest.TestCase):
    def test_seven_cases(self):
        catalog = inv.verify_local()
        self.assertEqual({s['case'] for s in catalog['cases']}, set(inv.TECHNICAL) | set(inv.STORIES))
        self.assertEqual(sum(s['cells'] for s in catalog['cases']), 614906)

    def test_only_batch_guard_diff(self):
        self.assertTrue(inv.verify_wrappers()['only_batch_count_guard_and_message_differ'])

    def test_all_parameters_locked(self):
        for case in inv.verify_local()['cases']: self.assertEqual(case['parameters'], inv.PARAMETERS)

    def test_no_data_or_environment_package_copy(self):
        for row in inv.verify_local()['files']:
            if 'copy' not in row: continue
            self.assertNotIn(Path(row['copy']).suffix, ('.npy', '.npz', '.rds', '.h5ad'))
            self.assertNotIn('.Rlib', Path(row['copy']).parts)


if __name__ == '__main__': unittest.main()
