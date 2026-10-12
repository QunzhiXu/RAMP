"""Read-only historical-input reconstruction / isolated runtime diagnosis.

Never writes original files, overwrites outputs, or changes canonical RAMP.
Diagnostic fits are not accepted study runs and have no COMPLETE marker.
"""
import argparse
import ast
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import inventory as inv
import safe_io as io
import study
import worker


def reconstructed_input(spec, source):
    script = source / spec['source_script']
    expected = inv.catalog()['sources'][spec['source_script']]
    io.check(script, expected)
    tree = ast.parse(script.read_text())
    names = ('build_moldoveanu', 'build_engelbrecht')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if [n.name for n in functions] != list(names):
        raise ValueError('Historical builder functions changed')
    manifest = io.read(source / spec['historical_manifest'])
    records = {}
    paths = {}
    relocated = {
        'moldoveanu_shared13_input': source / 'figure_melanoma/inputs/cohort_package/moldoveanu_imc/alignment_input.csv.gz',
        'engelbrecht_shared13_input': source / 'figure_melanoma/inputs/cohort_package/engelbrecht_citeseq/alignment_input.csv.gz',
    }
    for name, value in manifest['sources'].items():
        if name.endswith('_sha256'):
            continue
        p = relocated.get(name, Path(value))
        # Paths here must be the existing original files, not arbitrary overrides.
        if not p.resolve(strict=True).is_relative_to(source):
            raise ValueError('Source escaped original tree')
        record = io.record(p)
        if record['sha256'] != manifest['sources'][name + '_sha256']:
            raise ValueError('Provider source hash differs: ' + name)
        records[str(p)] = record
        paths[name] = p
    namespace = {'pd': pd, 'np': np,
                 'MOLD_RAW': paths['moldoveanu_raw'],
                 'ENG_ADT': paths['engelbrecht_adt']}
    for name in ('IDENTITY', 'CLEAN15', 'MOLD_NEW', 'ENG_NEW'):
        namespace[name] = inv.literal(script, name)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(script), 'exec'), namespace)
    mold, _ = namespace['build_moldoveanu'](pd.read_csv(paths['moldoveanu_shared13_input'], low_memory=False))
    eng, _ = namespace['build_engelbrecht'](pd.read_csv(paths['engelbrecht_shared13_input'], low_memory=False))
    combined = pd.concat([mold, eng], ignore_index=True)
    raw = combined[namespace['CLEAN15']].to_numpy(np.float64)
    md = combined[namespace['IDENTITY']].copy()
    md['unit_key'] = md['dataset'].astype(str) + '::' + md['donor'].astype(str)
    for path, rec in records.items():
        io.check(path, rec)
    io.check(script, expected)
    return raw, md, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['input', 'fit'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = inv.HERE / 'runs/inputs_20261011/melanoma_story'
    spec = study.verify_bundle(bundle)
    if not args.output.resolve().is_relative_to(inv.HERE / 'validation'):
        parser.error('Use a new validation directory')
    out = io.reserve(args.output, study.protected(bundle))
    code = study.code_records()
    try:
        raw = np.load(bundle / 'input/raw.npy', allow_pickle=False)
        metadata = pd.read_csv(bundle / 'input/fit_metadata.csv.gz', low_memory=False)
        md = worker.validate_input(raw, metadata, spec['markers'], spec)
        if args.mode == 'input':
            fresh, original_md, records = reconstructed_input(spec, inv.LEGACY)
            for key in ('dataset', 'unit_key'):
                if not original_md[key].astype(str).equals(md[key].astype(str)):
                    raise ValueError('Original fitting identities differ')
            if not original_md.cell_uid.astype(str).equals(metadata.cell_id.astype(str)):
                raise ValueError('Original cell IDs differ')
            result = {'raw_array_equal': bool(np.array_equal(fresh, raw)),
                      'maximum_difference': float(np.max(np.abs(fresh - raw))),
                      'changed_values': int(np.count_nonzero(fresh != raw)),
                      'fresh_strides': list(fresh.strides), 'frozen_strides': list(raw.strides),
                      'dtype': str(raw.dtype), 'shape': list(raw.shape),
                      'source_records': records, 'identity_equal': True}
        else:
            from threadpoolctl import threadpool_info
            sys.path.insert(0, str(bundle / 'vendor'))
            from ramp import ramp_ver2 as ramp
            before = {'threads': {k: os.environ.get(k) for k in worker.THREADS},
                      'threadpools': threadpool_info(), 'numpy': np.__version__,
                      'python': sys.version, 'canonical': io.record(ramp.__file__)}
            io.write(out / 'ENVIRONMENT.json', before)
            fitted, profiles = worker.calculate(raw, md, spec, ramp)
            outputs = worker.save_result(out, fitted, profiles, spec, md)
            comparisons = {}
            for name, ref in spec['references'].items():
                io.check(inv.LEGACY / ref['source'], ref)
                row = {'byte_identical': outputs[name]['sha256'] == ref['sha256']}
                if name.endswith('.npy'):
                    old = np.load(inv.LEGACY / ref['source'], allow_pickle=False)
                    new = np.load(out / 'result' / name, allow_pickle=False)
                    row.update(maximum_difference=float(np.max(np.abs(old - new))),
                               changed_values=int(np.count_nonzero(old != new)))
                comparisons[name] = row
            result = {'diagnostic_only': True, 'comparisons': comparisons, 'outputs': outputs}
        study.check_code(code)
        study.verify_bundle(bundle)
        result['controller_unchanged'] = code
        io.write(out / 'report.json', result)
        print(result, flush=True)
    except BaseException as exc:
        io.failed(out, exc)
        raise


if __name__ == '__main__':
    main()
