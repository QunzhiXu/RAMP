"""Execute one explicit fresh-fit or cached-triplet replay in an owned directory."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from migrate import create_json, sha
from run import launch
from study_inputs import SNAPSHOT, output_records, verify_bundle
from verify import verify_local


def load_core():
    verify_local(SNAPSHOT)
    adapter = SNAPSHOT / "adapter"
    sys.path.insert(0, str(adapter))
    import maxfuse_repro
    if not Path(maxfuse_repro.__file__).resolve().is_relative_to(adapter):
        raise RuntimeError("Wrong MaxFuse adapter imported")
    return maxfuse_repro


def fresh_options(branch):
    if branch["mode"] == "pairwise":
        return []
    e, order = branch["extension"], branch["order"]
    return ["--views", *order, "--hub", order[0], "--svd-components",
            *["none" if e["svd_components"][n] is None else str(e["svd_components"][n]) for n in order],
            "--gcca-components", str(e["gcca_components"]),
            "--gcca-initialization-max-iter", str(e["gcca_initialization_max_iter"]),
            "--gcca-refinement-max-iter", str(e["gcca_refinement_max_iter"]),
            "--gcca-tolerance", str(e["gcca_tolerance"]),
            "--outer-reuse-policy", e["outer_reuse_policy"]]


def validate_triplets(branch, arrays, triplets):
    cache = branch["cache"]
    if cache is None or branch["mode"] != "three-view":
        raise ValueError("Replay requires an explicitly frozen historical triplet cache")
    order = cache["triplet_index_order"]
    if set(order) != set(branch["order"]) or order[0] != branch["order"][0]:
        raise ValueError("Cached hub/view order differs")
    if cache["audit"]["outer_reuse_policy"] != branch["extension"]["outer_reuse_policy"]:
        raise ValueError("Cached outer-reuse policy differs")
    indices = {}
    for name, key in zip(order, ("hub", "outer_left", "outer_right")):
        values = triplets[key]
        if values.ndim != 1 or values.dtype.kind not in "iu" or len(values) == 0:
            raise ValueError("Cached indices must be a nonempty integer vector")
        ids = arrays[name]["cell_ids"]
        if np.any(values < 0) or np.any(values >= len(ids)):
            raise ValueError("Cached index out of bounds")
        if not np.array_equal(ids[values], triplets[key + "_cell_id"]):
            raise ValueError("Cached cell IDs do not match newly assembled fit-input indices")
        indices[name] = values
    lengths = [len(a) for a in indices.values()]
    if len(set(lengths)) != 1:
        raise ValueError("Unequal triplet lengths")
    for key in ("left_score", "right_score"):
        scores = triplets[key]
        if scores.shape != (lengths[0],) or not np.isfinite(scores).all():
            raise ValueError("Invalid cached scores")
    return indices


def replay(branch, fit_input, cache_path, output):
    core = load_core()
    from maxfuse_repro.gcca import fit_archive_gcca, matched_component_correlations
    from maxfuse_repro.projection import build_projection_model
    order, e = branch["order"], branch["extension"]
    with np.load(fit_input, allow_pickle=False) as f:
        expected = {f"view{i}_{key}" for i in range(3) for key in (
            "shared", "active", "cell_ids", "shared_feature_names", "active_feature_names")}
        if set(f.files) != expected:
            raise ValueError("Unexpected fit-input fields")
        arrays = {name: {key: f[f"view{i}_{key}"].copy() for key in (
            "active", "cell_ids", "active_feature_names")} for i, name in enumerate(order)}
    with np.load(cache_path, allow_pickle=False) as f:
        triplets = {key: f[key].copy() for key in f.files}
    indices = validate_triplets(branch, arrays, triplets)
    pre = {n: core.archive_svd_scores(arrays[n]["active"], e["svd_components"][n]) for n in order}
    gcca = fit_archive_gcca([pre[n].scores for n in order], [indices[n] for n in order],
                          requested_components=e["gcca_components"],
                          initialization_max_iter=e["gcca_initialization_max_iter"],
                          refinement_max_iter=e["gcca_refinement_max_iter"], tolerance=e["gcca_tolerance"])
    features = {n: tuple(arrays[n]["active_feature_names"].astype(str)) for n in order}
    model = build_projection_model(order, pre, gcca, features)
    roundtrip = {}
    output.mkdir(exist_ok=False)
    for i, n in enumerate(order):
        error = float(np.max(np.abs(model.transform(n, arrays[n]["active"], features[n]) - gcca.embeddings[i])))
        if error > 1e-9:
            raise RuntimeError("Replay projection roundtrip failed")
        roundtrip[n] = error
        np.save(output / f"view{i}_gcca_embedding.npy", gcca.embeddings[i])
        np.save(output / f"view{i}_gcca_cell_ids.npy", arrays[n]["cell_ids"])
    model.save(output / "projection_model.npz")
    correlations = matched_component_correlations(list(gcca.embeddings), [indices[n] for n in order])
    if not np.isfinite(correlations).all():
        raise RuntimeError("Nonfinite matched correlations")
    create_json(output / "manifest.json", {
        "status": "complete", "route": "cached_pairwise_triplets_then_archive_gcca",
        "input_sha256": sha(fit_input), "triplet_sha256": sha(cache_path),
        "pairwise_subprocess_invoked": False, "legacy_ridge_embedding_or_model_used": False,
        "historical_pairwise_wrapper": branch["cache"]["wrapper_version"],
        "current_wrapper": core.__version__, "configuration": e,
        "gcca_components": gcca.n_components, "gcca_iteration_counts": list(gcca.iteration_counts),
        "projection_roundtrip_max_abs_error": roundtrip,
        "mean_matched_component_correlation": float(np.mean(correlations)),
        "output_sha256": {k: r["sha256"] for k, r in output_records(output).items()}})
    return output


def full_output(bundle, manifest, branch, formal, destination):
    core = load_core()
    ids = np.load(bundle / "cell_ids.npy", allow_pickle=False)
    if branch["mode"] == "pairwise":
        parts = [np.load(formal / f"cca_embedding{i}.npy", allow_pickle=False) for i in (1, 2)]
        part_ids = np.concatenate([np.load(formal / f"cca_embedding{i}_cell_ids.npy", allow_pickle=False) for i in (1, 2)])
        if len(part_ids) != len(ids) or not pd.Index(part_ids).is_unique:
            raise ValueError("Pairwise output IDs are incomplete/duplicated")
        positions = pd.Index(part_ids).get_indexer(ids)
        if np.any(positions < 0):
            raise ValueError("Pairwise output lacks cohort IDs")
        values = np.concatenate(parts)[positions]
    else:
        model = core.ThreeViewExtensionProjectionModel.load(formal / "projection_model.npz")
        width = model.views[model.view_order[0]].gcca_weight.shape[1]
        values = np.empty((len(ids), width), dtype=np.float64)
        for v in manifest["views"]:
            directory = bundle / "views" / v["name"]
            rows = np.load(directory / "rows.npy", allow_pickle=False)
            active = np.load(directory / "active.npy", allow_pickle=False)
            view_ids = np.load(directory / "ids.npy", allow_pickle=False)
            if not np.array_equal(ids[rows], view_ids):
                raise ValueError("Full projection ID mismatch")
            values[rows] = model.transform(v["name"], active, v["features"])
    if values.ndim != 2 or len(values) != len(ids) or not np.isfinite(values).all():
        raise ValueError("Invalid full-cohort embedding")
    np.save(destination / "values.npy", values, allow_pickle=False)
    np.save(destination / "cell_ids.npy", ids, allow_pickle=False)
    return list(values.shape)


def execute(bundle, name, route, output):
    bundle, output = Path(bundle).resolve(strict=True), Path(output).resolve(strict=True)
    # The public launcher atomically reserves this directory, then supplies a
    # request. The worker does not accept an arbitrary pre-existing destination.
    request = json.loads((output / "REQUEST.json").read_text())
    if request["bundle"] != str(bundle) or request["branch"] != name or request["route"] != route:
        raise ValueError("Worker request does not match its owned output")
    if request["bundle_manifest_sha256"] != sha(bundle / "MANIFEST.json"):
        raise ValueError("Bundle manifest changed before execution")
    if (output / "STARTED.json").exists() or (output / "COMPLETE.json").exists():
        raise FileExistsError("Worker already started")
    create_json(output / "STARTED.json", {"route": route})
    manifest = verify_bundle(bundle)
    branches = {b["name"]: b for b in manifest["branches"]}
    branch = branches[name]
    p = bundle / "branches" / name
    if route == "replay":
        formal = replay(branch, p / "fit_input.npz", p / "triplet_matching.npz", output / "formal")
    elif route == "fresh":
        launched = launch(Path(sys.executable).absolute(), branch["mode"], p / "fit_input.npz",
                          p / "config.json", output / "fit", fresh_options(branch))
        formal = launched / "result"
    else:
        raise ValueError("Unknown route")
    shape = full_output(bundle, manifest, branch, formal, output)
    verify_bundle(bundle)
    verify_local(SNAPSHOT)
    if sha(bundle / "MANIFEST.json") != request["bundle_manifest_sha256"]:
        raise RuntimeError("Bundle manifest changed during execution")
    create_json(output / "COMPLETE.json", {
        "case": manifest["case"], "branch": name, "route": route, "shape": shape,
        "bundle_manifest_sha256": request["bundle_manifest_sha256"],
        "values_sha256": sha(output / "values.npy"), "cell_ids_sha256": sha(output / "cell_ids.npy"),
        "formal_manifest_sha256": sha(formal / "manifest.json"),
        "scientific_scope": "Embedding only; no new UMAP, metric, biological analysis or provider extraction"})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--route", choices=["fresh", "replay"], required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    execute(args.bundle, args.branch, args.route, args.output)


if __name__ == "__main__":
    main()
