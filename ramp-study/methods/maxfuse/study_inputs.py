"""Portable seven-case fit-input assembly from frozen full-native inputs.

This is NOT a provider extractor. Technical cases start at method-scaled CSVs;
application cases start at full-native prepared arrays. No new normalization,
cohort selection, or RAMP matrix is substituted. Historical fit selection and
dimension resolution functions are loaded verbatim, without executing the
historical module's top-level code.
"""

import ast
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from migrate import HERE, create_json, sha
from run import reserve
from verify import matches, verify_local, within

SNAPSHOT = HERE / "frozen_20261010"
BENCHMARK = "benchmark_results/_maxfuse_benchmark.py"
FUNCTIONS = ("stable_seed", "select_fit_positions", "resolve_pairwise_dimensions",
             "load_label_free_inputs")
REPLAY_CASES = {"bone_marrow", "melanoma", "lung_cancer"}


class Sources:
    """Use the accepted catalog, then exact fields of pinned manifests as anchors."""

    def __init__(self, root, snapshot=SNAPSHOT, native_root=None, native_case="bone_marrow"):
        self.root = Path(root).resolve(strict=True)
        self.snapshot = Path(snapshot).resolve(strict=True)
        verify_local(self.snapshot)
        self.catalog = json.loads((self.snapshot / "catalog.json").read_text())
        self.expected = {r["source"]: r["sha256"] for r in self.catalog["files"]}
        self.used = {}
        self.attestations = []
        self.native_root = Path(native_root).resolve(strict=True) if native_root is not None else None
        self.native_paths = {}
        self.native_provenance = None
        self.native_manifest_name = "manifest.json"
        if self.native_root is not None:
            if native_case not in ("bone_marrow", "tnbc", "melanoma", "lung_cancer", "hnscc_story", "melanoma_story", "lung_story"):
                raise ValueError("Native provider rebuild supports only the seven registered benchmark cases")
            if native_case == "tnbc":
                self.native_manifest_name = "input_manifest.json"
            manifest_path = within(self.native_root, self.native_manifest_name)
            completion_path = within(self.native_root, "COMPLETE.json")
            completion = json.loads(completion_path.read_text())
            expected_cells = {"bone_marrow": 88031, "tnbc": 62701, "melanoma": 60000, "lung_cancer": 60000, "hnscc_story": 94488, "melanoma_story": 165686, "lung_story": 84000}[native_case]
            if completion.get("status") != "complete" or completion.get("cells") != expected_cells or completion.get("manifest_sha256") != sha(manifest_path):
                raise ValueError("Unverified or incomplete provider rebuild for " + native_case)
            if native_case == "bone_marrow":
                for slug, width in (("bm_codex", 48), ("bm_cytof", 31), ("triana_abseq", 96)):
                    for leaf in (f"active_input_{width}.csv.gz", "shared_input_14.csv.gz"):
                        relative = f"{slug}/{leaf}"
                        self.native_paths[f"benchmark_results/bone_marrow/maxfuse_v2/processed/{relative}"] = relative
            elif native_case == "tnbc":
                if completion.get("case") != "tnbc":
                    raise ValueError("Not a completed TNBC provider rebuild")
                relatives = [f"{slug}/{kind}_processed.csv.gz"
                             for slug in ("tietscher_imc", "meyer_imc", "shiao_codex") for kind in ("active", "shared")]
                relatives.append("shared_processed_float64.npy")
                self.native_paths = {"benchmark_results/tnbc/maxfuse_archive/input/" + p: p for p in relatives}
            elif native_case == "melanoma":
                if completion.get("case") != "melanoma" or completion.get("stage") != "method":
                    raise ValueError("Not a completed melanoma technical method-input rebuild")
                self.native_paths = {"benchmark_results/melanoma/maxfuse_v2/method_inputs/" + p: p
                    for p in (f"{slug}_{kind}.csv.gz" for slug in ("hickey_codex", "hoch_imc", "ptacek_mibi") for kind in ("active", "shared"))}
            elif native_case == "lung_cancer":
                if completion.get("case") != "lung_cancer" or completion.get("stage") != "method":
                    raise ValueError("Not a completed lung technical method-input rebuild")
                self.native_paths = {"benchmark_results/lung_cancer/maxfuse_v2/method_inputs/" + p: p
                    for p in (f"{slug}_{kind}.csv.gz" for slug in ("senosain_cytof", "cords_imc", "monkman_codex") for kind in ("active", "shared"))}
            elif native_case == "hnscc_story":
                if completion.get("case") != "hnscc_story" or completion.get("stage") != "native":
                    raise ValueError("Not a completed HNSCC native-input rebuild")
                self.native_paths = {"figure_HNSCC/Comparison/maxfuse_full_panel/data/" + p: p
                    for p in (f"{slug}/{leaf}" for slug in ("rahim_adt", "haist_codex")
                              for leaf in ("active.npy", "shared.npy", "cell_ids.npy", "global_row_indices.npy",
                                           "active_feature_names.txt", "shared_feature_names.txt"))}
            elif native_case == "melanoma_story":
                if completion.get("case") != "melanoma_story" or completion.get("stage") != "native":
                    raise ValueError("Not a completed melanoma story native-input rebuild")
                self.native_paths = {"figure_melanoma/Comparison/maxfuse_full_panel/data/" + p: p
                    for p in (f"{slug}/{leaf}" for slug in ("moldoveanu_imc", "engelbrecht_adt")
                              for leaf in ("active.npy", "shared.npy", "cell_ids.npy", "global_row_indices.npy",
                                           "active_feature_names.txt", "shared_feature_names.txt"))}
            else:
                if completion.get("case") != "lung_story" or completion.get("stage") != "native":
                    raise ValueError("Not a completed lung story native-input rebuild")
                self.native_paths = {"story_lung_HLA_ABC/Comparison_senosain140k_20260921/maxfuse_full_panel/data/" + p: p
                    for p in (f"{slug}/{leaf}" for slug in ("senosain_cytof", "cords_imc", "zheng_mif")
                              for leaf in ("active.npy", "shared.npy", "cell_ids.npy", "global_row_indices.npy",
                                           "active_feature_names.txt", "shared_feature_names.txt"))}
            self.native_provenance = {"root": str(self.native_root), "manifest_sha256": sha(manifest_path),
                                      "completion_sha256": sha(completion_path), "file_mapping": self.native_paths}

    def location(self, relative):
        if relative in self.native_paths:
            return within(self.native_root, self.native_paths[relative])
        return within(self.root, relative)

    def relative(self, path):
        value = str(path)
        if Path(value).is_absolute():
            # Historical manifests contain the old host's prefix. No access to it.
            marker = "/final_xu/"
            if marker not in value:
                raise ValueError(f"Unrecognized historical absolute path: {value}")
            value = value.split(marker, 1)[1]
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ValueError(f"Invalid relative input: {value}")
        return value

    def read(self, relative):
        relative = self.relative(relative)
        if relative not in self.expected:
            raise ValueError(f"Unattested source: {relative}")
        path = self.location(relative)
        observed = sha(path)
        if observed != self.expected[relative]:
            raise ValueError(f"Frozen source changed: {relative}")
        self.used[relative] = {"sha256": observed, "bytes": path.stat().st_size}
        if relative in self.native_paths:
            self.used[relative]["read_from_provider_rebuild"] = str(path)
        return path

    def json(self, relative):
        return json.loads(self.read(relative).read_text())

    def bind(self, manifest, keys, relative):
        relative = self.relative(relative)
        obj = self.json(manifest)
        for key in keys:
            obj = obj[key]
        if not isinstance(obj, str) or len(obj) != 64:
            raise ValueError("Historical field is not a SHA-256 digest")
        if relative in self.expected and self.expected[relative] != obj:
            raise ValueError(f"Conflicting historical hashes: {relative}")
        self.expected[relative] = obj
        self.attestations.append({"manifest": self.relative(manifest), "keys": keys,
                                  "target": relative, "sha256": obj})
        return self.read(relative)

    def recheck(self):
        for path, record in list(self.used.items()):
            matches(self.location(path), record)
        if self.native_provenance:
            for name, key in ((self.native_manifest_name, "manifest_sha256"), ("COMPLETE.json", "completion_sha256")):
                if sha(within(self.native_root, name)) != self.native_provenance[key]:
                    raise ValueError("Provider-rebuild manifest changed during assembly")


def historical_functions(snapshot=SNAPSHOT):
    """Only these four exact function definitions; no module-level I/O/imports."""
    path = Path(snapshot) / "study_sources" / BENCHMARK
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    if {n.name for n in selected} != set(FUNCTIONS):
        raise ValueError("Historical function inventory changed")
    import hashlib
    context = {"np": np, "pd": pd, "copy": copy, "Any": Any, "Path": Path,
               "hashlib": hashlib,
               "PAIRWISE_RANK_ADAPTATION": "clip_non_null_official_targets_to_each_view_feature_rank"}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), context)
    return context


def npz_equal(left, right, current_order=None, historical_order=None):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError("Fit NPZ fields changed")
        for key in a.files:
            old_key = key
            if current_order is not None:
                position, suffix = key.split("_", 1)
                name = current_order[int(position.removeprefix("view"))]
                old_key = f"view{historical_order.index(name)}_{suffix}"
            if a[key].dtype != b[old_key].dtype or not np.array_equal(a[key], b[old_key]):
                raise ValueError(f"Fit NPZ array differs: {key}")
    return True


def technical_views(source, case, config):
    first_manifest = source.json(case["runs"][0]["manifest"])
    if case["case"] in REPLAY_CASES:
        prepared = source.relative(first_manifest["prepared_source"]["directory"])
        prepared_manifest = prepared + "/manifest.json"
    else:
        prepared_manifest = source.relative(first_manifest["preprocessing_manifest"])
    frozen = source.json(prepared_manifest)
    for index, record in enumerate(frozen["sources"]):
        source.bind(prepared_manifest, ["sources", index, "sha256"], record["path"])
    ctx = historical_functions(source.snapshot)
    ctx.update(resolve_from_final_xu=source.read, sha256=sha)
    shared, metadata, active, features, _ = ctx["load_label_free_inputs"](config)
    views = []
    for spec in config["datasets"]:
        slug = spec["slug"]
        rows = np.flatnonzero(metadata.dataset.eq(spec["name"]).to_numpy())
        fit = ctx["select_fit_positions"](metadata, spec["name"], config["fit_sampling"], config["base_seed"])
        informative = np.isfinite(shared[fit]).all(axis=1)
        if config["fit_sampling"]["drop_zero_within_cell_variance"]:
            informative &= np.std(shared[fit], axis=1) > 1e-10
        fit = fit[informative]
        if len(fit) < 3:
            raise ValueError("Fewer than three informative fit cells")
        local = np.searchsorted(rows, fit)
        if np.any(local >= len(rows)) or not np.array_equal(rows[local], fit):
            raise ValueError("Invalid fit selection")
        old_fit = source.bind(prepared_manifest, ["outputs", frozen["fit_index_files"][slug]],
                              str(Path(prepared_manifest).parent / frozen["fit_index_files"][slug]))
        if not np.array_equal(fit, np.load(old_fit, allow_pickle=False)):
            raise ValueError("Historical fit selection changed")
        views.append({"name": slug, "dataset": spec["name"], "active": active[slug],
                      "shared": shared[rows], "features": features[slug], "rows": rows,
                      "fit_local": local, "ids": metadata.iloc[rows].cell_id.to_numpy(dtype=str)})
    return views, metadata.cell_id.to_numpy(dtype=str), list(config["shared_markers"])


def application_views(source, case, config):
    base = case["source_root"] + "/data"
    manifest_path = base + "/manifest.json"
    manifest = source.json(manifest_path)
    views = []
    for index, spec in enumerate(manifest["views"]):
        def get(leaf):
            relative = spec["slug"] + "/" + leaf
            return source.bind(manifest_path, ["output_sha256", relative], base + "/" + relative)
        rows = np.load(get("global_row_indices.npy"), allow_pickle=False)
        ids = np.load(get("cell_ids.npy"), allow_pickle=False)
        views.append({"name": f"view{index}", "dataset": spec["dataset"],
                      "active": np.load(get("active.npy"), allow_pickle=False),
                      "shared": np.load(get("shared.npy"), allow_pickle=False),
                      "features": get("active_feature_names.txt").read_text().splitlines(),
                      "rows": rows, "ids": ids, "fit_local": np.arange(len(ids))})
        if get("shared_feature_names.txt").read_text().splitlines() != config["markers"]:
            raise ValueError("Shared marker order changed")
    ids = np.empty(sum(len(v["ids"]) for v in views), dtype=object)
    for view in views:
        ids[view["rows"]] = view["ids"]
    return views, ids.astype(str), list(config["markers"])


def validate_views(views, ids, markers):
    if len(views) not in (2, 3) or pd.Index(ids).has_duplicates:
        raise ValueError("Invalid view count or duplicate full IDs")
    rows = np.concatenate([v["rows"] for v in views])
    if not np.array_equal(np.sort(rows), np.arange(len(ids))):
        raise ValueError("View rows do not partition the cohort")
    for v in views:
        if v["active"].shape != (len(v["ids"]), len(v["features"])) or v["shared"].shape != (len(v["ids"]), len(markers)):
            raise ValueError("Matrix shape mismatch")
        if not np.isfinite(v["active"]).all() or not np.isfinite(v["shared"]).all():
            raise ValueError("Nonfinite full panel")
        if not np.array_equal(ids[v["rows"]], v["ids"]):
            raise ValueError("View/full cohort ID mismatch")


def branch_input(views, markers, order, pairwise):
    payload = {}
    for position, i in enumerate(order):
        v = views[i]
        content = {"shared": v["shared"][v["fit_local"]], "active": v["active"][v["fit_local"]],
                   "cell_ids": np.asarray(v["ids"][v["fit_local"]], dtype=str),
                   "shared_feature_names": np.asarray(markers, dtype=str),
                   "active_feature_names": np.asarray(v["features"], dtype=str)}
        for key, array in content.items():
            if pairwise:
                name = key.replace("_feature_names", f"{position + 1}_feature_names") if key.endswith("_feature_names") else f"{key}{position + 1}"
            else:
                name = f"view{position}_{key}"
            payload[name] = array
    return payload


def resolve_configs(profile, views, order, width, pairwise):
    resolver = historical_functions()["resolve_pairwise_dimensions"]
    configs, audits = {}, {}
    for i in order[1:]:
        cfg, audit = resolver(profile, policy="clip_non_null_official_targets_to_each_view_feature_rank",
                              shared_width_first=width, shared_width_second=width,
                              active_width_first=len(views[order[0]]["features"]),
                              active_width_second=len(views[i]["features"]))
        configs[views[i]["name"]], audits[views[i]["name"]] = cfg, audit
    return (next(iter(configs.values())) if pairwise else configs), audits


def output_records(root):
    return {p.relative_to(root).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}


def prepare(case_name, source_root, output, with_cache=False, native_root=None):
    if native_root is not None and case_name not in ("bone_marrow", "tnbc", "melanoma", "lung_cancer", "hnscc_story", "melanoma_story", "lung_story"):
        raise ValueError("--native-root supports only the seven registered benchmark cases")
    source = Sources(source_root, native_root=native_root, native_case=case_name)
    cases = {c["case"]: c for c in source.catalog["cases"]}
    case = cases[case_name]
    if with_cache and case_name not in REPLAY_CASES:
        raise ValueError("Only the three historical replay cases accept --with-replay-cache")
    config = source.json(case["config"])
    root = reserve(output, (source.root, SNAPSHOT, *((source.native_root,) if source.native_root else ())))
    try:
        technical = case_name in ("bone_marrow", "tnbc", "melanoma", "lung_cancer")
        views, ids, markers = (technical_views if technical else application_views)(source, case, config)
        validate_views(views, ids, markers)
        expected = pd.read_csv(source.read(case["common_metadata"]), usecols=["cell_id"], dtype=str).cell_id.to_numpy()
        if not np.array_equal(ids, expected):
            raise ValueError("Full IDs do not match the common benchmark cohort")
        np.save(root / "cell_ids.npy", ids, allow_pickle=False)
        profile = source.json(config["pairwise_config"] if technical else "benchmark_results/bone_marrow/maxfuse_v2/pairwise_config.json")
        stored_views = []
        for v in views:
            p = root / "views" / v["name"]
            p.mkdir(parents=True)
            for key in ("active", "ids", "rows", "fit_local"):
                np.save(p / (key + ".npy"), v[key], allow_pickle=False)
            stored_views.append({k: v[k] for k in ("name", "dataset", "features")})
        branches = []
        for index, old_run in enumerate(case["runs"]):
            pairwise = len(views) == 2
            order = list(range(2)) if pairwise else [(index + j) % 3 for j in range(3)]
            name = old_run["name"]
            p = root / "branches" / name
            p.mkdir(parents=True)
            np.savez_compressed(p / "fit_input.npz", **branch_input(views, markers, order, pairwise))
            resolved, audit = resolve_configs(profile, views, order, len(markers), pairwise)
            create_json(p / "config.json", resolved)
            create_json(p / "dimension_audit.json", audit)
            if technical:
                mp = source.json(old_run["manifest"])
                old_formal = source.relative(mp["legacy_pairwise_source"]["alignment"]) + "/formal_fit/manifest.json" if case_name in REPLAY_CASES else str(Path(old_run["manifest"]).parent / "formal_fit/manifest.json")
            else:
                old_formal = old_run["manifest"]
            fm = source.json(old_formal)
            old_input = source.bind(old_formal, ["input_sha256"], fm["input"])
            # Legacy v2 stores hub + canonical outer order. Archive uses cyclic
            # view order. Compare every array by named view, NOT position alone.
            current_order = [views[i]["name"] for i in order]
            npz_equal(p / "fit_input.npz", old_input,
                      None if pairwise else current_order,
                      None if pairwise else fm["views"])
            field = "config" if pairwise else "pair_config_map"
            hash_field = "config_file_sha256" if pairwise else "pair_config_map_sha256"
            # Pairwise CLI records config_file, unlike the three-view map.
            field = "config_file" if pairwise else field
            old_config = source.bind(old_formal, [hash_field], fm[field])
            if resolved != json.loads(old_config.read_text()):
                raise ValueError("Resolved pairwise parameters differ from historical run")
            cache = None
            if with_cache:
                relative = str(Path(old_formal).parent / "triplet_matching.npz")
                original = source.bind(old_formal, ["output_sha256", "triplet_matching.npz"], relative)
                # New inode, no writable hard link to original results.
                with original.open("rb") as src, (p / "triplet_matching.npz").open("xb") as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
                cache = {"triplet_index_order": fm["triplet_index_order"],
                         "audit": fm["audit"], "wrapper_version": fm["wrapper_version"],
                         "source_manifest": old_formal, "source_manifest_sha256": sha(source.read(old_formal))}
            extension = None if pairwise else (config["three_view_extension"] if technical else {
                "svd_components": {v["name"]: None for v in views}, "gcca_components": 15,
                "gcca_initialization_max_iter": 2000, "gcca_refinement_max_iter": 2000,
                "gcca_tolerance": 0.001, "outer_reuse_policy": "allow"})
            branches.append({"name": name, "mode": "pairwise" if pairwise else "three-view",
                             "order": [views[i]["name"] for i in order], "extension": extension,
                             "cache": cache, "historical_fit_arrays_equal": True,
                             "fit_array_comparison": "by named view; Archive cyclic order may differ from v2 NPZ positions",
                             "historical_resolved_config_equal": True})
        source.recheck()
        verify_local(SNAPSHOT)
        manifest = {"schema": "ramp-study.maxfuse.fit_bundle.v1", "status": "complete", "case": case_name,
                    "catalog_sha256": sha(SNAPSHOT / "catalog.json"), "markers": markers,
                    "views": stored_views, "branches": branches, "cells": len(ids),
                    "source_files": source.used, "historical_field_links": source.attestations,
                    "native_input_override": source.native_provenance,
                    "boundary": "Frozen method-scaled full-native inputs, not provider extraction; no new normalization or cohort sampling",
                    "outputs": output_records(root)}
        create_json(root / "MANIFEST.json", manifest)
    except BaseException as error:
        create_json(root / "FAILED.json", {"error": repr(error), "partial_output_retained": True})
        raise
    return root


def verify_bundle(root):
    root = Path(root).resolve(strict=True)
    manifest = json.loads(within(root, "MANIFEST.json").read_text())
    if manifest.get("schema") != "ramp-study.maxfuse.fit_bundle.v1" or manifest.get("status") != "complete":
        raise ValueError("Incomplete or unsupported fit bundle")
    if manifest["catalog_sha256"] != sha(SNAPSHOT / "catalog.json"):
        raise ValueError("Bundle belongs to a different accepted catalog")
    def single_name(value):
        if not isinstance(value, str) or not value or value in (".", "..") or Path(value).name != value or "\\" in value:
            raise ValueError("Unsafe bundle view/branch name")
    names = [v["name"] for v in manifest["views"]]
    branch_names = [b["name"] for b in manifest["branches"]]
    for name in names + branch_names:
        single_name(name)
    if len(names) not in (2, 3) or len(set(names)) != len(names) or not branch_names or len(set(branch_names)) != len(branch_names):
        raise ValueError("Duplicate/invalid view or branch inventory")
    for b in manifest["branches"]:
        if len(b["order"]) != len(names) or set(b["order"]) != set(names):
            raise ValueError("Invalid branch view order")
        if b["mode"] != ("pairwise" if len(names) == 2 else "three-view"):
            raise ValueError("Branch mode does not match view count")
    expected = set(manifest["outputs"]) | {"MANIFEST.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected:
        raise ValueError("Unexpected or missing bundle files")
    for path, record in manifest["outputs"].items():
        matches(within(root, path), record)
    return manifest
