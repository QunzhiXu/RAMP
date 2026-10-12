"""Read-only verification of copied MaxFuse driver/configs and existing results."""

import argparse
import json
from pathlib import Path

from migrate import COMMON, HERE, check_cell_order, field, sha


def within(root, name):
    root = Path(root).resolve(strict=True)
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"Nonrelative catalog path: {name}")
    path = (root / name).resolve(strict=True)
    if root not in path.parents:
        raise ValueError(f"Catalog path escapes root: {name}")
    return path


def matches(path, record):
    if path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
        raise ValueError(f"Frozen file changed: {path}")


def verify_local(snapshot):
    snapshot = Path(snapshot).resolve(strict=True)
    catalog = json.loads((snapshot / "catalog.json").read_text())
    count = 0
    for row in catalog["files"]:
        if "copy" in row:
            matches(within(snapshot, row["copy"]), row)
            count += 1
    if sha(COMMON) != catalog["common_registry_sha256"]:
        raise ValueError("Common seven-task registry changed")
    return {"status": "passed", "copied_text_files": count,
            "catalog_sha256": sha(snapshot / "catalog.json"), "original_tree_required": False}


def verify_sources(snapshot, root):
    local = verify_local(snapshot)
    catalog = json.loads((Path(snapshot) / "catalog.json").read_text())
    for row in catalog["files"]:
        matches(within(root, row["source"]), row)
    for link in catalog["historical_hash_links"]:
        expected = field(json.loads(within(root, link["manifest"]).read_text()), link["keys"])
        if sha(within(root, link["target"])) != expected:
            raise ValueError(f"Historical hash link failed: {link}")
    runs = []
    for case in catalog["cases"]:
        for run in case["runs"]:
            observed = check_cell_order(within(root, run["embedding"]), within(root, run["cell_ids"]),
                                        within(root, case["common_metadata"]))
            if any(observed[key] != run[key] for key in observed):
                raise ValueError(f"Result identity changed: {case['case']} / {run['name']}")
            runs.append({"case": case["case"], "run": run["name"], **observed})
    return {**local, "original_tree_required": True, "source_files_checked": len(catalog["files"]),
            "historical_hash_links_checked": len(catalog["historical_hash_links"]), "runs": runs,
            "scope": "Byte/manifest/cell-order checks only; no model or UMAP refit"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, default=HERE / "frozen_20261010")
    p.add_argument("--source-root", type=Path, help="Omit for copied-code/config check only")
    args = p.parse_args(argv)
    result = verify_sources(args.snapshot, args.source_root) if args.source_root else verify_local(args.snapshot)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
