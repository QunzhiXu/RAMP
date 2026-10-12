"""Provenance, write boundaries and explicit release checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
Q_GRID = tuple(i / 20 for i in range(1, 20))


class ProtocolError(ValueError):
    """A violated experimental contract, not an imputable missing score."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_for(base: int, *parts: str) -> int:
    return int(digest([int(base), *parts])[:8], 16)


def output_path(path: str | Path) -> Path:
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    require(path.is_relative_to(ROOT.resolve()), "Writes must stay inside the v2 folder")
    return path


def write_json(path: str | Path, value: Any, *, replace: bool = False) -> Path:
    path = output_path(path)
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if replace else "x") as f:
        f.write(text)
    return path


def unique_ids(ids, name="cell IDs") -> tuple[str, ...]:
    result = tuple(str(x) for x in ids)
    require(all(x and x not in {"None", "nan"} for x in result), f"Invalid {name}")
    require(len(result) == len(set(result)), f"Duplicate {name}")
    return result


def representation_cache_key(*, input_sha256: str, cell_ids, markers,
                             masked_markers, input_contract: dict,
                             method: str, parameters: dict, seed: int,
                             code_sha256: str, environment_sha256: str) -> str:
    # Includes order and the full U. No implicit reuse of historical full-panel outputs.
    return digest(dict(input_sha256=input_sha256, cell_ids=unique_ids(cell_ids),
                       markers=list(markers), masked_markers=sorted(masked_markers),
                       input_contract=input_contract, method=method,
                       parameters=parameters, seed=seed, code_sha256=code_sha256,
                       environment_sha256=environment_sha256))


RELEASE_CHECKS = (
    "input_dependencies", "patient_mapping", "source_target_identity", "fixed_fitting_pool",
    "mask_isolation", "supervision_budget", "j3_equal_budget",
    "source_only_selection", "representation_provenance", "aggregation_checks",
    "runtime_pilot", "canonical_melanoma", "configuration_frozen",
)


def validate_release(config: dict, checks: dict) -> None:
    require(config.get("status") == "frozen", "Configuration is still a draft")
    require(tuple(config.get("q_grid", ())) == Q_GRID, "Q grid differs from protocol")
    for name in RELEASE_CHECKS:
        item = checks.get(name, {})
        require(item.get("status") == "passed", f"Release check not passed: {name}")
        evidence = Path(item.get("evidence_path", ""))
        require(evidence.is_file(), f"Missing evidence for {name}")
        require(file_sha256(evidence) == item.get("sha256"), f"Evidence changed: {name}")
