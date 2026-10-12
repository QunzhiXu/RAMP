"""Read-only runtime/installed-upstream check using the relocated local adapter."""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", type=Path, required=True)
    args = p.parse_args(argv)
    adapter = args.adapter.resolve(strict=True)
    sys.path.insert(0, str(adapter))
    import maxfuse_repro
    from maxfuse_repro import runtime_manifest, verify_upstream_lock
    if not Path(maxfuse_repro.__file__).resolve().is_relative_to(adapter):
        raise RuntimeError("Imported another adapter instead of the relocated copy")
    mismatches = []
    locked = {}
    for line in (adapter / "maxfuse_repro/environment-lock.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        observed = version(name)
        locked[name] = observed
        if observed != expected:
            mismatches.append({"package": name, "expected": expected, "observed": observed})
    print(json.dumps({"adapter": str(Path(maxfuse_repro.__file__).resolve()),
                      "upstream": asdict(verify_upstream_lock()), "runtime": runtime_manifest(),
                      "environment_lock_packages": locked, "environment_lock_mismatches": mismatches}, indent=2))
    if mismatches:
        raise RuntimeError("Installed runtime differs from the preserved environment lock")


if __name__ == "__main__":
    main()
