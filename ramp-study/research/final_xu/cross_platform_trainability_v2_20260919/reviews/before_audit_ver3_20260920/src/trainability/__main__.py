"""Inspect implementation status; deliberately no formal-run command yet."""
import argparse
import json

from .core import ROOT, Q_GRID, RELEASE_CHECKS
from .inputs import METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status"])
    parser.parse_args()
    config = json.loads((ROOT / "config/protocol.draft.json").read_text())
    print(json.dumps(dict(package="0.1.0", configuration_status=config["status"],
        methods=METHODS, q_grid=Q_GRID, release_checks=RELEASE_CHECKS,
        formal_run_available=False, tasks=[t["name"] for t in config["tasks"]],
        next_steps=["review native dependencies and patient mappings",
                    "implement and verify six alignment adapters",
                    "canonical melanoma and runtime pilots (not run)",
                    "freeze executable budgets and complete real-data orchestration"]), indent=2))


if __name__ == "__main__":
    main()
