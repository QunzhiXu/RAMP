"""Inspect integration and formal-release status without launching jobs."""
import argparse
import json

from .core import ROOT, Q_GRID, RELEASE_CHECKS
from .inputs import METHODS
from . import __version__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status"])
    parser.parse_args()
    config = json.loads((ROOT / "config/protocol.draft.json").read_text())
    print(json.dumps(dict(package=__version__, configuration_status=config["status"],
        methods=METHODS, q_grid=Q_GRID, release_checks=RELEASE_CHECKS,
        manifest_runner_available=True, formal_run_enabled=config['formal_run_enabled'],
        tasks=[t["name"] for t in config["tasks"]],
        registered_inputs=str(ROOT/'data/real_inputs_v0_3_r2/SUMMARY.json'),
        real_e2e_pilot=str(ROOT/'validation/e2e_pilot_bone_v0_3_r2/SUMMARY.json'),
        next_steps=["resolve or explicitly exclude Ptacek upstream dependencies for strict J2",
                    "keep Monkman patient-unknown results unit-descriptive",
                    "freeze formal numerical grid and release evidence before full execution"]), indent=2))


if __name__ == "__main__":
    main()
