"""Read-only runtime probe for fresh-UMAP handoff verification."""
import argparse
import json
from pathlib import Path
from bootstrap import previous_worker as w


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    args = p.parse_args(); w.load_evaluator(args.bundle)
    print(json.dumps(w.runtime()))
