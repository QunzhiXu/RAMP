"""Write a small synthetic file-based RAMP input bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from quickstart import synthetic_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/output/input"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    targets = {
        "raw": args.output / "raw.npy",
        "metadata": args.output / "cell_metadata.csv.gz",
        "markers": args.output / "markers.txt",
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "example input already exists; remove it or choose another --output"
        )

    raw, metadata, markers = synthetic_data()
    np.save(targets["raw"], raw)
    metadata.to_csv(
        targets["metadata"],
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    targets["markers"].write_text("\n".join(markers) + "\n", encoding="utf-8")
    print(f"Wrote example inputs to {args.output}")


if __name__ == "__main__":
    main()
