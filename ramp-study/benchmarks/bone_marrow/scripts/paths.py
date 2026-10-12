"""Path safety for relocated study scripts (no scientific transformations)."""

from pathlib import Path


def check_new_output(path, inputs=()):
    """Read-only preflight; does not reserve a path or authorize writing."""
    original = Path(path).expanduser()
    if original.exists() or original.is_symlink():
        raise FileExistsError(f"Output already exists; choose a new run directory: {original}")
    destination = original.resolve()
    for value in inputs:
        source = Path(value).expanduser().resolve(strict=True)
        if destination == source or destination in source.parents or source in destination.parents:
            raise ValueError(f"Output overlaps an input: {destination} / {source}")
    return destination


def new_output(path, inputs=()):
    """Atomically claim a new directory; only the successful caller may write.

    mkdir(exist_ok=False), not the preceding check, arbitrates competing jobs.
    Keep the directory after failures: another job must never reuse or remove
    partial output automatically. This protects cooperating pipeline tasks,
    not against an external process deliberately replacing directory ancestors.
    """
    destination = check_new_output(path, inputs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=False)
    return destination
