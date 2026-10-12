"""Safe launcher for the unchanged generic MaxFuse adapter, not a cohort builder.

Input must already contain the study's prepared shared AND full active arrays.
No implicit preprocessing, downsampling, dimension clipping or seed changes.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess

from migrate import HERE, create_json, sha
from verify import verify_local


def reserve(output, protected):
    supplied = Path(output).expanduser()
    if supplied.exists() or supplied.is_symlink():
        raise FileExistsError(f"Choose a new output directory: {supplied}")
    path = supplied.resolve()
    for item in protected:
        original = Path(item).resolve(strict=True)
        if path == original or path in original.parents or original in path.parents:
            raise ValueError(f"Output overlaps input/code: {original}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path


def command(python, adapter, mode, input_path, config_path, result_path, extra):
    # Never allow forwarding to override protected input/output/config arguments,
    # including argparse abbreviations such as --out or --pair-conf.
    reserved = ["--output", "--input", "--config", "--pair-config-map"]
    for token in extra:
        head = token.split("=", 1)[0]
        if head.startswith("--") and any(option.startswith(head) for option in reserved):
            raise ValueError(f"Protected forwarded option: {token}")
    if mode not in {"pairwise", "three-view"}:
        raise ValueError(f"Unknown mode: {mode}")
    script = "run_maxfuse_pairwise.py" if mode == "pairwise" else "run_maxfuse_three_view_extension.py"
    config_flag = "--config" if mode == "pairwise" else "--pair-config-map"
    return [str(python), "-B", str(adapter / script), "--input", str(input_path),
            config_flag, str(config_path), *extra, "--output", str(result_path)]


def launch(python, mode, input_path, config_path, output, extra=(), snapshot=None):
    snapshot = Path(snapshot or HERE / "frozen_20261010").resolve(strict=True)
    local_check = verify_local(snapshot)
    # Do NOT resolve the interpreter symlink: that can discard virtualenv identity.
    python = Path(python).expanduser().absolute()
    if not python.is_file():
        raise FileNotFoundError(python)
    input_path, config_path = [Path(p).resolve(strict=True) for p in (input_path, config_path)]
    input_hash, config_hash = sha(input_path), sha(config_path)
    adapter = snapshot / "adapter"
    # Validate option routing before reserving output.
    command(python, adapter, mode, input_path, config_path, Path(output) / "result", extra)
    root = reserve(output, (input_path, config_path, snapshot, python))
    cmd = command(python, adapter, mode, input_path, config_path, root / "result", extra)
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    env.pop("PYTHONPATH", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = "1"
    for key, dirname in [("TMPDIR", "tmp"), ("NUMBA_CACHE_DIR", "cache/numba"),
                         ("MPLCONFIGDIR", "cache/matplotlib"), ("XDG_CACHE_HOME", "cache/xdg")]:
        folder = root / dirname
        folder.mkdir(parents=True, exist_ok=True)
        env[key] = str(folder)
    execution = {"mode": mode, "command": cmd, "input_sha256": input_hash, "config_sha256": config_hash,
                 "copied_source_check": local_check, "launcher_sha256": sha(Path(__file__)),
                 "preprocessing_performed_by_launcher": False}
    create_json(root / "REQUEST.json", execution)
    with (root / "stdout.log").open("x") as stdout, (root / "stderr.log").open("x") as stderr:
        result = subprocess.run(cmd, cwd=root, env=env, stdout=stdout, stderr=stderr)
    if result.returncode:
        create_json(root / "FAILED.json", {"returncode": result.returncode})
        raise RuntimeError(f"MaxFuse failed; partial output retained at {root}")
    if sha(input_path) != input_hash or sha(config_path) != config_hash:
        raise RuntimeError("Input/config changed during run; refusing COMPLETE")
    verify_local(snapshot)
    manifest = json.loads((root / "result/manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("input_sha256") != input_hash:
        raise RuntimeError("Adapter result manifest does not attest to completion/input")
    config_field = "config_file_sha256" if mode == "pairwise" else "pair_config_map_sha256"
    if manifest.get(config_field) != config_hash:
        raise RuntimeError("Adapter configuration mismatch")
    create_json(root / "COMPLETE.json", {**execution, "result_manifest_sha256": sha(root / "result/manifest.json")})
    return root


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, required=True, help="Explicit compatible MaxFuse interpreter; not auto-installed")
    p.add_argument("--mode", choices=["pairwise", "three-view"], required=True)
    p.add_argument("--input", type=Path, required=True, help="Prepared NPZ, NOT shared raw.npy")
    p.add_argument("--config", type=Path, required=True, help="Pairwise config or three-view pair-config map")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="After --: original adapter options")
    args = p.parse_args(argv)
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    print(launch(args.python, args.mode, args.input, args.config, args.output, extra))


if __name__ == "__main__":
    main()
