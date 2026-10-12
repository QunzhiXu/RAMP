# Examples

Run the in-memory synthetic example after installing the package:

```bash
python examples/quickstart.py
```

To exercise the complete file interface and donor metadata contract:

```bash
python examples/make_file_inputs.py
ramp-proteomics align \
  --raw examples/output/input/raw.npy \
  --metadata examples/output/input/cell_metadata.csv.gz \
  --markers examples/output/input/markers.txt \
  --datasets CODEX CyTOF \
  --family synthetic-file-example-v1 \
  --output examples/output/ramp
```

Both examples are deterministic and contain no provider or patient data.
