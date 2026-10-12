"""Allow ``python -m ramp_proteomics`` to invoke the CLI."""

from .cli import main

raise SystemExit(main())

