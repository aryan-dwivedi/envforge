"""FastAPI app; run with ``uvicorn server.app:app`` from the package root."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import EnvforgeAction, EnvforgeObservation  # noqa: E402
from openenv.core.env_server import create_app  # noqa: E402

from server.environment import EnvforgeProceduralEnv  # noqa: E402

app = create_app(EnvforgeProceduralEnv, EnvforgeAction, EnvforgeObservation, env_name="envforge_procedural")
