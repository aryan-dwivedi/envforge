"""Action and observation models for the envforge procedural environment."""

from __future__ import annotations

from openenv.core.env_server import Action, Observation


class EnvforgeAction(Action):
    answer: str


class EnvforgeObservation(Observation):
    prompt: str = ""
    task_id: str = ""
    feedback: str = ""
