# envforge

Validation-first compiler for RL environments and evals for language models.
envforge takes a seed, has a builder model write a procedural generator,
solver, and checker, and accepts the result only after it passes a fixed set
of gates: build and reset determinism, reference and independent-solver
agreement, verifier determinism, null-answer rejection, mutation testing,
difficulty profiling against a model panel, and near-duplicate and
contamination checks. Accepted tasks export to TRL reward functions and to
OpenEnv servers.

Status: pre-release (0.1.0, unreleased). The procedural family is implemented
end to end; the tool-world, repository and spreadsheet families are not.

## Quick start

```bash
uv sync
uv run pytest -q
uv run envforge validate tests/fixtures/modsum_good.py --solver tests/fixtures/modsum_solver.py
```

Synthesis runs need model endpoints; see `envforge.example.toml`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Instructions for AI coding agents are
in [AGENTS.md](AGENTS.md).

## License

[MIT](LICENSE)
