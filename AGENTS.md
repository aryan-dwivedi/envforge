# AGENTS.md - envforge

Agentic coding instructions for envforge, the open-source, validation-first
compiler that turns seeds into verified RL environments and evals for language
models.

## Purpose

Work in this repository as an engineering agent for the synthesis pipeline and
its exporters. Keep changes deterministic, typed, testable without network or
model endpoints, and conservative about what counts as a verified task. A
false accept (a verifier that rewards a wrong answer) is the defect this
project exists to prevent; treat any change that could raise the false-accept
rate as a behavior change that needs a test.

## Read First

- Check `git status --short` before edits and preserve unrelated changes.
- Inspect the nearest existing gate, backend, schema, exporter, or test before
  introducing a new pattern.
- Local agent tooling under `.omp/` is gitignored and may hold working notes.
  Read it when present; never move its content, wording, or section numbers
  into tracked files.
- Keep the no-LLM path working: `envforge validate` and the full test suite run
  with no API keys and no Docker.
- Bump `src/envforge/__init__.py` only when cutting a release (see
  `RELEASE.md`).

## Stack and Commands

- Python 3.11+, uv project, src layout, hatchling build. License MIT.
- Pydantic v2 models for every schema, Typer for the CLI, the `openai` SDK for
  every model role (any OpenAI-compatible endpoint), datasketch for MinHash.
- Version source of truth: `src/envforge/__init__.py`.
- Common commands:

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run envforge --help
```

- Useful focused commands:

```bash
uv run pytest tests/path/to/test_file.py
uv run pytest tests/path/to/test_file.py -k test_name
uv run pytest -m docker -q          # only where the docker CLI exists
uv run ruff check src/envforge/path tests/path
uv run envforge validate tests/fixtures/modsum_good.py --solver tests/fixtures/modsum_solver.py
```

- Repository guards, runnable locally and in CI:

```bash
bash scripts/clean-for-public.sh
python3 scripts/check-file-size.py
python3 scripts/check-changelog-order.py
bash scripts/check-commit-trailers.sh
```

Use the focused commands first when they cover the touched surface. Run the
full suite for changes to schemas, the runner, gate order, or the store.

## Architecture Map

- CLI: `src/envforge/cli.py` (`validate`, `run`, `stats`, `export`, `audit`)
- Config: `src/envforge/config.py`; example surface in `envforge.example.toml`
- Schemas: `src/envforge/schemas.py`
- Model roles: `src/envforge/llm.py` (`ChatModel`, `EmbedModel` protocols)
- Answer format: `src/envforge/answer.py`
- Hashing and ids: `src/envforge/hashing.py`
- Sandbox protocol and backends: `src/envforge/sandbox/` (`process.py`,
  `docker.py`, `harness.py`)
- Sandbox runner, the trust boundary: `src/envforge/sandbox/runner.py`
- Procedural family: `src/envforge/procedural/` (`contract.py`, `mutate.py`,
  `scoring.py`, `gates.py`, `dedup.py`, `profile.py`, `pipeline.py`,
  `package.py`, `seeds.py`, `seeds.toml`)
- Shared gen/score path: `src/envforge/procedural/scoring.py`. Gates,
  profiling, exports and audits all generate instances and score responses
  through `generate` and `score`; do not add a second path.
- Prompt templates: `src/envforge/prompts/`
- Store: `src/envforge/store.py`; run statistics: `src/envforge/stats.py`
- Exporters: `src/envforge/export/` (`trl.py`, `openenv.py`,
  `openenv_template/`)
- Human audit: `src/envforge/audit.py`
- Tests and fixtures: `tests/`, `tests/fixtures/`

Keep code files near 300 lines when practical. Split before modules become
hard to review, except generated files, fixtures, and data tables. Every
`.py` file under `src/`, `tests/` and `scripts/` is held to 500 lines by
`scripts/check-file-size.py`; files already past it are listed in
`.file-size-baseline` and may shrink but never grow. Mark an intentional
exemption from the soft rule with `# Exempt from 300 LOC soft rule: <reason>`.

## Coding Rules

- Type every public function. Schemas are frozen Pydantic models with
  `extra="forbid"`; avoid loose dictionaries where a model belongs.
- Schema field names and gate keys are a published contract consumed by stored
  reports and exports. Do not rename them; add fields instead.
- `sandbox/runner.py` and everything under `export/openenv_template/` stay
  stdlib-only (the template may import `openenv.core.env_server`). Neither may
  import `envforge`.
- Generated code never runs in the host process. It runs through the runner in
  a sandbox fork, and every gate runs on `base.fork()` inside
  `try/finally: fork.destroy()`.
- Seed ranges stay disjoint: probes 0-4, profiling 1000-1007, export 10000+,
  audit 20000+.
- Determinism is a gate, not a hope. Generators use `random.Random(seed)` only;
  ids are content hashes; JSON goes through `canonical_json`.
- Gate thresholds are engineering decisions. Do not loosen one to make a run
  pass; add seeds, fix the prompt, or fix the generator.
- The pipeline depends on the `ChatModel` and `EmbedModel` protocols, never on
  a concrete client, so tests inject scripted doubles.
- Missing configuration fails fast with `ConfigError` and an exact message.
  There are no silent fallbacks between backends or endpoints.

## Sandbox and Trust Boundary

The process backend is for development only: it has no network isolation.
Anything that claims isolation must go through the Docker backend, which runs
with `--network none`, dropped capabilities, and resource limits.

Do not widen `IMPORT_ALLOWLIST` or shrink `BANNED_NAMES` without a test that
proves the new surface cannot read the host, the network, or the verifier's
hidden answer.

Timeouts kill the whole process group or container. A change that lets a hung
generator outlive its budget is a regression.

## Working in the Open

This is a public, open-source repository. Every commit, diff, and branch is
visible the moment it is pushed and stays in history permanently, so a mistake
cannot be un-published by deleting it later. Review what a change actually
contains before committing.

- **Never commit secrets.** API keys, tokens, endpoint credentials, and `.env`
  files stay out of the tree. `envforge.toml` is gitignored because it names
  endpoints and key variables; commit changes to `envforge.example.toml`
  instead. If a secret does land in a commit, treat it as compromised and
  rotate it.
- **Never commit real deployment detail.** Internal hostnames, IP addresses,
  private endpoint URLs, account names, and paths from a developer machine are
  an attack surface. Use placeholders such as `https://llm.example.com/v1`,
  `192.168.1.50`, and `my-org`.
- **Never commit other people's data.** Personal names, email addresses,
  customer or employer names, and raw model transcripts or audit labels that
  contain any of the above do not belong in a public repository.
- **Tests are published too.** Fixtures, scripted model responses, golden
  reports, and sample JSONL get the same care as source.
- **Respect licensing when bringing in outside code or data.** Third-party
  source is vendored into a vendor directory with its license intact and is
  never pasted into our own modules. Seeds and eval suites carry their license
  in provenance.
- **Keep contributions technical.** Architecture, APIs, commands, schemas,
  configuration, gates, and troubleshooting. Commercial, pricing, or roadmap
  commentary does not belong in the codebase.
- **Comments, log strings, commit messages, and PR titles are public too.** Keep
  them bland, factual, and technical. Name the behavior or the principle, never
  an internal planning record or a section of a private document.

## Commits and Changelog

- One logical change per commit. The subject line states the behavior that
  changed, optionally prefixed by the area: `Sandbox: the process backend kills
  the process group on timeout`.
- No AI co-author or `Signed-off-by` trailers naming an AI tool or vendor.
  `scripts/check-commit-trailers.sh` and the Guards workflow reject them.
- Add user-visible changes under `## [Unreleased]` in `CHANGELOG.md`, newest
  first, in Keep a Changelog sections (`Added`, `Changed`, `Fixed`, `Removed`,
  `Security`).

## Verification

- Runner, sandbox backend, or harness change: run `tests/test_sandbox.py` and
  the `validate` smoke on `modsum_good.py` (all gates PASS) and
  `modsum_hang.py` (`G2 FAIL` with `timeout`, bounded wall time).
- Gate, contract, or mutator change: run `tests/test_gates.py`,
  `tests/test_contract.py` and `tests/test_mutate.py`, then the `validate`
  smoke on the good fixture and on the bad fixture that targets the touched
  gate.
- Dedup, contamination, or profiling change: run
  `tests/test_profile_dedup.py`.
- Pipeline, store, or prompt change: run `tests/test_pipeline.py`,
  `tests/test_store_prompts.py` and `tests/test_stats.py`.
- Schema or model-client change: run `tests/test_schemas_llm.py`.
- Exporter or audit change: run `tests/test_export_trl.py`,
  `tests/test_export_openenv.py` and `tests/test_audit.py`; for the OpenEnv
  template, also serve an export with `uvicorn server.app:app` outside the
  envforge environment and step it once with `GenericEnvClient`.
- CLI change: run the matching tests and a bounded `uv run envforge <cmd>
  --help`.
- Docker backend change: `uv run pytest -m docker -q` where Docker exists;
  report it as skipped otherwise.
- Any change: `uv run ruff check .` and the repository guards listed under
  Stack and Commands.

Before finalizing, run `git diff --check` and report any skipped checks.

## Review Expectations

When reviewing, list findings first and focus on false accepts, verifier
nondeterminism, sandbox escapes, imports that reach the host, seed-range
overlap, gate-order changes, loosened thresholds, renamed schema fields,
exports that import `envforge`, cost-accounting gaps, and missing tests. Cite
file and line references.

For implementation work, keep changes scoped to the touched gate, backend, or
exporter and verify the smallest surface that proves the change.

## Cross-Project Impact

- OpenEnv export targets `openenv-core` 0.3 (`Environment.reset(seed,
  episode_id, **kwargs)`, `step(action, timeout_s, **kwargs)`, `create_app`).
  A version change needs the template, its `requirements.txt`, and
  `tests/test_export.py` updated together.
- The TRL reward function relies only on `**kwargs` absorption and the
  completion formats TRL passes. Envforge never imports TRL.
- Stored reports and exported datasets outlive the code that wrote them.
  Schema changes need a migration note in `CHANGELOG.md`.

## Related Public Projects

- [OpenEnv](https://github.com/huggingface/OpenEnv) - environment interface
  the OpenEnv exporter targets.
- [TRL](https://github.com/huggingface/trl) - trainer whose GRPO reward
  function interface the TRL exporter targets.
- [datasketch](https://github.com/ekzhu/datasketch) - MinHash used by the
  near-duplicate gate.
