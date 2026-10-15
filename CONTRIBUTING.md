# Contributing to envforge

envforge is licensed under the [MIT License](LICENSE). By contributing, you
agree that your contributions are licensed under the same terms. There is no
contributor licence agreement.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/)
- Git
- Docker, only for the Docker sandbox backend and its `-m docker` tests

## Development Setup

```bash
git clone <repository-url> envforge
cd envforge
uv sync
git config core.hooksPath .githooks
git config blame.ignoreRevsFile .git-blame-ignore-revs
uv run pytest -q
```

The test suite and `envforge validate` need no API keys and no Docker. Real
synthesis runs need model endpoints: copy `envforge.example.toml` to
`envforge.toml` (gitignored), fill in the roles, and export the key variables
it names.

## Before You Open a Pull Request

```bash
uv run ruff check .
uv run pytest -q
bash scripts/clean-for-public.sh
python3 scripts/check-file-size.py
python3 scripts/check-changelog-order.py
git diff --check
```

`AGENTS.md` lists the focused check for each area of the code and the rules
that apply to every change, including changes made with an AI coding agent.

## Pull Request Guidelines

- Fork the repository and branch from `main`: `feat/<short-description>` or
  `fix/<short-description>`.
- Keep each pull request to one feature or fix. For larger changes, open an
  issue first to agree on the approach.
- Add or update tests for new behaviour. A change to a gate, the runner, or a
  verifier needs a test that would have caught the defect it fixes.
- Describe what changed, why, and how you tested it.
- Add an entry under `## [Unreleased]` in `CHANGELOG.md` for user-visible
  changes.
- Do not bump the version; maintainers handle releases (`RELEASE.md`).
- Commit messages carry no AI co-author trailers. The Guards workflow rejects
  them.

## Issue Reporting

Use GitHub Issues for bugs and feature requests.

**For bug reports, include:**
- envforge version (`uv run python -c "import envforge; print(envforge.__version__)"`) and Python version
- Sandbox backend (`process` or `docker`) and operating system
- The command you ran and its full output
- The gate result and the evidence JSON for gate-related bugs, with endpoint
  URLs, keys, and model transcripts that contain personal data removed

**For feature requests, include:**
- The environment family or workflow you are targeting
- How the result would be verified

## Security

Do not open a public issue for a sandbox escape, a verifier bypass, or another
vulnerability. Use GitHub private vulnerability reporting on this repository.

## License

By contributing to envforge, you agree that your contributions will be
licensed under the [MIT License](LICENSE).
