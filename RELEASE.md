# Cutting a release

envforge releases are GitHub releases built from a `v<version>` tag. The
release workflow refuses to publish unless the tag equals the package version,
the checks pass, and the clean-for-public sweep passes. It attaches the wheel
and sdist and uses the matching `CHANGELOG.md` section as the release notes.

## Steps

1. Bump `__version__` in `src/envforge/__init__.py`, the single source of
   truth. `pyproject.toml` reads it back through `[tool.hatch.version]`.
2. Rename `## [Unreleased]` in `CHANGELOG.md` to `## [<version>] - <YYYY-MM-DD>`
   and open a new empty `## [Unreleased]` above it.
3. Commit as `Release <version>` and push to `main`. Wait for CI and Guards to
   pass on that commit.
4. Tag and push the tag:

   ```bash
   V=$(uv run python -c "import envforge; print(envforge.__version__)")
   git tag "v$V"
   git push origin "v$V"
   ```

## Why tagging is manual

A tag pushed by a workflow using the default Actions token does not trigger
other workflows, so an automatic tag would publish nothing. Tagging from a
developer checkout (or any client with a real token) triggers the release
workflow normally.
