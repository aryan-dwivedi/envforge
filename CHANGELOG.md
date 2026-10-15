# Changelog

All notable changes to envforge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic
versioning. Entries are newest-first; the Guards workflow enforces the order.

## [Unreleased]

### Added

- Repository scaffold: agent instructions, contribution and release guides,
  repository guards (clean-for-public sweep, AI co-author trailer check,
  source size ratchet, changelog order) with self-tests, CI, and Dependabot.
- Procedural environment synthesis: a builder model writes a generator module
  (`gen`, `solve`, `check`), an independent solver from a different model is
  written from the problem description alone, and the module is repaired up to
  `max_repairs` times until it passes gates G1-G5 (build and reset
  determinism, reference and independent-solver agreement, verifier
  determinism, null-answer rejection, mutation testing), then G10
  (near-duplicate and eval-suite contamination) and G9 (difficulty band
  against a model panel and the policy).
- Sandbox protocol with a process backend (development only, no network
  isolation) and a Docker backend (`--network none`, dropped capabilities,
  resource limits); generated code runs only through the stdlib-only runner.
- 60 built-in seeds and a content-addressed store of environments, verifiers,
  tasks, validation reports, rejections and gate evidence; rerunning a seed
  that yields the same module is idempotent.
- CLI: `validate` (gates G1-G5 with no model calls), `run`, `stats`,
  `export trl` (dataset plus `make_reward_func` for TRL reward functions),
  `export openenv` (self-contained OpenEnv server package that does not
  import envforge), `audit sample` and `audit ingest`.
