#!/usr/bin/env bash
# Clean-for-public sweep. This repository is public, so no internal-planning
# reference, AI attribution, or developer-machine path may land in code,
# fixtures, docs, or configuration. The sweep greps the whole working tree and
# exits non-zero on any hit. It runs in CI, in the pre-push hook, and on demand.
#
#   bash scripts/clean-for-public.sh              # sweep the tree
#   bash scripts/clean-for-public.sh --self-test  # prove every matcher fires
#
# This file is PUBLIC, so it holds only GENERIC structural patterns. Identity,
# employer, customer, and other sensitive literals are never written here; they
# are loaded at run time from a private, gitignored denylist (default
# `.private/denylist.txt`, or the path in $ENVFORGE_DENYLIST_FILE), one
# extended regex per line, blank lines and # comments ignored. Hits on those
# patterns print file and line only, never the matched text, so a CI log cannot
# republish the term it caught.
#
# A guard that cannot fail is worse than no guard, because it is read as proof.
# Every sample in the self-test is assembled at run time: writing a literal
# sample into this file would make the file its own first hit. The file is not
# excluded from its own sweep, because an exclusion is a hole a real leak could
# sit in.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2
export LC_ALL=C

# An internal planning record number, a numbered internal rule, a batch label,
# or a section reference into a private document. Each points a public reader
# at something they cannot see; describe the change or name the principle.
# No \b anywhere: ERE grep implementations disagree on it, and a pattern that
# silently matches nothing reports a clean tree.
TAG_PATTERN='(^|[^A-Za-z])(DEC|MSN|LRN)-[0-9]{2,}'
RULE_PATTERN='(^|[^A-Za-z])[Rr]ule #?[0-9]+'
WAVE_PATTERN='(^|[^A-Za-z])Wave [0-9A-Z]( |$)'
SECTION_PATTERN='§[A-Z0-9]'
# Tool attribution lines that coding agents append to files and PR bodies.
ATTRIBUTION_PATTERN='Generated (with|by) \[?(Claude|Copilot|ChatGPT|Cursor|Codex|Gemini)'
# An absolute path under a home directory is a developer machine baked into
# shipped source. Obvious placeholders and CI runner accounts are allowed.
HOME_PATTERN='/(Users|home)/[A-Za-z][A-Za-z0-9_.-]*/'
HOME_ALLOW='/(Users|home)/(tester|test|user|username|you|me|someone|developer|dev|example|alice|bob|runner|ubuntu|root)/'

STRUCTURAL=("$TAG_PATTERN" "$RULE_PATTERN" "$WAVE_PATTERN" "$SECTION_PATTERN" "$ATTRIBUTION_PATTERN")

# Directories that are build output, local environments, or never published.
# `.private` holds the denylist itself and `.omp` is the gitignored local
# agent-tooling directory; a hit in either is noise, not a leak.
scan() {
  # scan ROOT PATTERN [extra grep flags] -> file:line:text for every hit
  grep -rnEI ${3:-} \
    --exclude-dir=.git --exclude-dir=.venv --exclude-dir=.omp --exclude-dir=.private \
    --exclude-dir=store --exclude-dir=dist --exclude-dir=build --exclude-dir=node_modules \
    --exclude-dir=__pycache__ --exclude-dir=.pytest_cache --exclude-dir=.ruff_cache \
    --exclude=uv.lock --exclude=.DS_Store \
    -- "$2" "$1" 2>/dev/null || true
}

scan_home() {
  scan "$1" "$HOME_PATTERN" | grep -vE "$HOME_ALLOW" || true
}

self_test() {
  local dir fail=0 i
  dir="$(mktemp -d)"
  # One sample per structural pattern, in the same order as STRUCTURAL.
  local samples=(
    "$(printf 'see %s%s%s for why' 'DEC' '-' '284')"
    "$(printf 'per %s %s' 'Rule' '44')"
    "$(printf 'landed in %s %s of the work' 'Wave' '3')"
    "$(printf 'as in %s%s of the plan' '§' 'C')"
    "$(printf '%s with [%s Code]' 'Generated' 'Claude')"
  )
  for i in "${!STRUCTURAL[@]}"; do
    printf '%s\n' "${samples[$i]}" > "$dir/sample.txt"
    if [ -z "$(scan "$dir" "${STRUCTURAL[$i]}")" ]; then
      echo "SELF-TEST FAILED: pattern $i does not match its known sample; this sweep would report a clean tree."
      fail=1
    fi
  done
  printf 'path %s\n' "$(printf '/Users/%s/src/x' 'realname')" > "$dir/sample.txt"
  if [ -z "$(scan_home "$dir")" ]; then
    echo "SELF-TEST FAILED: the home-path pattern does not match a known absolute home path."
    fail=1
  fi
  printf 'path %s\n' "$(printf '/home/%s/work' 'runner')" > "$dir/sample.txt"
  if [ -n "$(scan_home "$dir")" ]; then
    echo "SELF-TEST FAILED: an allowlisted placeholder home path is reported as a leak."
    fail=1
  fi
  printf 'G4 rejects null answers; see the rules in AGENTS.md.\n' > "$dir/sample.txt"
  for i in "${!STRUCTURAL[@]}"; do
    if [ -n "$(scan "$dir" "${STRUCTURAL[$i]}")" ]; then
      echo "SELF-TEST FAILED: pattern $i matches ordinary technical prose."
      fail=1
    fi
  done
  rm -rf "$dir"
  [ "$fail" -eq 0 ] && echo "PASS: clean-for-public self-test (${#STRUCTURAL[@]} structural patterns, home path, allowlist, clean prose)."
  return "$fail"
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

found=0
for p in "${STRUCTURAL[@]}"; do
  hits="$(scan . "$p")"
  if [ -n "$hits" ]; then
    echo "clean-for-public: forbidden pattern found:"
    printf '%s\n' "$hits"
    found=1
  fi
done
hits="$(scan_home .)"
if [ -n "$hits" ]; then
  echo "clean-for-public: absolute home-directory path found (use a relative path, an env var, or a placeholder name):"
  printf '%s\n' "$hits"
  found=1
fi

denylist_file="${ENVFORGE_DENYLIST_FILE:-.private/denylist.txt}"
denylist_count=0
if [ -f "$denylist_file" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    denylist_count=$((denylist_count + 1))
    # Names are matched in any case, and only file:line is printed.
    hits="$(scan . "$line" -i | cut -d: -f1,2)"
    if [ -n "$hits" ]; then
      echo "clean-for-public: named denylist pattern #$denylist_count found at:"
      printf '%s\n' "$hits" | sort -u
      found=1
    fi
  done < "$denylist_file"
fi
if [ "$denylist_count" -eq 0 ] && [ "${ENVFORGE_DENYLIST_REQUIRED:-0}" = "1" ]; then
  echo "clean-for-public: FAIL - named denylist missing or empty at '$denylist_file'."
  exit 2
fi

if [ "$found" -ne 0 ]; then
  echo
  echo "clean-for-public sweep FAILED. Remove the content above before pushing."
  exit 1
fi

if [ "$denylist_count" -gt 0 ]; then
  echo "PASS: clean-for-public sweep (structural + $denylist_count named pattern(s))."
else
  # Never print a full pass when the named half did not run.
  echo "PASS (partial): clean-for-public sweep, structural patterns only; no named denylist at '$denylist_file'."
fi
