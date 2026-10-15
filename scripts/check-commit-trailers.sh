#!/usr/bin/env bash
# Reject commit messages that credit an AI tool as co-author or signer, or that
# carry a tool attribution line. GitHub maps a vendor noreply address to a bot
# account, which then appears in this repository's contributor list
# permanently; an attribution line is the same leak in prose.
#
#   bash scripts/check-commit-trailers.sh                    # unpushed commits
#   bash scripts/check-commit-trailers.sh BASE..HEAD         # a range (any git-log args)
#   bash scripts/check-commit-trailers.sh --message-file F   # one message (commit-msg hook)
#   bash scripts/check-commit-trailers.sh --self-test
set -uo pipefail

export LC_ALL=C
TRAILER='(Co-[Aa]uthored-[Bb]y|Signed-off-by):.*(claude|anthropic|copilot|openai|chatgpt|cursor|codex|gemini)'
ATTRIBUTION='Generated (with|by) .*(claude|copilot|chatgpt|cursor|codex|gemini)'

matches() {
  grep -inE "$TRAILER|$ATTRIBUTION"
}

case "${1:-}" in
  --self-test)
    fail=0
    # Samples are synthesised so this file never carries a real trailer.
    if ! printf 'x\n\nCo-Authored-By: Bot <noreply@%s>\n' 'anthropic.com' | matches >/dev/null; then
      echo "SELF-TEST FAILED: the trailer pattern does not match a known co-author trailer."; fail=1
    fi
    if ! printf 'x\n\n%s with [%s Code](https://example.com)\n' 'Generated' 'Claude' | matches >/dev/null; then
      echo "SELF-TEST FAILED: the attribution pattern does not match a known attribution line."; fail=1
    fi
    if printf 'LLM: openai SDK client retries on 429\n\nCo-authored-by: Dev <dev@example.com>\n' | matches >/dev/null; then
      echo "SELF-TEST FAILED: a human co-author or a technical mention of a vendor SDK is rejected."; fail=1
    fi
    [ "$fail" -eq 0 ] && echo "PASS: commit-trailer self-test."
    exit "$fail"
    ;;
  --message-file)
    # Comment lines are stripped by git before the commit is stored.
    if grep -v '^#' "$2" | matches; then
      echo "FAIL: this commit message credits an AI tool (see above). Remove the trailer or attribution line."
      exit 1
    fi
    exit 0
    ;;
esac

if [ "$#" -gt 0 ]; then
  range=("$@")
else
  # Everything not yet on a remote branch. With no remotes this is every commit.
  range=(HEAD --not --remotes)
fi

if ! git rev-parse --verify -q HEAD >/dev/null; then
  echo "PASS: no commits yet."
  exit 0
fi

msgs="$(git log --format='%H %s%n%b%n' "${range[@]}")" || { echo "FAIL: git log ${range[*]} failed."; exit 2; }
if printf '%s' "$msgs" | matches; then
  echo "FAIL: a commit in ${range[*]} credits an AI tool (see above)."
  exit 1
fi
echo "PASS: no AI co-author trailer or attribution line in ${range[*]}."
