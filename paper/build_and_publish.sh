#!/usr/bin/env bash
# Rebuild paper/main.pdf and publish it to origin/paper.
#
# Invoked by a Claude Code PostToolUse hook (see .claude/settings.json), which
# pipes the tool-call JSON on stdin. Also safe to run by hand:
#     ./paper/build_and_publish.sh --force
#
# Safety rules, in order:
#   1. Only reacts to edits of paper/*.tex or paper/*.bib (unless --force).
#   2. Only commits/pushes when on the `paper` branch.
#   3. Only commits when main.pdf actually changed.
#   4. Only stages main.pdf — never touches your other work in progress.
#   5. Never force-pushes. A rejected push is reported, not retried.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER_DIR="${REPO_ROOT}/paper"
BRANCH="paper"
LOG="${PAPER_DIR}/.build.log"

# --- 1. Should we run at all? -------------------------------------------------
if [[ "${1:-}" != "--force" ]]; then
  # Hook mode: read the tool payload from stdin and check the edited path.
  payload="$(cat 2>/dev/null || true)"
  edited="$(printf '%s' "$payload" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' \
    2>/dev/null || true)"
  case "$edited" in
    *"/paper/"*.tex|*"/paper/"*.bib) : ;;
    *) exit 0 ;;
  esac
fi

cd "$PAPER_DIR" || exit 0

# --- 2. Build -----------------------------------------------------------------
if ! make pdf >"$LOG" 2>&1; then
  echo "paper: LaTeX build FAILED — see ${LOG}" >&2
  tail -25 "$LOG" >&2
  exit 2
fi

# Surface undefined citations/references even on a successful build.
if grep -qE "Citation .* undefined|Reference .* undefined" main.log 2>/dev/null; then
  echo "paper: build succeeded but has undefined citations/references:" >&2
  grep -E "Citation .* undefined|Reference .* undefined" main.log | sort -u | head -10 >&2
fi

# --- 3. Publish ---------------------------------------------------------------
current="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null)"
if [[ "$current" != "$BRANCH" ]]; then
  echo "paper: built main.pdf (on branch '${current}', not '${BRANCH}' — not publishing)"
  exit 0
fi

git -C "$REPO_ROOT" add -- paper/main.pdf
if git -C "$REPO_ROOT" diff --cached --quiet -- paper/main.pdf; then
  exit 0  # PDF byte-identical; nothing to publish
fi

git -C "$REPO_ROOT" commit -q -m "paper: rebuild PDF [auto]" -- paper/main.pdf || {
  echo "paper: commit failed" >&2; exit 2; }

if git -C "$REPO_ROOT" push -q origin "$BRANCH" 2>>"$LOG"; then
  echo "paper: main.pdf rebuilt and pushed to origin/${BRANCH}"
else
  echo "paper: main.pdf committed locally but PUSH FAILED (pull/rebase needed?) — see ${LOG}" >&2
  exit 2
fi
