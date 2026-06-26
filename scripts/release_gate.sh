#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "RELEASE GATE: FAIL"
  echo "- synthetic-panel must be inside a git work tree so tracked files can be audited"
  exit 1
fi

mapfile -t tracked_files < <(git ls-files -- .)
if ((${#tracked_files[@]} == 0)); then
  echo "RELEASE GATE: FAIL"
  echo "- no tracked files found under synthetic-panel/"
  exit 1
fi

failures=0

echo "Checking tracked tree for prohibited terms..."
brand_term="startup""bros"
niche_phrase="side"" business"
offer_phrase="course"" or ""membership"
responses_phrase="survey"" responses"
survey_count="11"",240"
name_one="mi""ke"
name_two="sar""ah"
name_three="rob""ert"

term_pattern="${brand_term}|${niche_phrase}|${offer_phrase}|${responses_phrase}|${survey_count}"
name_pattern="\\<(${name_one}|${name_two}|${name_three})\\>"
term_hits="$({
  git grep -nIi -E "$term_pattern" -- . || true
  git grep -nIi -E "$name_pattern" -- . || true
} | sort -u)"

if [[ -n "$term_hits" ]]; then
  echo "Prohibited term hits:"
  echo "$term_hits"
  failures=1
fi

echo "Checking tracked tree for data blobs..."
blob_hits=""
for path in "${tracked_files[@]}"; do
  case "$path" in
    *.gz|*.csv|*.db|calibration_data/*|*/calibration_data/*)
      blob_hits+="${path}"$'\n'
      ;;
  esac
done

if [[ -n "$blob_hits" ]]; then
  echo "Tracked data blob hits:"
  printf '%s' "$blob_hits" | sort -u
  failures=1
fi

if ((failures)); then
  echo "RELEASE GATE: FAIL"
  exit 1
fi

echo "RELEASE GATE: CLEAN"
