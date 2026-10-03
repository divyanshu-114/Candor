#!/usr/bin/env bash
# Copy the LLM response cache to .cache/llm_backup_<timestamp>/ and print its size.
# Never deletes anything. For a cold measurement set LLM_CACHE_DIR to a new dir instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SRC="${LLM_CACHE_DIR:-.cache/llm}"
if [ ! -d "$SRC" ]; then
  echo "No cache at $SRC yet: nothing to back up (0 files, 0 bytes)."
  exit 0
fi
DEST=".cache/llm_backup_$(date +%Y%m%d_%H%M%S)"
cp -R "$SRC" "$DEST"
echo "Backed up $(find "$DEST" -type f | wc -l | tr -d ' ') files, $(du -sh "$DEST" | cut -f1) -> $DEST"
