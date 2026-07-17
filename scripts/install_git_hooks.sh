#!/usr/bin/env bash
# Remove Cursor/agent co-author trailers from commit messages
MSG_FILE="$1"
if [[ ! -f "$MSG_FILE" ]]; then
  exit 0
fi
python3 - "$MSG_FILE" <<'PY'
import sys
path = sys.argv[1]
lines = open(path, encoding="utf-8").read().splitlines()
blocked = ("co-authored-by:", "made-with:", "made with cursor", "cursor agent")
clean = [ln for ln in lines if not any(b in ln.lower() for b in blocked)]
open(path, "w", encoding="utf-8").write("\n".join(clean).rstrip() + "\n")
PY
