#!/usr/bin/env bash
# Download the MemGate checkpoint into the path from configs/MemZero+MemGate.json
# (gitignored; default: output_data/checkpoints/Memgate.pt).
#
#   ./scripts/fetch_memgate_checkpoint.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_JSON="$REPO_ROOT/baselines/configs/MemZero+MemGate.json"
DEST="$(python3 - "$REPO_ROOT" "$CONFIG_JSON" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
data = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
raw = ((data.get("control") or {}).get("checkpoint") or "output_data/checkpoints/Memgate.pt")
path = Path(raw)
print(path if path.is_absolute() else root / path)
PY
)"
DEST_DIR="$(dirname "$DEST")"
URL="https://github.com/Kevin-Zh-CS/MemGate/raw/05f89d003854e99523b2c74df54db7b0d2fa2643/checkpoints/Memgate.pt"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST" ]] && [[ "$(stat -c%s "$DEST" 2>/dev/null || echo 0)" -gt 1000000 ]]; then
  echo "already present: $DEST ($(du -h "$DEST" | cut -f1))"
  exit 0
fi

echo "downloading $URL"
curl -L --fail --retry 3 -o "$DEST" "$URL"
echo "MemGate checkpoint: $DEST ($(du -h "$DEST" | cut -f1))"
