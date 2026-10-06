#!/usr/bin/env bash
# Download the quantized checkpoints listed in artifacts/MANIFEST.tsv and verify their SHA-256.
#   bash tools/fetch_artifacts.sh [--only REGEX]
# Downloads from https://huggingface.co/chahh9808/QuAR-checkpoints/resolve/main
# by default; set ARTIFACT_BASE_URL to pull from somewhere else.
# Files are placed at the manifest path (checkpoints/...). Rows whose source starts with EXTERNAL
# are not hosted by us: the script prints where to get them (AdaLog release, Hugging Face) instead.
# Existing files with a matching checksum are skipped, so the script is resumable.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
ONLY=".*"; [ "${1:-}" = "--only" ] && ONLY="$2"
BASE="${ARTIFACT_BASE_URL:-https://huggingface.co/chahh9808/QuAR-checkpoints/resolve/main}"
ok=0; skipped=0; ext=0; failed=0
while IFS=$'\t' read -r path bytes sha kind src used; do
  [ "$path" = "path" ] && continue; [[ "$path" =~ $ONLY ]] || continue
  if [ -f "$path" ] && [ "$(sha256sum "$path" | cut -d' ' -f1)" = "$sha" ]; then echo "ok       $path"; ok=$((ok+1)); continue; fi
  if [[ "$src" == EXTERNAL* ]]; then echo "external $path  <- ${src#EXTERNAL: }"; ext=$((ext+1)); continue; fi
  if [ -z "$BASE" ]; then echo "missing  $path  (set ARTIFACT_BASE_URL to download; or rebuild: $src)"; skipped=$((skipped+1)); continue; fi
  mkdir -p "$(dirname "$path")"; echo "fetch    $path ($((bytes/1048576)) MB)"
  if curl -fL --retry 3 -o "$path.part" "$BASE/$(basename "$path")" && [ "$(sha256sum "$path.part" | cut -d' ' -f1)" = "$sha" ]; then mv "$path.part" "$path"; ok=$((ok+1)); else echo "FAILED   $path (download or checksum)"; rm -f "$path.part"; failed=$((failed+1)); fi
done < artifacts/MANIFEST.tsv
echo "done: $ok present, $ext external, $skipped not downloaded, $failed failed"
[ "$failed" = 0 ]
