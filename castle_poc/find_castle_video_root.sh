#!/usr/bin/env bash
set -euo pipefail

rels=(
  "main/day1/Werner/video/10.mp4"
  "main/day1/Meeting/video/10.mp4"
  "main/day1/Tien/video/10.mp4"
  "main/day3/Florian/video/17.mp4"
  "main/day3/Cathal/video/17.mp4"
  "main/day3/Meeting/video/17.mp4"
)

bases=(
  "/scratch/xy3257/castle_poc"
  "/scratch/xy3257/castle"
  "/scratch/xy3257/CASTLE"
  "/scratch/xy3257/castle_dataset"
  "/scratch/xy3257/castle_data"
  "/scratch/xy3257"
  "/vast/xy3257"
  "/archive/xy3257"
  "/scratch/work/public"
  "/vast/work/public"
)

echo "Checking explicit candidate roots..."
for base in "${bases[@]}"; do
  [[ -d "$base" ]] || continue
  hits=0
  for rel in "${rels[@]}"; do
    if [[ -f "$base/$rel" ]]; then
      hits=$((hits+1))
    fi
  done
  printf "%-45s hits=%s\n" "$base" "$hits"
  if [[ "$hits" -gt 0 ]]; then
    echo "  Example files:"
    for rel in "${rels[@]}"; do
      [[ -f "$base/$rel" ]] && echo "  $base/$rel"
    done
  fi
done

echo
echo "Searching shallowly for directories named main/day1..."
for base in "${bases[@]}"; do
  [[ -d "$base" ]] || continue
  echo "== $base =="
  timeout 20 find "$base" -maxdepth 6 -type d -path "*/main/day1" 2>/dev/null | head -10 || true
done
