#!/usr/bin/env bash
# Re-render all DMML lecture animations at high resolution and (optionally)
# regenerate the Canvas GIFs from the fresh masters.
#
# Usage:
#   ./render_all.sh                 # 4K60 everything + refresh GIFs
#   ./render_all.sh -q p            # 1440p60 instead of 4K
#   ./render_all.sh --no-gif        # skip GIF regeneration
#   ./render_all.sh w08 w09 w10     # only these week folders
#
# Quality codes:  k = 4K60 (2160p60, default) | p = 1440p60 | h = 1080p60
#
# Heavy job: 30 scenes at 4K60 can take a long time. Run in the background
# (`./render_all.sh &> render.log &`) or submit it to the cluster.
set -uo pipefail
cd "$(dirname "$0")"                       # -> animations/
MANIM="$(cd .. && pwd)/env/bin/manim"      # absolute: survives the cd into each week dir

Q=k
DO_GIF=1
GIF_WIDTH=1000
GIF_FPS=15
WEEKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -q) Q="$2"; shift 2;;
    --no-gif) DO_GIF=0; shift;;
    w*) WEEKS+=("$1"); shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

case "$Q" in
  k) QFLAG=-qk; RES=2160p60;;
  p) QFLAG=-qp; RES=1440p60;;
  h) QFLAG=-qh; RES=1080p60;;
  *) echo "bad quality: $Q (use k|p|h)" >&2; exit 2;;
esac

# Which week folders to process
if [[ ${#WEEKS[@]} -eq 0 ]]; then
  mapfile -t DIRS < <(find . -maxdepth 1 -type d -name 'w[0-9][0-9]_*' | sort)
else
  DIRS=(); for w in "${WEEKS[@]}"; do DIRS+=("$(find . -maxdepth 1 -type d -name "${w}_*")"); done
fi

fail=0
for dir in "${DIRS[@]}"; do
  [[ -d "$dir" ]] || continue
  for py in "$dir"/*.py; do
    [[ -e "$py" ]] || continue
    stem=$(basename "$py" .py)
    # discover every Scene subclass in the file
    scenes=$(grep -oE 'class[[:space:]]+[A-Za-z0-9_]+\([^)]*Scene[^)]*\)' "$py" \
             | sed -E 's/class[[:space:]]+([A-Za-z0-9_]+).*/\1/')
    for scene in $scenes; do
      echo ">>> [$Q] $dir/$stem :: $scene"
      ( cd "$dir" && "$MANIM" $QFLAG --format mp4 "$(basename "$py")" "$scene" ) || { fail=1; continue; }
      mp4="$dir/media/videos/$stem/$RES/$scene.mp4"
      if [[ ! -f "$mp4" ]]; then
        echo "    WARN: expected mp4 not found: $mp4" >&2; fail=1; continue
      fi
      # Refresh the Canvas GIF from the fresh master (before moving it).
      if [[ $DO_GIF -eq 1 ]]; then
        gif="$dir/$scene.gif"
        ffmpeg -y -loglevel error -i "$mp4" \
          -vf "fps=$GIF_FPS,scale=$GIF_WIDTH:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
          "$gif" && echo "    gif -> $gif"
      fi
      # Collect the final master into renders/ (week-prefixed, easy to grab for slides).
      wk=$(basename "$dir" | cut -d_ -f1)
      mkdir -p renders
      mv -f "$mp4" "renders/${wk}_${scene}.mp4" && echo "    master -> renders/${wk}_${scene}.mp4"
    done
  done
done

[[ $fail -eq 0 ]] && echo "ALL DONE ($Q)" || { echo "FINISHED WITH ERRORS ($Q)"; exit 1; }
