#!/usr/bin/env bash
# Render the HTML deck to a pixel-perfect 16:9 PDF (one slide per page) with headless Chrome.
# The @page size in deck.css (1280x720 px) drives the page geometry; Chrome embeds the fonts.
#   ./build_pdf.sh        # -> presentation/deck.pdf
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
[ -x "$CHROME" ] || CHROME="$(command -v google-chrome || command -v chromium || true)"
[ -n "$CHROME" ] || { echo "Chrome not found; set \$CHROME"; exit 1; }

"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
  --run-all-compositor-stages-before-draw --virtual-time-budget=10000 \
  --print-to-pdf="$DIR/deck.pdf" "file://$DIR/index.html"
echo "wrote $DIR/deck.pdf"
