# Presentation — Energy Market Intelligence System

The defense deck. Two builds live here; the **HTML deck is the primary one**.

```
presentation/
├── index.html          ★ the deck (bespoke HTML/CSS, 26 slides, 16:9)
├── deck.pdf            ★ rendered PDF (one slide per page, vector, via Chrome)
├── deck.pptx           ★ PowerPoint (one full-bleed slide-image per page)
├── build_pdf.sh          index.html → deck.pdf via headless Chrome
├── build_pptx.py         deck.pdf → deck.pptx (image-per-slide)
├── assets/
│   ├── deck.css          design system (Fraunces + IBM Plex, palette, layout)
│   ├── fonts/            vendored woff2 (self-contained, no CDN)
│   ├── img/             raster charts (price line, missingness heatmap, Module A plots)
│   └── make_charts.py    regenerates the two raster charts from project data
└── (main.tex, figures.py, figures/)   ← earlier LaTeX/Beamer version, kept as an alternative
```

## Design

An editorial "field-report" aesthetic with an industrial edge: warm-paper content slides
punctuated by cinematic full-bleed **dark section dividers**; type pairing of **Fraunces**
(display) + **IBM Plex Sans** (body) + **IBM Plex Mono** (kickers, data tags, tabular
numerals); a per-module accent system (data · imputation · A · B · C) and a recurring
quantile-fan motif. Bar charts and the pipeline / SSHB / dispatch / cascade diagrams are
drawn natively in HTML+CSS+SVG (crisp at any zoom); only the price series and the
missingness heatmap are raster PNGs. Everything is self-contained — **no network, no CDN,
fonts embedded** — so it renders identically offline.

## View it

Open `index.html` in any browser (slides stack vertically), or just open `deck.pdf`.

## Rebuild the PDF

```bash
./build_pdf.sh            # needs Google Chrome / Chromium; writes deck.pdf
```
The page geometry comes from `@page { size: 1280px 720px }` in `deck.css`, so Chrome emits
one perfect 16:9 page per slide with backgrounds and embedded fonts. Override the browser
with `CHROME=/path/to/chrome ./build_pdf.sh`.

## Rebuild the PowerPoint

```bash
python3 build_pptx.py     # deck.pdf → deck.pptx (needs python-pptx + pdftoppm)
```
Because the deck is a bespoke HTML/CSS design (with non-system fonts), the faithful
PowerPoint is **image-based**: each 16:9 slide is the high-resolution render of the
matching `deck.pdf` page, so it looks identical everywhere and needs no fonts installed.
The text is therefore not editable in PowerPoint — edit `index.html` and re-run both build
scripts to change content. (When converting `deck.pptx` back to PDF for QA, use a different
output name so you don't overwrite the crisp Chrome `deck.pdf`.)

## Regenerate the raster charts (optional)

```bash
python3 assets/make_charts.py     # writes assets/img/{price_volatility,missingness_heatmap}.png
```
Reads git-ignored local data (`data/splits/test.parquet`, `data/processed/emis_mask.parquet`);
the Module A plots are copied from `reports/`. matplotlib only.

## Before presenting

- **Author names:** edit the placeholder on the title slide (search `>>> EDIT` in `index.html`).
- **Speaker notes:** keep a separate sheet — the PDF is the visual; notes are not embedded.

## Numbers (canonical · SSHB-imputed run)

Source of truth: `reports/phase7_rerun_summary.md`, `module_b_final_leaderboard.parquet`,
`module_c_*.json`. Headlines: Module B test MAE **23.29 €/MWh** (h1–6, coverage 0.82,
−46% vs. seasonal-naive, DM p<0.001); Module C SAC **€13,054**/episode (> €11,474 perfect-foresight
benchmark; +B = +55%, +A = +€1,412); Module A MAE **2,591 MW**; SSHB engages **53.9%** of imputed cells
(+3 → green, 0 violations). Superseded figures are deliberately not shown.
