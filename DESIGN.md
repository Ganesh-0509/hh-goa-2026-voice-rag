# Design

<!-- impeccable:design-doc built-from-code -->

Built world for `web/index.html`, the single-page voice/text RAG demo UI. Recorded from the shipped implementation, not intention.

## World

Brief-pinned direction: a Goa-beach flat-illustration travel-poster, from a user-supplied reference image. Deep emerald ground, sun-gold rays, sand-white content surfaces, one hot-pink accent reserved for the primary action and the "guardrail fired" state. Flat shapes only — no photography, no gradients standing in for material.

## Color strategy: Full palette (4 named roles)

| Role | Token | Hex | Used for |
|---|---|---|---|
| Ground | `--green-deep` / `--green-mid` | `#0B4A38` / `#14634C` | Horizon band, footer, headings on dark |
| Accent (primary action) | `--gold` / `--gold-deep` | `#F6C90E` / `#C99400` | Sun mark, Live Demo pill, recording-state pulse, dense-search metric |
| Accent (call to action / notable state) | `--pink` / `--pink-deep` / `--pink-pale` | `#E8195C` / `#A80F44` / `#FCE4EC` | Record button, task badge, abstained status + answer tint |
| Surface | `--cream` / `--sand` | `#FBF8EF` / `#FFFFFF` | Page ground / panel surfaces |
| Ink | `--ink` / `--ink-soft` | `#0A2E22` / `#3D5A4E` | Body text, secondary text (tinted from the palette, never gray) |

Light/dark was decided by scene: a judge scans dense multilingual data on a laptop, so content panels are light (sand-white, like the poster's sand) for legibility; the brand shell (header/footer) carries the poster's dominant deep green, mirroring the reference's own light/dark balance.

## Type

- Display / brand voice: **Syne** (700/800) — headings, panel titles, metric numbers.
- Body / UI / data: **IBM Plex Sans** (400–700) — controls, labels, transcripts, citations, latency table. Chosen partly for its Devanagari-adjacent technical character and because Space Grotesk/Plus Jakarta Sans/Inter were flagged by the detector as overused AI-default faces.
- Numerals use `font-variant-numeric: tabular-nums` throughout the latency table and metric cards.

## Motifs (hand-drawn SVG, not photography)

- Radiating sun mark (header brand lockup).
- Three flat two-tone palm silhouettes anchoring the horizon band, cropped at the viewport edges (matches the reference's panoramic crop).
- A single wavy line at the horizon/panel seam, echoing the reference's shore squiggles — deliberately partial/glimpsed, not a continuous divider.
- Citation cards carry a small gold diamond marker instead of a colored `border-left` (the latter is a recognized AI-slop tell — see `.claude/skills/impeccable/reference/craft-floor.md`).

## Components

- **Panels**: white/cream rounded cards (20px radius), independent height per column (`align-items: start` on the grid — panels are not stretched to match each other).
- **Primary action** (record button): pink, bold, full-width; recording state shifts to gold with a pulse animation (same hue family as idle → cohesive, not an unrelated 5th color).
- **Status badge**: five states (ready / recording / processing / success / abstained). Abstained deliberately shares the pink family with recording rather than a red "error" color — it's the guardrail working, not a failure state, and the answer box tints to match.
- **Citations**: cream cards, gold diamond marker, RRF score in deep green.
- **Latency**: 3 headline metric cards (rounded to 1 decimal, `clamp()`-sized to avoid wrapping) + a full-precision stage-by-stage table below.

## Motion

One authored moment: the record button's pulse (idle→recording), plus a slow blink on the header's live-status dot. No scroll-triggered reveals — this is a single-viewport operate surface, not a scrolling narrative. Respects `prefers-reduced-motion`.

## States verified

Empty/initial, grounded-answer (real multilingual query), abstained (off-topic query) — captured at desktop (1440×900) and mobile (390×844) via Playwright screenshots in `.impeccable/review/`.

## What a future surface in this world should reuse

Panel shell, status-badge state system, the pink/gold hue-family split (pink = primary action + "notable" state, gold = accent + secondary success), Syne/IBM Plex Sans pairing, and the "abstain reads as confidence, not error" principle — this is a durable product principle (see `PRODUCT.md`), not just a color choice.
