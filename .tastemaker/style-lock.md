# Style lock — T3 medical imaging workbench

Locked once, reused for every later change in this project. Generated with the
tastemaker palette generator, verified with its contrast matrix.

## Direction contract

- **Thesis**: a clinical instrument, not a marketing page. The screen exists to
  let a reviewer read one case: what the agent decided, on what evidence, and
  how it called its tools.
- **First viewport**: the answer panel and the verdict strip. The tool journal
  sits in the right rail, the operating point on a single rail above it.
- **System**: one dark technical surface family (bg → surface → border) with a
  single cyan-blue primary for state and a single warm amber accent reserved
  for caution (refusal, suspect Grad-CAM, forced tool calls).
- **Risk**: dark UI on a projector washes out; every pairing was solved against
  the 4.5:1 text floor and 3.0:1 UI floor, so no information is carried by
  colour alone.

## Color contract

| role | hex |
|---|---|
| text | #e6f5fb |
| bg | #0a1114 |
| surface | #151d1f |
| primary | #0681a0 |
| on-primary | #ffffff |
| secondary | #0e333f |
| accent | #e28158 |
| border | #21292c |

**Text-safe (≥ 4.5:1)** — text/bg, text/surface, text/border, bg/accent,
surface/accent, accent/border, primary/on-primary, bg/on-primary, surface/on-primary.

**UI-safe (≥ 3.0:1)** — bg/primary, text/primary, surface/primary, primary/border.

**Decorative (< 3.0:1)** — accent/on-primary, text/accent, primary/accent,
bg/border, surface/border, text/on-primary, bg/surface.
Hairlines and tints only. Never the sole carrier of state: every caution is
also written in words ("refused", "Grad-CAM falls outside the thorax mask").

## Type pairing

One sans for the interface, one mono for values and identifiers.
No third family, no italics, no display serif — the content is a report.

- UI sans: Inter → Segoe UI → system-ui → -apple-system → sans-serif
- Mono (numbers, tool names, JSON keys): JetBrains Mono → Cascadia Mono →
  Consolas → ui-monospace → monospace

Sizes: masthead 1.32rem / 620, section labels 0.68rem mono uppercase 0.1em
tracking, body 0.95rem / 1.62 line height, values 0.82rem mono.

## Density & spacing

Scale: 4 / 8 / 12 / 16 / 24 / 32 / 48 (--t3-s1 … --t3-s7).
Internal padding is never larger than the gap to the neighbour: panels use 24px
inside and 16px between; the right rail steps are 8px inside / 12px column gap.
Section rhythm on this single screen is 32px below the masthead, 24px between
blocks.

## Shape language

4px radius on panels and controls, 3px on chips and icons. One hairline
border, one 3px accent rule on the answer panel. No shadows, no gradients, no
rounded-pill buttons.

## Motion (app-shell track)

Interface only: 140ms ease on button hover/press, no scroll choreography
(nothing to scroll in an app shell), everything else static. `prefers-reduced-motion`
kills the transitions.

## Assets

Icons are real Lucide SVGs in `t3_agentic/ui/assets/icons/`, tinted to the
primary, inlined by `theme.icon()`. No emoji as icons. No photography: this
screen shows data, not mood.

## Mode

Active aesthetic mode: none (the skill's optional mode files are not installed
in this environment; the base rules above apply).
