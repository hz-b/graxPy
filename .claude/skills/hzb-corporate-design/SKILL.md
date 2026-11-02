---
name: hzb-corporate-design
description: >
  Reference for the Helmholtz-Zentrum Berlin (HZB) 2021 "CD Re-Fresh" corporate
  design — logo rules, the HZB color palette (incl. BESSY II's lime accent),
  Source Sans Pro typography, the 6-column grid/keyvisual system, and photo and
  illustration style. Use this whenever producing anything that should look
  like an official HZB or BESSY II deliverable: reports, one-pagers, slide
  decks, posters, web pages/dashboards, or print layouts for HZB, BESSY II, or
  the HZB Schülerlabor. Trigger even if the user just says "make this look
  official / on-brand / like HZB" or "for BESSY II" without naming "corporate
  design" explicitly — apply the colors, type, and layout rules below rather
  than guessing at a generic scientific-institute look.
---

# HZB Corporate Design (2021 Re-Fresh)

This is Helmholtz-Zentrum Berlin's actual, current brand system (effective
October 2021), not a generic "make it look sciency" placeholder. When a task
calls for an HZB- or BESSY II-branded output, pull real values from this file
— colors, font, and spacing rules — rather than inventing a blue-and-white
scheme from general intuition. The specific numbers matter: HZB deliverables
get reviewed against this manual, and a `#1E5AA8` guess where the real value
is `#184e8b` reads as visibly off-brand to anyone familiar with the system.

Two constraints that shape how far to take this:
- Everything below is styling guidance for *your own generated output*
  (documents, artifacts, decks, mockups). It is not permission to reproduce,
  redistribute, or claim authorship of HZB's actual logo files, illustrations,
  or InDesign templates — those are proprietary assets held by HZB's KOM
  (communications) department. If a task needs the literal logo file, say so
  and point the user to their internal HZB archive instead of fabricating one.
- The manual repeatedly notes that final publications must be coordinated
  with and approved by KOM before release. Mention this once, plainly, when
  you produce something that reads as an official/external deliverable —
  don't withhold the work over it, just don't imply you've replaced that
  sign-off step.

## Color palette

Prefer hex/RGB for anything digital (web pages, artifacts, slides); use the
CMYK/Pantone values only if the deliverable is destined for print.

| Role | Hex | RGB | CMYK | Pantone |
|---|---|---|---|---|
| HZB Basisfarbe 01 (primary blue) | `#184e8b` | 24, 78, 139 | 100/50/0/35 | 7692 C |
| HZB Co-Farbe 02 (HZB Cyan, secondary) | `#6cabe9` | 108, 171, 233 | 80/20/0/0 | 2925 C |
| HZB Aktionsfarbe (accent, coral-red) | `#d15e57` | 209, 94, 87 | 0/80/65/0 | — |

Secondary palette (use sparingly, for accents/data viz, not as a primary brand color):

| Name | Hex (approx.) | RGB | CMYK |
|---|---|---|---|
| Hellgrün (light green) | `#c6d970` | 198, 217, 112 | 35/0/75/0 |
| Dunkelblau (dark blue) | `#093566` | 9, 53, 102 | 100/50/0/60 |
| Schülerlabor orange | `#cb5a2e` | 203, 90, 46 | 0/80/95/0 |
| Hellblau (light blue) | `#9ac6f3` | 154, 198, 243 | 55/0/0/0 |

Special case: **BESSY II** (the storage ring) uses a lime green as its
co-color instead of HZB Cyan — CMYK 40/0/100/0. If a task is specifically
about BESSY II (not HZB generally), lead with this lime accent alongside HZB
Blue rather than the standard Cyan, matching the manual's "HZB-3-Sprung"
color-coding (see below).

For grayscale/tint work (fills, subtle backgrounds, chart secondary series),
the manual uses black tints: 12%, 30%, 50%, 70% K. A 12% K gray is explicitly
sanctioned as a background for complex scientific graphics that need to stay
on a near-white field.

Recommended graphic/data-viz accent colors (from the manual's "grafische
Elemente" section, each paired with a black-tint counterpart for variety):

- C28 M0 Y100 K0 (yellow-green) ↔ 12% K
- C0 M50 Y0 K5 (magenta-pink) ↔ 30% K
- C0 M18 Y50 K2 (tan/gold) ↔ 50% K
- C41 M0 Y1 K0 (cyan) ↔ 70% K
- C60 M37 Y15 K10 (slate)
- C40 M40 Y0 K18 (purple-gray)

Use these (converted to hex as needed) as a categorical palette when a chart
or diagram needs more than the 3 core brand colors — they're the manual's own
answer to "what else can I use," so reach for them before inventing new hues.

## Typography

**Source Sans Pro** (also available as the variable font **Source Sans Var.**)
is the HZB house font, designed by Paul D. Hunt, open-source and free to use
everywhere (fonts.adobe.com/fonts/source-sans, or via Google Fonts/CDN for
web output). Weights in use: Extra Light, Light, Regular, Semibold, Bold,
Black (each with an italic).

Weight → role mapping (apply this, don't just pick "bold for headings" by
generic instinct):

- **Light** — long-form running body text (min. size 10.5pt / 16pt leading in print)
- **Regular** — body text under 10.5pt, captions, special text formats
- **Semibold** — subheadings, callouts/emphasis, graphic titles
- **Bold** — headlines, emphasis, key statements ("Kernsätze")

Fallback: if Source Sans Pro genuinely isn't available (e.g. a constrained
Office template), the manual's designated substitute for internal use
(correspondence, PowerPoint) is **Calibri** — not an arbitrary system sans.

The HZB Schülerlabor (student lab) sub-brand uses a different open-source
font, **Share**, but *only* for that sub-brand — don't apply it to general
HZB or BESSY II material.

### Title/keyvisual ("Bühne") typography specifically

When laying out a cover, hero section, or title slide in the HZB style:

- Headline: set in caps (Versalien), Source Sans Pro/Var **Bold** — lowercase
  is an accepted alternative when caps would look off (e.g. hard-to-read words).
- Subline: lowercase, **Light** or **Regular**, at 50% opacity of HZB Cyan.
- Optional second subline: first subline becomes Light White, second subline
  is Semibold at 50% HZB Cyan.
- Headline : subline size ratio ≈ 2:1 (subline ~50% of headline size).
- Alignment: center all lines within the stage area; left-align the block to
  the "3-Sprung" element (see grid system below).
- Digital versions specifically use the **Light** weight for sublines.

## Logo rules

You will not have HZB's actual logo file — don't fabricate a lookalike
wordmark and pass it off as the real thing. What you *can* apply correctly:

- Base size is defined relative to page margin: the page's margin width
  becomes the logo height, scalable by a factor of 1.5 for other formats.
  Standard concrete sizes: 8mm height on A4, 75% of that on A5/DIN-long
  (minimum size), 12mm on A3, 32mm on A0 posters, 36mm on a 2m roll-up, 6mm on
  a DIN-long flyer.
- Clear space ("Schutzzone") around the logo is defined by the height of the
  "H" in the HZB lettering — leave at least that much empty space on all sides.
- Placement: top margin, above the "stage" (Bühne), not floating mid-layout.
- Never distort, stretch, squash, recolor, flip, or rearrange the logo's
  typography — treat it as a fixed unit, not something to restyle per-page.
- Never place the (standard, non-inverse) logo on a colored background — HZB
  maintains a separate inverse/reverse-color variant for that case, which you
  won't have access to; if a colored background is unavoidable, flag it
  rather than improvising an inverted version yourself.
- Sub-brands (e.g. a department or facility under HZB) get a submark: the
  "HZB" abbreviation with the sub-brand wordmark to its right. BESSY II is the
  documented exception with its own treatment. Actually producing a new
  submark is restricted to HZB's KOM department in practice — treat this as
  "don't invent new submarks," not as a layout you should generate freely.

## Grid system

Six-column grid, used consistently across print and digital:

- **Print (A4 brochure)**: 6 columns, 8mm column width, 16pt baseline grid,
  margins 18mm top / 16mm other sides, identical on left/right pages. Page
  numbers sit centered, outside the type area.
- **Digital**: 6 columns, 17px column width, 16pt baseline grid, 50px margins.

### Proportional design system (for covers, posters, roll-ups)

The manual defines a reusable calculation, not just fixed sizes — useful when
asked for a non-standard format:

1. Margin `X` = the format's defined margin size (see table below).
2. Divide the page height by 6 → each unit is one "stage" (Bühne) increment.
   Minimum stage height = 1 unit, extendable in half- or full-unit steps.
   - With a cover image: stage starts at unit 2 + margin `X`.
   - Without an image: stage starts at unit 2 (margin `X` optional).
3. `X` is also the base spacing unit for indents and for sizing the
   "3-Sprung" element (below) — scale by ×1.5 or ×0.5 for custom formats
   rather than picking arbitrary spacing values.

Known `X` values:

| Format | Margin X | 3-Sprung width |
|---|---|---|
| A4 portrait | 8mm | 24mm (3×8) |
| A3 poster | 12mm | 36mm (3×12) |
| A0 poster | 40mm | 120mm (3×40) |
| Roll-up (85×200cm) | 32mm | 144mm (3×32, ×1.5 optical correction) |
| Mobile/digital | 25px | 75px (3×25) |

## Keyvisuals

- **Bühne (stage)**: the graphic carrier for a page's theme — a banner-like
  zone, sized per the grid rules above, optionally flanked by an "action
  zone" that can hold a themed illustration. HZB's flagship illustration
  motif is "Licht und Linie" (Light and Line). Default to the stage *without*
  an illustration (the manual calls this the "Basic" treatment) unless the
  task specifically calls for one of the themed motifs — combining
  illustrations well requires real publication-design skill and is explicitly
  flagged in the source manual as advanced/specialist work, so keep it simple
  and restrained rather than busy.
- **HZB-3-Sprung**: a three-step color-coded graphic element symbolizing
  progression, used to tag content by theme (e.g. Energy/Information/Matter,
  or by facility — HZB, BESSY II, Schülerlabor each get distinct color
  coding). Position it flush above or below the stage; it can double as a
  graphic divider. Its width scales with the grid system's `X` value (see
  table above) — don't size it arbitrarily.

## Photography & illustration style

- Photos: authentic (not stocky/staged-looking), showing real activity,
  achievements, or equipment from interesting angles. Bright, high-contrast,
  with a subtle blue cast, consistent across a single publication or series.
  Cover images should be large-format and not too colorful, so they don't
  fight the brand colors. Scientific publications get a defined title-area
  slot for generated scientific imagery specifically.
- Where a photo doesn't fit, use structural/abstract stock imagery or an
  illustration instead, in the same restrained, technical mood.
- Diagrams/charts: fit them to the column grid (can bleed to the edge for
  large ones). Simple, low-detail graphics can sit on an HZB-blue field;
  complex scientific graphics stay on white or a 12% K gray field. Use the
  data-viz accent palette above for multi-series distinction.

## Applying this to common outputs

- **Web page / dashboard / artifact**: HZB Blue `#184e8b` as primary,
  HZB Cyan `#6cabe9` as secondary/accent, Source Sans Pro loaded from Google
  Fonts or a CDN, generous white space, 6-column-ish responsive grid, coral
  accent (`#d15e57`) used sparingly for CTAs/highlights only.
- **Report / doc / slide deck**: Bold caps or Bold headline in HZB Blue,
  Light-weight body copy, Semibold for subheadings, the 3-Sprung motif (as a
  simple three-color bar) as a section divider if a keyvisual element is
  wanted without full illustration work.
- **Chart / data viz**: lead with HZB Blue and Cyan for the primary series,
  reach for the recommended graphic-color set above (not arbitrary
  categorical palettes) once more than two series are needed.
- If the task is BESSY II-specific, swap HZB Cyan for the lime accent
  (CMYK 40/0/100/0) as the co-color, per the 3-Sprung facility coding.
