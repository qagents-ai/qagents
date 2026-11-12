# qagents logo

The selected mark is **concept 2: the journal q**. Three horizontal bands form a circular bowl; the final teal band continues into a short rounded tail. The open counter keeps the loop readable, while the horizontal gaps suggest an append-only record of completed steps. The accent marks the latest committed step and the continuation of a run.

## Concept selection

1. **Loop q** — a closed indigo loop with a teal directional chevron. Direct and simple, but the conventional ring and arrow give it less individual character.
2. **Journal q — selected** — stacked journal bands with a teal final entry and tail. It connects the agent loop to the project's durable journal in one compact symbol. Its broad shapes also survive reduction to 32 px.
3. **Three-node q** — three peers on one loop, with a teal node and tail. It communicates cooperation well, but gives multiple agents more emphasis than the small core loop.

The final SVG is a deliberate hand-drawn refinement of concept 2, not an automatic bitmap trace. The circular geometry is consistent, the two journal gaps are 8 units wide, and the tail is integrated into the last band. The icon uses three filled paths, with no gradients, effects, or raster content.

## Colours

| Role | On white | On dark backgrounds |
| --- | --- | --- |
| Journal bands | Indigo `#3B3F9E` | Periwinkle `#A5AAFF` |
| Final entry and tail | Teal `#008D82` | Mint teal `#45D6BF` |
| Wordmark neutral | Ink `#18243B` | Off-white `#F0F3FA` |
| Preview background | `#FFFFFF` | `#0D1117` |

The icon uses two colours; each lockup adds one neutral for the wordmark. All three SVGs have transparent backgrounds. The backgrounds in the PNG previews are presentation backdrops, not additional colours embedded in the SVG artwork.

Measured contrast ratios against the intended backgrounds:

| Element | Light | Dark |
| --- | --- | --- |
| Journal bands | 8.78:1 | 8.83:1 |
| Accent | 4.09:1 | 10.48:1 |
| Wordmark | 15.51:1 | 17.04:1 |

## Files and use

| File | Purpose |
| --- | --- |
| `concepts/concept-1.png`, `concepts/concept-2.png`, `concepts/concept-3.png` | Original image-generated concept explorations; 1254 × 1254 px each |
| `qagents-icon.svg` | Square standalone icon; `viewBox="0 0 256 256"` |
| `qagents-logo.svg` | Horizontal icon and lowercase wordmark for light backgrounds |
| `qagents-logo-dark.svg` | Matching horizontal geometry with colours adjusted for dark backgrounds |
| `qagents-icon.png` | 512 × 512 px preview on white |
| `qagents-logo.png` | 512 × 158 px preview on white |
| `qagents-logo-dark.png` | 512 × 158 px preview on `#0D1117` |
| `qagents-icon-32.png` | Additional 32 × 32 px readability preview |
| `build-logo.ps1` | Rebuilds the hand-drawn SVGs, outlines the wordmark, and renders the three 512 px previews locally |

The lowercase wordmark is Segoe UI Semibold converted to filled SVG paths. The delivered SVGs require no fonts and contain no `<text>` elements or external resources. The two lockups share the same `viewBox="0 0 829 256"` and exactly the same path geometry.

Use the standalone icon at 32 px or larger. Keep the supplied square padding and preserve the icon-to-wordmark spacing when scaling a lockup. Use the dark lockup on `#0D1117` or a similarly dark background.

The three previews were rendered with the installed ImageMagick/librsvg converter and visually inspected. The 32 px icon was also checked. SVG structure, palette limits, matching lockup paths, and PNG dimensions were checked locally.

To rebuild on Windows with Segoe UI Semibold and ImageMagick installed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build-logo.ps1
```

## Image generation prompts

The three original concepts were generated with the built-in `image_gen.imagegen` tool, one call per concept, with `transparent_background: false`. The generated PNGs were copied into this directory unchanged. No CLI/API fallback was used. Network access was limited to the image generation tool.

The exact prompts follow.

### Concept 1

```text
Use case: logo-brand
Asset type: square app/repository icon for qagents, an open-source Python AI agent framework.
Primary request: Create one exceptionally simple, distinctive lowercase-q monogram: a bold indigo closed circular loop with generous white counterspace and a short diagonal tail at lower right. Integrate one small teal directional chevron at the junction of loop and tail to suggest model -> tools -> model. A single compact silhouette; sophisticated geometric proportions; strong at 32 px.
Scene/backdrop: completely plain pure white background.
Style/medium: crisp flat vector-style logo, solid shapes only.
Composition/framing: exactly one centered icon occupying about 70 percent of the square, generous white margins.
Color palette: deep indigo #3B3F9E and teal #009C8C only, plus white.
Text: none. The q is a symbolic monogram, with no text or lettering inside it.
Constraints: no wordmark, no labels, no extra symbols, no tiny decoration, no gradients, no 3D, no shadows, no mockup, no texture, no border. Do not depict a robot, brain, sparkles, graphs, or crews.
```

### Concept 2

```text
Use case: logo-brand
Asset type: square app/repository icon for qagents, an open-source Python AI agent framework.
Primary request: Create one minimal lowercase-q monogram built from three bold horizontal journal strokes. Their rounded ends and carefully shaped white gaps form a unified circular q bowl with open white counterspace; the bottom stroke turns into a short lower-right diagonal tail. The top two strokes are deep indigo; the last committed stroke and tail are teal. The emblem should evoke an append-only journal, durable progress, and a simple repeatable loop. Make the stacked-stroke construction visually distinct and immediately legible as q at 32 px.
Scene/backdrop: completely plain pure white background.
Style/medium: crisp flat vector-style logo, solid shapes only.
Composition/framing: exactly one centered icon occupying about 70 percent of the square, generous white margins.
Color palette: deep indigo #3B3F9E and teal #009C8C only, plus white.
Text: none. The q is a symbolic monogram, with no text or lettering inside it.
Constraints: no wordmark, no labels, no tiny decoration, no gradients, no 3D, no shadows, no mockup, no texture, no border. Do not depict a notebook, robot, brain, sparkles, or a flowchart.
```

### Concept 3

```text
Use case: logo-brand
Asset type: square app/repository icon for qagents, an open-source Python AI agent framework.
Primary request: Create one simple geometric lowercase-q emblem from a circular loop with exactly three large integrated round nodes, one at the top, one lower left, one lower right. The nodes connect along one continuous thick indigo ring; the lower-right node is teal and extends into a short diagonal q tail. Balance the three nodes as peers on one small readable loop. Bold, compact, memorable; readable at 32 px; deliberately distinct from a conventional plain q monogram.
Scene/backdrop: completely plain pure white background.
Style/medium: crisp flat vector-style logo, solid shapes only.
Composition/framing: exactly one centered icon occupying about 70 percent of the square, generous white margins.
Color palette: deep indigo #3B3F9E and teal #009C8C only, plus white.
Text: none. The q is a symbolic monogram, with no text or lettering inside it.
Constraints: no wordmark, no labels, no extra nodes or branches, no tiny decoration, no gradients, no 3D, no shadows, no mockup, no texture, no border. Do not depict a robot, brain, sparkles, flowchart, or large graph.
```


## Origin

Generated with the Codex CLI (codex-cli 0.167) and its built-in image generation tool, then redrawn by hand as SVG. The brief given to Codex is in `codex-prompt.txt`; the image prompts Codex wrote from it are listed above.
