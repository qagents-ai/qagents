# Diagrams

The README uses code-authored SVG illustrations with explicit layout, live text,
accessible titles and descriptions. The navy, mint, amber and violet palette
distinguishes execution, approval and memory. These are reproducible drawings;
no image-generation service is involved.

## README illustrations

```bash
python scripts/draw_readme.py
```

The renderer uses only the Python standard library and writes four SVGs under
`assets/diagrams`: the agent journey, memory layers, support approval flow and
provider architecture. Change the text, positions and icon paths in the renderer,
then regenerate the images. Inspect them at README width before committing.

## Detailed reference diagrams

Mermaid CLI 12.2.0 renders the detailed reference sources into
`assets/diagrams/reference`, so it never overwrites the README artwork.

- `agent-flow.mmd`: application, agent, provider, permissions, tools and journal.
- `approval-flow.mmd`: the runnable support-review example and its approval boundary.
- `provider-adapters.mmd`: one model contract, three separate provider adapters.

### Rebuild reference diagrams

Install the diagram tool locally; it is not a Python runtime dependency:

```powershell
$env:PUPPETEER_SKIP_DOWNLOAD = 'true'
npm install --prefix .work/diagram-tools --save-exact @mermaid-js/mermaid-cli@12.2.0
python scripts/render_diagrams.py --browser 'C:\Program Files\Google\Chrome\Application\chrome.exe' --previews .work/diagram-previews
```

Supply your installed Chrome/Chromium path. `--cli` can point at an existing
Mermaid CLI `src/cli.js`, avoiding a second installation. The command renders
headlessly and leaves source files intact. Inspect PNG previews for clipping,
readability and arrow direction before committing the generated SVGs.

The source-to-image workflow follows the [Mermaid CLI documentation](https://github.com/mermaid-js/mermaid-cli).
