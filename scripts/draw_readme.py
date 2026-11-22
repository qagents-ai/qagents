"""Draw the README illustrations as deterministic, accessible SVGs.

Run: python scripts/draw_readme.py
Uses only the standard library. Layout, text and icon paths are versioned here.
"""

from html import escape
from pathlib import Path

NAVY = "#101e2b"
INK = "#142d3c"
MINT = "#8ce6c4"
TEAL = "#26796c"
MUTED = "#a8becb"
WHITE = "#f7fbfa"
VIOLET = "#c4b7fa"
AMBER = "#f3cc85"


class Drawing:
    def __init__(self, name: str, height: int, title: str, description: str, *, dark=False):
        self.name = name
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}" '
            f'viewBox="0 0 1200 {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            """<defs>
              <linearGradient id="night" x2="1" y2="1">
                <stop stop-color="#122b38"/><stop offset="1" stop-color="#101722"/>
              </linearGradient>
              <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">
                <circle cx="1" cy="1" r="1" fill="#476272" opacity=".22"/>
              </pattern>
              <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 1 1 L 8 5 L 1 9" fill="none" stroke="#6d9b9a" stroke-width="1.5"/>
              </marker>
            </defs>""",
            '<g font-family="Segoe UI, Inter, Helvetica, Arial, sans-serif">',
        ]
        self.rect(0, 0, 1200, height, "url(#night)" if dark else "#f1f5f3", 26)
        if dark:
            self.rect(0, 0, 1200, height, "url(#dots)", 26)

    def add(self, value):
        self.parts.append(value)

    def rect(self, x, y, w, h, fill, radius=12, stroke="none"):
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )

    def text(self, x, y, value, size=18, color=INK, weight=400, anchor="start"):
        self.add(
            f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>'
        )

    def path(self, d, color="#6d9b9a", width=2, *, arrow=False, dashed=False):
        self.add(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            'stroke-linecap="round" stroke-linejoin="round"'
            + (' marker-end="url(#arrow)"' if arrow else "")
            + (' stroke-dasharray="5 7"' if dashed else "")
            + "/>"
        )

    def circle(self, x, y, r, fill, stroke="none"):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}"/>')

    def icon(self, kind, x, y, color, scale=1):
        paths = {
            "plan": "M3 4h18v20H3z M8 9h8 M8 14h8 M8 19h5",
            "tool": "M5 6l-4 7 4 7 M19 6l4 7-4 7 M15 3l-6 20",
            "check": "M3 13l6 6L22 5",
            "resume": "M4 8a10 10 0 1 1-1 10 M4 1v7h7",
            "memory": "M3 7c0-5 20-5 20 0s-20 5-20 0v14c0 5 20 5 20 0V7 M3 14c0 5 20 5 20 0",
            "scope": "M13 1L2 6v9c0 6 11 11 11 11s11-5 11-11V6z M8 13l4 4 6-7",
            "branch": "M5 4v16 M5 12h13V4 M5 12h13v10",
            "book": "M1 3h9l3 3 3-3h9v20h-9l-3 2-3-2H1z M13 6v19",
        }
        self.add(f'<g transform="translate({x} {y}) scale({scale})">')
        self.path(paths[kind], color, 1.8)
        self.add("</g>")

    def eyebrow(self, number, title, *, dark=False):
        self.text(48, 48, number, 15, MINT if dark else TEAL, 700)
        self.text(90, 48, title.upper(), 15, MUTED if dark else "#526d75", 600)

    def save(self, directory):
        self.add("</g></svg>")
        (directory / f"{self.name}.svg").write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def journey(out):
    d = Drawing(
        "agent-flow",
        620,
        "Deep work, with continuity",
        "A long-running agent plans, uses typed tools, pauses for human review and resumes. "
        "Sessions retain execution history and state. Application memory supplies scoped context.",
        dark=True,
    )
    d.eyebrow("01", "The agent journey", dark=True)
    d.text(48, 105, "Deep work, with continuity.", 42, WHITE, 650)
    d.text(
        48,
        143,
        "Plan across steps. Delegate work. Keep context when the conversation pauses.",
        20,
        MUTED,
    )
    d.path("M101 219H1099", "#38545f", 2)
    stages = [
        (
            48,
            "01",
            "Plan & delegate",
            "Break down a task.",
            "Call specialist agents.",
            "plan",
            MINT,
        ),
        (333, "02", "Use tools", "Validate arguments.", "Apply permissions in code.", "tool", MINT),
        (
            618,
            "03",
            "Review",
            "Pause for a decision.",
            "Inspect the proposed action.",
            "check",
            AMBER,
        ),
        (
            903,
            "04",
            "Resume",
            "Load the saved session.",
            "Continue from its journal.",
            "resume",
            VIOLET,
        ),
    ]
    for x, n, title, a, b, icon, accent in stages:
        d.circle(x + 24, 219, 15, NAVY, accent)
        d.text(x + 24, 224, n, 12, accent, 700, "middle")
        d.rect(x, 252, 249, 154, "#1b303d", 16, "#35505c")
        d.icon(icon, x + 20, 271, accent, 1)
        d.text(x + 58, 293, title, 23, WHITE, 600)
        d.text(x + 20, 336, a, 17, MUTED)
        d.text(x + 20, 366, b, 17, MUTED)
        d.path(f"M{x + 124} 407V448", "#426c73", dashed=True)
    d.rect(48, 448, 1104, 120, "#203e46", 18, "#3b6b6b")
    d.icon("memory", 70, 476, MINT, 1.2)
    d.text(116, 487, "CONTINUITY LAYER", 14, MINT, 700)
    d.text(116, 524, "Journal + structured state", 24, WHITE, 600)
    d.path("M564 472V543", "#44646b")
    d.icon("scope", 597, 476, VIOLET, 1.2)
    d.text(643, 487, "APPLICATION MEMORY", 14, VIOLET, 700)
    d.text(643, 524, "Scoped retrieval + Gym playbooks", 24, WHITE, 600)
    d.text(
        48,
        598,
        "Provider choice, memory scope and action safety remain explicit application decisions.",
        16,
        MUTED,
    )
    d.save(out)


def memory(out):
    d = Drawing(
        "memory-layers",
        610,
        "Three kinds of memory",
        "Execution history lives in the journal. "
        "Structured application state belongs to a session. "
        "Optional retrieved memory and ACE playbooks are supplied through "
        "application or QAgent Gym adapters. "
        "The application derives tenant and user scope before retrieving context.",
    )
    d.eyebrow("02", "Memory architecture")
    d.text(48, 102, "Remember the right things.", 40, INK, 650)
    d.text(
        48,
        140,
        "Execution history, application state and learned context have different jobs.",
        20,
        "#526d75",
    )
    layers = [
        (
            48,
            "EXECUTION",
            "Journal",
            "Messages, decisions,",
            "tool starts and results.",
            "Session replay & recovery",
            "memory",
            TEAL,
        ),
        (
            425,
            "APPLICATION",
            "Structured state",
            "Preferences and facts",
            "owned by your application.",
            "RunContext.state",
            "scope",
            "#6952a2",
        ),
        (
            802,
            "LEARNING",
            "Retrieved context",
            "Mem0 memory and",
            "ACE-style playbooks.",
            "Optional QAgent Gym adapters",
            "book",
            "#9a6526",
        ),
    ]
    for x, label, title, a, b, foot, icon, accent in layers:
        d.rect(x, 184, 350, 274, "#ffffff", 18, "#d8e3de")
        d.rect(x + 22, 207, 52, 52, "#eef4f1", 14)
        d.icon(icon, x + 35, 220, accent)
        d.text(x + 91, 239, label, 14, accent, 700)
        d.text(x + 24, 300, title, 28, INK, 650)
        d.text(x + 24, 341, a, 20, "#526d75")
        d.text(x + 24, 371, b, 20, "#526d75")
        d.path(f"M{x + 24} 395H{x + 326}", "#e1e9e4", 1)
        d.text(x + 24, 429, foot, 17, accent, 600)
    d.rect(48, 490, 1104, 76, INK, 16)
    d.icon("scope", 70, 513, MINT)
    d.text(116, 523, "Identity comes from your application.", 20, WHITE, 600)
    d.text(
        116,
        549,
        "Derive tenant / user scope before loading memory; select what enters the model context.",
        17,
        "#c7d8dc",
    )
    d.save(out)


def approval(out):
    d = Drawing(
        "approval-flow",
        510,
        "A support case with human approval",
        "The agent reads a demo order, proposes a support case and persists the pending decision. "
        "Approval writes to the local SQLite queue using a stable operation id. "
        "Decline leaves the queue unchanged. "
        "A retry with the same payload returns the existing case.",
    )
    d.eyebrow("03", "Runnable support assistant")
    d.text(48, 102, "An action you can inspect.", 40, INK, 650)
    d.text(
        48,
        140,
        "A real local queue, an explicit approval and an operation id that binds the request.",
        20,
        "#526d75",
    )
    d.rect(48, 196, 272, 186, "#ffffff", 18, "#d8e3de")
    d.icon("tool", 72, 219, TEAL)
    d.text(72, 277, "Inspect an order", 26, INK, 600)
    d.text(72, 315, "Read demo order data.", 18, "#526d75")
    d.text(72, 345, "Propose a support case.", 18, "#526d75")
    d.path("M320 290H377", arrow=True)
    d.rect(391, 196, 330, 186, INK, 18)
    d.icon("check", 415, 218, AMBER)
    d.text(415, 277, "Pause for approval", 26, WHITE, 600)
    d.text(415, 315, "Show the exact arguments.", 18, "#c7d8dc")
    d.text(415, 345, "Save the pending decision.", 18, "#c7d8dc")
    d.path("M721 262H803", arrow=True)
    d.text(748, 247, "Approve", 15, TEAL, 600)
    d.rect(817, 196, 335, 186, "#dff2e9", 18, "#a9cdbc")
    d.icon("memory", 841, 216, TEAL)
    d.text(841, 277, "Persist the case", 26, INK, 600)
    d.text(841, 315, "SQLite queue + payload binding.", 18, "#355e54")
    d.text(841, 345, "Same retry → same case.", 18, "#355e54")
    d.path("M556 382V431H790", "#7e8f94", arrow=True)
    d.text(581, 418, "Decline", 15, "#526d75", 600)
    d.text(817, 438, "Queue stays unchanged.", 18, "#526d75")
    d.text(
        48,
        481,
        "Example boundary: demo orders and local review cases. "
        "Model requests use your selected provider.",
        16,
        "#526d75",
    )
    d.save(out)


def providers(out):
    d = Drawing(
        "provider-adapters",
        390,
        "One agent API, explicit model adapters",
        "QCloudModel, OpenAIChatModel and AnthropicModel implement the model contract. "
        "Tools, permissions and sessions remain application-owned. "
        "Other endpoints need explicit compatibility validation.",
        dark=True,
    )
    d.eyebrow("04", "Model adapters", dark=True)
    d.text(48, 103, "Your agent. Your provider.", 40, WHITE, 650)
    d.text(
        48,
        141,
        "Keep the application contract stable while configuring each model explicitly.",
        20,
        MUTED,
    )
    d.rect(48, 192, 278, 140, "#203e46", 18, "#3b6b6b")
    d.icon("branch", 70, 213, MINT)
    d.text(112, 239, "Agent + Model", 25, WHITE, 600)
    d.text(72, 278, "Typed tools · Permissions", 17, MUTED)
    d.text(72, 307, "Sessions · Streaming", 17, MUTED)
    d.path("M326 262H366V174H1031", "#6d9b9a")
    for x in (529, 780, 1031):
        d.path(f"M{x} 174V190", arrow=True)
    for x, title, cls, color in [
        (408, "QCloud", "QCloudModel", MINT),
        (659, "OpenAI", "OpenAIChatModel", WHITE),
        (910, "Anthropic", "AnthropicModel", VIOLET),
    ]:
        d.rect(x, 192, 242, 140, "#1b303d", 16, "#35505c")
        d.circle(x + 24, 220, 4, color)
        d.text(x + 24, 266, title, 28, color, 600)
        d.text(x + 24, 305, cls, 17, MUTED)
    d.text(
        48,
        369,
        "Compatible HTTP endpoints use OpenAIChatModel "
        "with an explicit API root and supported request fields.",
        16,
        MUTED,
    )
    d.save(out)


def main():
    out = Path(__file__).resolve().parents[1] / "assets/diagrams"
    out.mkdir(parents=True, exist_ok=True)
    for draw in (journey, memory, approval, providers):
        draw(out)
    print(f"Rendered 4 accessible SVGs to {out}")


if __name__ == "__main__":
    main()
