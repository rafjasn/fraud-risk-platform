"""HTML report pages: one layout and stylesheet for every report.

A page is self-contained: charts are embedded PNGs and nothing is fetched, so a
report opens offline. Every function escapes the text it is given, except `body`
and `intro` arguments, which take HTML built from these functions.
"""

import base64
import json
from html import escape

from fraud_risk_platform.reporting.palette import (
    BORDER,
    CRITICAL,
    GOOD,
    GRID,
    INK,
    MUTED,
    PAGE,
    SECONDARY,
    SERIOUS,
    SURFACE,
    WARNING,
)

STYLE = (
    f":root{{color-scheme:light;--page:{PAGE};--surface:{SURFACE};--ink:{INK};"
    f"--secondary:{SECONDARY};--muted:{MUTED};--hairline:{GRID};--border:{BORDER};"
    f"--good:{GOOD};--warning:{WARNING};--serious:{SERIOUS};--critical:{CRITICAL}}}"
    """
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:32px 24px 48px}
.area{margin:0;color:var(--secondary);font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase}
h1{font-size:30px;line-height:1.25;margin:4px 0 14px}
h2{font-size:21px;line-height:1.3;margin:0 0 12px}
h3{font-size:17px;margin:20px 0 8px}
p{max-width:80ch;margin:0 0 12px}
.status{margin:0 0 14px}
.badge{display:inline-flex;align-items:center;gap:8px;padding:4px 14px 4px 5px;border:1px solid var(--border);border-radius:999px;background:var(--surface);font-size:14px;font-weight:600}
.badge .icon{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;background:var(--tone);color:var(--ink);font-size:12px;line-height:1}
.good{--tone:var(--good)}.warning{--tone:var(--warning)}.serious{--tone:var(--serious)}.critical{--tone:var(--critical)}.neutral{--tone:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0 4px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
.tile .label,.tile .note{display:block;color:var(--secondary);font-size:13px}
.tile strong{display:block;font-size:26px;font-weight:600;line-height:1.25;margin-top:4px;overflow-wrap:anywhere}
.tile .note{margin-top:4px}
section,details{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:24px;margin:20px 0}
section details{background:none;border:0;border-radius:0;padding:0;margin:16px 0 0}
.table-scroll{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;vertical-align:top;padding:9px 10px;border-bottom:1px solid var(--hairline)}
th{background:var(--page);font-weight:600}
td{font-variant-numeric:tabular-nums}
th.r,td.r{text-align:right}
td.r,td.n{white-space:nowrap}
.muted,figcaption{color:var(--secondary);font-size:13px}
.lead{font-size:19px}
.notice{border-left:4px solid var(--warning);background:var(--page);padding:12px 14px;border-radius:0 8px 8px 0}
figure{margin:20px 0 8px}
img{display:block;max-width:100%;height:auto}
.timeline{width:100%;max-height:180px;font-size:15px}
.timeline text{fill:var(--secondary)}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;margin:12px 0 0}
code{font:13px ui-monospace,SFMono-Regular,Menlo,monospace}
summary{cursor:pointer;font-weight:600}
a{color:inherit}
@media (max-width:760px){main{padding:20px 16px 32px}section,details{padding:16px}h1{font-size:24px}.tile strong{font-size:22px}}
"""
)
# Tone of a status badge: its color, and an icon so color never carries it alone.
ICONS = {"good": "✓", "warning": "!", "serious": "!", "critical": "×", "neutral": "•"}


def e(value) -> str:
    return escape(str(value))


def number(value) -> str:
    return "Unavailable" if value is None else f"{value:,}"


def percent(value, digits=2, *, missing="Unavailable") -> str:
    return missing if value is None else f"{value:.{digits}%}"


def share(count, total):
    return count / total if total else None


def money(value) -> str:
    return "Unavailable" if value is None else f"${float(value):,.2f}"


def table(headers, rows, *, align=None) -> str:
    """A scrollable table of text cells, with one `align` code per column.

    "l" is text that may wrap, "n" text kept on one line (dates, identifiers), and
    "r" a right-aligned number, also kept on one line.
    """
    align = align or "l" * len(headers)
    if len(align) != len(headers) or set(align) - {"l", "n", "r"}:
        raise ValueError("Give one alignment, l, n, or r, per column")

    def cell(tag, value, side):
        attributes = (' scope="col"' if tag == "th" else "") + (
            f' class="{side}"' if side != "l" else ""
        )
        return f"<{tag}{attributes}>{e(value)}</{tag}>"

    head = "".join(cell("th", h, side) for h, side in zip(headers, align))
    body = "".join(
        "<tr>" + "".join(cell("td", v, side) for v, side in zip(row, align)) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def section(key, title, body) -> str:
    return f'<section id="{e(key)}"><h2>{e(title)}</h2>{body}</section>'


def figure(png: bytes, caption=None, *, alt=None) -> str:
    """An embedded chart; `alt` defaults to the caption."""
    source = base64.b64encode(png).decode()
    text = f"<figcaption>{e(caption)}</figcaption>" if caption else ""
    return (
        f'<figure><img alt="{e(alt or caption or "")}" '
        f'src="data:image/png;base64,{source}">{text}</figure>'
    )


def tiles(items) -> str:
    """Headline numbers: (label, value) or (label, value, note) tuples."""
    cards = []
    for label, value, *note in items:
        extra = f'<span class="note">{e(note[0])}</span>' if note and note[0] else ""
        cards.append(
            f'<div class="tile"><span class="label">{e(label)}</span>'
            f"<strong>{e(value)}</strong>{extra}</div>"
        )
    return f'<div class="tiles">{"".join(cards)}</div>' if cards else ""


def badge(label, tone) -> str:
    """A state: good, warning, serious, critical, or neutral."""
    if tone not in ICONS:
        raise ValueError(f"Unknown badge tone: {tone}")
    return (
        f'<p class="status"><span class="badge {tone}">'
        f'<span class="icon" aria-hidden="true">{ICONS[tone]}</span>{e(label)}</span></p>'
    )


def evidence(value, summary="Recorded findings and provenance (JSON)") -> str:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str)
    return (
        f'<details id="provenance"><summary>{e(summary)}</summary>'
        f"<pre>{e(text)}</pre></details>"
    )


def page(*, area, title, body, badge="", intro="", tiles="", evidence="") -> str:
    """A complete report page: header, then sections, then collapsed evidence.

    `area` names the part of the platform ("Data", "Training"); `badge`, `intro`,
    `tiles`, and `evidence` take the HTML of the functions with those names.
    """
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{e(title)}</title><style>{STYLE}</style></head><body><main>"
        f'<header><p class="area">{e(area)}</p><h1>{e(title)}</h1>'
        f"{badge}{intro}{tiles}</header>{body}{evidence}</main></body></html>"
    )
