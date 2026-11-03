"""Matplotlib styling for report charts, in the shared palette.

Charts are drawn on Figure objects instead of pyplot, so no global figure state
outlives a report. png() gives equal bytes for equal charts, because data reports
are hashed into snapshot identities.
"""

import io

from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from fraud_risk_platform.reporting.palette import (
    AXIS,
    GRID,
    INK,
    MUTED,
    NEUTRAL,
    ON_DARK,
    SECONDARY,
    SEQUENTIAL,
    SURFACE,
)

DPI = 120
LINE = 1.2  # 2px at DPI 120
HAIRLINE = 0.6  # 1px at DPI 120
MARKER = 6  # 10px diameter at DPI 120
# Matplotlib otherwise records its version, which would change report bytes.
METADATA = {"Software": "fraud-risk-platform"}


def new_figure(width: float, height: float) -> Figure:
    return Figure(figsize=(width, height), layout="constrained", facecolor=SURFACE)


def png(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, facecolor=SURFACE, metadata=METADATA)
    return buffer.getvalue()


def style_axis(axis, title, xlabel="", ylabel="", *, grid="both"):
    """Left-aligned title, recessive solid hairline grid and axes, secondary text.

    `grid` is "both", "x", "y", or None; bar charts keep only the value axis's grid.
    """
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", color=INK, fontsize=11, pad=22)
    axis.set_xlabel(xlabel, color=SECONDARY, fontsize=9)
    axis.set_ylabel(ylabel, color=SECONDARY, fontsize=9)
    if grid:
        axis.grid(True, axis=grid, color=GRID, linewidth=HAIRLINE)
    else:
        axis.grid(False)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(AXIS)
        axis.spines[side].set_linewidth(HAIRLINE)
    axis.tick_params(colors=SECONDARY, labelsize=8, width=HAIRLINE)


def reference(axis, points, label, *, at="end"):
    """A dashed muted line for a baseline or requirement, labeled directly."""
    xs, ys = points
    axis.plot(xs, ys, color=MUTED, linewidth=HAIRLINE * 1.5, linestyle="--")
    if at == "above":
        # Above a rising line near its end: validation rarely beats training there.
        anchor, offset, ha, va = (xs[-2], ys[-2]), (-8, 8), "right", "bottom"
    elif at == "start":
        anchor, offset, ha, va = (xs[0], ys[0]), (4, 4), "left", "bottom"
    else:
        anchor, offset, ha, va = (xs[-1], ys[-1]), (-4, 4), "right", "bottom"
    axis.annotate(
        label,
        anchor,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        color=SECONDARY,
        fontsize=8,
    )


def point(axis, x, y, color, label=None):
    # Operating points wear a 2px surface ring so they stay legible on lines.
    axis.plot(
        [x],
        [y],
        marker="o",
        markersize=MARKER,
        color=color,
        markeredgecolor=SURFACE,
        markeredgewidth=LINE,
        linestyle="none",
        label=label,
        zorder=3,
    )


def add_legend(axis, **kwargs):
    # Above the plot area, between title and axes, so it never covers data.
    axis.legend(
        frameon=False,
        fontsize=8,
        labelcolor=SECONDARY,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        ncols=4,
        borderaxespad=0.2,
        handlelength=1.5,
        **kwargs,
    )


def date_axis(axis):
    locator = AutoDateLocator(minticks=3, maxticks=7)
    formatter = ConciseDateFormatter(locator)
    # The offset below the axis names the year only, not the last tick's month.
    formatter.offset_formats = ["", "%Y", "%Y", "%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M"]
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


def compact(value) -> str:
    """A count as 950, 200K, or 1.6M."""
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def count_axis(axis):
    """Compact tick labels on a y axis of counts, instead of 1e6 offsets."""
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: compact(value)))


def sequential_colormap():
    """Magnitude on the sequential ramp; masked cells show as neutral."""
    return LinearSegmentedColormap.from_list("sequential", SEQUENTIAL).with_extremes(
        bad=NEUTRAL
    )


def _luminance(rgb) -> float:
    return sum(
        weight * (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
        for weight, c in zip((0.2126, 0.7152, 0.0722), rgb)
    )


def label_color(fill) -> str:
    """Ink or white, whichever contrasts more with a fill, for labels inside it."""
    background = _luminance(to_rgb(fill))
    on_ink = (background + 0.05) / (_luminance(to_rgb(INK)) + 0.05)
    return INK if on_ink >= 1.05 / (background + 0.05) else ON_DARK
