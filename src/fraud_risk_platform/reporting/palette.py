"""Every report color, named by the job it does. No other module defines a color.

The values are the light-mode dataviz reference palette. The categorical slots pass
the adjacent-pair color-vision checks in this order, so series take them in order
and never cycle; slots 3-5 sit below 3:1 against the surface, so charts using them
carry a legend and a report table. Reports are light only: charts are PNGs drawn on
the light surface, which a dark page would frame as bright rectangles.
"""

# Identity: series in a fixed order, so a recipe keeps its color across cycles.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)
# One measure in a panel takes the first slot.
SERIES = CATEGORICAL[0]
# The two classes keep their colors in every report.
LEGITIMATE, FRAUD = CATEGORICAL[:2]
# Magnitude: one hue, from near zero (light) to high (dark).
SEQUENTIAL = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
# Order: steps of the same hue for train, validation, evaluation.
ORDINAL = ("#86b6ef", "#2a78d6", "#104281")
# States only, never a series; always shown with an icon and a label.
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# Page and chart chrome.
PAGE = "#f9f9f7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
# Baselines, reference lines, and de-emphasized series.
MUTED = "#898781"
GRID = "#e1e0d9"
# Axis lines, and recessive fills such as pending labels and the context band.
AXIS = "#c3c2b7"
# Cells with too little data to show a value.
NEUTRAL = "#f0efec"
# Text inside a dark fill.
ON_DARK = "#ffffff"
BORDER = "rgba(11, 11, 11, 0.10)"
