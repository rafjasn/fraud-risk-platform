"""One look for every report: palette, chart styling, and page layout.

palette.py defines every color, and no other module defines one; style.py applies
the palette to matplotlib charts; page.py builds the HTML pages. They need only the
standard library, matplotlib, and numpy, so the ingestion image, which has no model
libraries, renders data reports exactly like training and monitoring reports.
"""
