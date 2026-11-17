"""Palette, number formats and chart helpers shared by every page.

Colours are the first three slots of a palette validated for colour-vision deficiency
(adjacent and all-pairs) in light and dark mode, plus a neutral for context marks.
Each mode has its own steps; the page reads the active theme from st.context.
"""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import altair as alt
import streamlit as st


@dataclass(frozen=True)
class Palette:
    blue: str
    orange: str
    aqua: str
    neutral: str
    faint: str
    sequential: tuple[str, str]
    ink: str          # text on light cells
    ink_inverse: str  # text on dark cells
    subtitle: str


LIGHT = Palette(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", neutral="#9a9890",
                faint="#d9d8d3", sequential=("#eef4fc", "#104281"),
                ink="#262624", ink_inverse="#ffffff", subtitle="#52514e")
DARK = Palette(blue="#3987e5", orange="#d95926", aqua="#199e70", neutral="#8a8984",
               faint="#3a3a38", sequential=("#1f2a38", "#9ec5f4"),
               ink="#f0efec", ink_inverse="#0b0b0b", subtitle="#c3c2b7")


def palette() -> Palette:
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except AttributeError:  # older Streamlit without st.context.theme
        return LIGHT


def num(x: float, digits: int = 3) -> str:
    """Round half up from the decimal representation, so 0.6625 reads 0.663 as in the README."""
    q = Decimal(1).scaleb(-digits)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return num(100 * float(x), digits) + "%"


def ci(lo_hi: list[float], digits: int = 3) -> str:
    return f"95% CI {num(lo_hi[0], digits)} to {num(lo_hi[1], digits)}"


def chart(c: alt.TopLevelMixin, height: int = 320) -> None:
    """Render an Altair chart with the Streamlit theme and no chart junk.

    `height` is the height of the whole element (plot, title, axes and legend) in pixels;
    Streamlit fits the chart into it."""
    c = (c.configure_view(strokeWidth=0)
          .configure_axis(grid=False, domainOpacity=0.6, tickOpacity=0.6, labelFontSize=12,
                          titleFontSize=12, titleFontWeight="normal")
          .configure_axisY(grid=True, gridOpacity=0.25)
          .configure_title(fontSize=15, anchor="start", fontWeight=600, subtitleFontSize=12,
                           color=palette().ink, subtitleColor=palette().subtitle, subtitlePadding=4)
          .configure_legend(labelFontSize=12, titleFontSize=12, orient="bottom", titleFontWeight="normal",
                            labelLimit=0))
    st.altair_chart(c, width="stretch", height=height, theme="streamlit")


def title(text: str, subtitle: str | None = None, wrap: int | None = None) -> alt.TitleParams:
    """Chart title stating the takeaway; `wrap` breaks it into lines for narrow (half-width) charts."""
    lines: str | list[str] = textwrap.wrap(text, wrap) if wrap else text
    sub: str | list[str] = textwrap.wrap(subtitle, int(wrap * 1.25)) if (wrap and subtitle) else (subtitle or "")
    return alt.TitleParams(text=lines, subtitle=sub)


def page_intro(heading: str, lede: str) -> None:
    st.title(heading)
    st.markdown(f"<p style='font-size:1.08rem; opacity:0.85; max-width:52rem'>{lede}</p>",
                unsafe_allow_html=True)


def caption(text: str) -> None:
    st.caption(text)
