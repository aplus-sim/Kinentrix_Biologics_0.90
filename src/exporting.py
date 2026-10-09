"""Chart export: convert on-screen plotly figures to CSV and PNG.

PNG is redrawn with matplotlib rather than kaleido (not in the venv; it needs a separate Chrome download).
Drawn on a white background for reports; line, band and marker colors and line styles are copied from the plotly traces.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib import font_manager  # noqa: E402

# Inter font, same as the app UI (assets/fonts, SIL OFL); falls back to matplotlib's default if the file is missing.
_FONT = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Inter.ttf"
_FAMILY = "sans-serif"
try:
    if _FONT.exists():
        font_manager.fontManager.addfont(str(_FONT))
        _FAMILY = font_manager.FontProperties(fname=str(_FONT)).get_name()
except Exception:       # a broken font file must not stop the app; use the default font
    _FAMILY = "sans-serif"

_DASH = {"dash": (0, (5, 3)), "dot": (0, (1, 2)), "dashdot": (0, (5, 2, 1, 2)),
         "solid": "-", None: "-"}


def _rgba(c, default="#1f77b4"):
    """plotly color string -> matplotlib color. Also accepts rgba(r,g,b,a)."""
    if not c:
        return default
    c = str(c).strip()
    if c.startswith("rgba") or c.startswith("rgb"):
        v = [float(x) for x in c[c.index("(") + 1:c.index(")")].split(",")]
        a = v[3] if len(v) > 3 else 1.0
        return (v[0] / 255, v[1] / 255, v[2] / 255, a)
    return c


def _darken(c):
    """Darken light colors (e.g. light sky blue) that are invisible on white."""
    rgb = matplotlib.colors.to_rgba(c)
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    if lum > 0.6:
        return (rgb[0] * 0.65, rgb[1] * 0.65, rgb[2] * 0.65, rgb[3])
    return rgb


def profile_csv(fig) -> bytes:
    """Curve/marker traces -> long table (series, time_day, conc_ugmL); bands expand to lo/hi."""
    rows = []
    for tr in fig.data:
        x = np.asarray(tr.x if tr.x is not None else [], dtype=float)
        y = np.asarray(tr.y if tr.y is not None else [], dtype=float)
        name = tr.name or ""
        if getattr(tr, "fill", None) == "toself" and len(x) % 2 == 0 and len(x):
            h = len(x) // 2
            for t, hi, lo in zip(x[:h], y[:h], y[h:][::-1]):
                rows.append((name + " [upper]", t, hi))
                rows.append((name + " [lower]", t, lo))
            continue
        rows += [(name, t, c) for t, c in zip(x, y)]
    df = pd.DataFrame(rows, columns=["series", "time_day", "conc_ugmL"])
    return df.to_csv(index=False).encode("utf-8-sig")


def params_csv(params) -> bytes:
    """Parameter table (list of dicts or DataFrame) -> CSV."""
    df = params if isinstance(params, pd.DataFrame) else pd.DataFrame(params)
    return df.to_csv(index=False).encode("utf-8-sig")


def figure_png(fig, title="", log=True, dpi=200) -> bytes:
    """Redraw a plotly figure with matplotlib and return PNG bytes."""
    with plt.rc_context({"font.family": _FAMILY}):
        return _figure_png(fig, title, log, dpi)


def _figure_png(fig, title, log, dpi):
    f, ax = plt.subplots(figsize=(8.0, 4.6), dpi=dpi)
    for tr in fig.data:
        if tr.x is None or tr.y is None:
            continue
        x = np.asarray(tr.x, dtype=float)
        y = np.asarray(tr.y, dtype=float)
        name = tr.name or None
        mode = getattr(tr, "mode", None) or "lines"
        if getattr(tr, "fill", None) == "toself":
            ax.fill(x, y, color=_rgba(tr.fillcolor, "#cccccc"), lw=0, label=name)
            continue
        if "markers" in mode and "lines" not in mode:
            mk = tr.marker
            ax.scatter(x, y, s=(mk.size or 8) * 4, label=name, zorder=5,
                       color=_darken(_rgba(mk.color, "#333333")),
                       marker="D" if str(mk.symbol or "").startswith("diamond") else "o",
                       edgecolors="white", linewidths=0.6)
            continue
        ln = tr.line
        ax.plot(x, y, label=name, color=_darken(_rgba(ln.color)),
                lw=(ln.width or 2) * 0.8, ls=_DASH.get(ln.dash, "-"))
    if log:
        ax.set_yscale("log")
        # plain numbers at powers of ten (0.1, 1, 10, 100), like the app chart
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ys = np.concatenate([np.asarray(tr.y, dtype=float) for tr in fig.data
                             if tr.y is not None and getattr(tr, "fill", None) != "toself"]
                            or [np.array([1.0])])
        ys = ys[np.isfinite(ys) & (ys > 0)]
        if ys.size:
            ax.set_ylim(max(ys.min(), ys.max() * 1e-3) * 0.8, ys.max() * 1.5)
    else:
        ax.set_ylim(bottom=0)
    xr = fig.layout.xaxis.range
    if xr is not None:
        ax.set_xlim(xr[0], xr[1])
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("Concentration (µg/mL)")
    if title:
        ax.set_title(title, fontsize=11.5, loc="left")
    ax.grid(True, which="major", color="#e5e7eb", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#9ca3af")
    ax.legend(fontsize=8.5, frameon=False, loc="best")
    f.tight_layout()
    buf = io.BytesIO()
    f.savefig(buf, format="png", facecolor="white")
    plt.close(f)
    return buf.getvalue()
