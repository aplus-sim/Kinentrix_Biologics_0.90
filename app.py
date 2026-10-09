"""Antibody-drug Human PK prediction & simulation web app (Streamlit).

Tab 1: Existing drugs — allometry+ML prediction, traffic-light flag, PK profile overlay.
Tab 2: New drug input — Drug/In vitro/Monkey inputs -> predicted profile.
"""
import hmac
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import src.appdata as appdata     # precomputed app data
import src.predict as predict
import src.pksim as pksim
import src.exporting as exporting
import src.newdrug as newdrug    # New Drug tab simulation setup (SC F, 1-compartment)

# --- brand ------------------------------------------------------------------
_ASSETS = Path(__file__).resolve().parent / "assets"
# Logo mark (star + arch): assets/kinentrix_logo.png; square icon: assets/kinentrix_icon.png. Falls
# back to an emoji if the file is missing.
_LOGO = next((p for p in (_ASSETS / "kinentrix_logo.png",
                          _ASSETS / "kinentrix_logo.svg") if p.exists()), None)
_ICON = next((p for p in (_ASSETS / "kinentrix_icon.png", _ASSETS / "kinentrix_logo.png")
              if p.exists()), None)

st.set_page_config(page_title="KINENTRIX Biologics", layout="wide",
                   page_icon=str(_ICON) if _ICON else "🧬")

# --- theme -------------------------------------------------------------------
# Dark palette:
#   background #0B1220, card/chart #111C30, input #0B1729, border #1B2638, grid #192436
# Logo only: emerald-turquoise brand color (BRAND_TEAL), kept separate from the blue body accent.
PALETTES = {
    "dark": dict(
        BG="#0B1220", SURFACE="#111C30", SURFACE_2="#16233A", INPUT="#0B1729",
        BORDER="#1B2638", TEXT="#E2E8F0", MUTED="#94A3B8",
        ACCENT="#38BDF8", ACCENT_DIM="#0369A1",
        GOOD="#34D399", WARN="#F5A524", BAD="#F87171",
        BRAND_TEAL="#2DD4BF", CHART_BG="#111C30",
        VER_BORDER="rgba(56,189,248,0.45)", STATUS_BG="#0B1220",
        PILL_BG="rgba(56, 189, 248, 0.14)", PILL_BORDER="rgba(56, 189, 248, 0.35)",
        GRID="#192436", AXIS_LINE="rgba(255,255,255,0.25)",
        BAND_MODEL="rgba(255,200,150,0.38)", BAND_PRED="rgba(160,210,255,0.38)",
        HL_CL="rgba(56,189,248,0.18)", HL_V="rgba(52,211,153,0.18)"),
    "light": dict(
        BG="#FFFFFF", SURFACE="#F8FAFC", SURFACE_2="#EEF2F7",
        BORDER="#CBD5E1", TEXT="#0F172A", MUTED="#475569",
        ACCENT="#0284C7", ACCENT_DIM="#0369A1",
        GOOD="#059669", WARN="#D97706", BAD="#DC2626",
        BRAND_TEAL="#0D9488", CHART_BG="#FFFFFF",
        VER_BORDER="rgba(2,132,199,0.45)", STATUS_BG="#F1F5F9",
        PILL_BG="rgba(2, 132, 199, 0.10)", PILL_BORDER="rgba(2, 132, 199, 0.35)",
        GRID="rgba(15,23,42,0.10)", AXIS_LINE="rgba(15,23,42,0.45)",
        BAND_MODEL="rgba(251,146,60,0.22)", BAND_PRED="rgba(56,189,248,0.22)",
        HL_CL="rgba(2,132,199,0.12)", HL_V="rgba(5,150,105,0.12)"),
}
LIGHT_MODE = bool(st.session_state.get("light_mode", False))
# Active Streamlit widget theme (--theme.* / config / default = light). Dataframe canvases cannot be
# recolored via CSS, so colors are inverted only when the theme and display mode differ.
_ST_THEME = (getattr(getattr(st.context, "theme", None), "type", None)
             or st.get_option("theme.base") or "light")
_DF_INVERT = _ST_THEME != ("light" if LIGHT_MODE else "dark")
PAL = PALETTES["light" if LIGHT_MODE else "dark"]
BG, SURFACE, SURFACE_2 = PAL["BG"], PAL["SURFACE"], PAL["SURFACE_2"]
BORDER, TEXT, MUTED = PAL["BORDER"], PAL["TEXT"], PAL["MUTED"]
ACCENT, ACCENT_DIM = PAL["ACCENT"], PAL["ACCENT_DIM"]
GOOD, WARN, BAD = PAL["GOOD"], PAL["WARN"], PAL["BAD"]
BRAND_TEAL = PAL["BRAND_TEAL"]
CHART_BG = PAL["CHART_BG"]

# Light mode: the Streamlit widget theme cannot change at runtime, so when enabled, CSS overrides
# inputs, selectboxes, tabs, toggles, tables and the mode bar.
_DF_INVERT_CSS = ("""div[data-testid="stDataFrame"] canvas, div[data-testid="stDataFrame"] .dvn-scroller {
  filter: invert(1) hue-rotate(180deg); }
""" if _DF_INVERT else "")
_LIGHT_WIDGET_CSS = f"""
.stApp, header[data-testid="stHeader"] {{ background: {BG} !important; color: {TEXT}; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] span {{ color: {TEXT} !important; }}
div[data-testid="stWidgetLabel"] p, div[data-testid="stMarkdownContainer"],
div[data-testid="stExpander"] summary p, h1, h2, h3, h4 {{ color: {TEXT}; }}
div[data-testid="stTooltipIcon"] svg {{ color: {MUTED} !important; }}
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] {{
  background: {SURFACE} !important; border-color: {BORDER} !important; }}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input,
[data-testid="stNumberInputField"], [data-testid="stTextInputField"] {{
  background: {SURFACE} !important; color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important;
  caret-color: {TEXT} !important; }}
.stApp input::placeholder, .stApp textarea::placeholder {{ color: {MUTED} !important;
  -webkit-text-fill-color: {MUTED} !important; opacity: 1; }}
[data-testid="stSelectbox"] [role="group"] svg, [data-testid="stMultiSelect"] [role="group"] > button svg,
[data-testid="stMultiSelect"] [role="group"] > div > button svg {{ color: {MUTED} !important; }}
[data-testid="stMultiSelectTagsContainer"] [data-tag] {{ background: {ACCENT} !important; color: #fff !important; }}
[data-testid="stMultiSelectTagsContainer"] [data-tag] * {{ color: #fff !important; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{
  background: {SURFACE_2} !important; color: {TEXT} !important; }}
[data-testid="stNumberInputStepDown"]:hover, [data-testid="stNumberInputStepUp"]:hover {{
  background: {ACCENT} !important; color: #fff !important; }}
[data-testid="stNumberInputStepDown"]:disabled, [data-testid="stNumberInputStepUp"]:disabled {{
  background: {SURFACE_2} !important; color: {MUTED} !important; opacity: 0.45; }}
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectVirtualDropdown"],
div[role="listbox"] {{ background: {BG} !important; border-color: {BORDER} !important;
  box-shadow: 0 4px 14px rgba(15,23,42,0.12) !important; }}
div[role="listbox"] [role="option"], div[role="listbox"] [role="option"] * {{ color: {TEXT} !important; }}
div[role="listbox"] [role="option"][data-focused="true"], div[role="listbox"] [role="option"]:hover {{
  background: {SURFACE_2} !important; }}
[data-testid="stCheckbox"] label > span + div {{ background: {BORDER} !important; }}
[data-testid="stCheckbox"] label:has(input:checked) > span + div {{ background: {ACCENT} !important; }}
[data-testid="stCheckbox"] label > span + div > div {{ background: #FFFFFF !important;
  box-shadow: 0 1px 2px rgba(15,23,42,0.35); }}
.stTabs [role="tab"] p {{ color: {MUTED} !important; }}
.stTabs [role="tab"][aria-selected="true"] p {{ color: {TEXT} !important; }}
.stTabs .react-aria-SelectionIndicator {{ background: {ACCENT} !important; }}
.stTabs [role="tablist"] {{ border-color: {BORDER} !important; }}
div[data-testid="stExpander"] details {{ background: {SURFACE} !important; border-color: {BORDER} !important; }}
div[data-testid="stExpander"] summary {{ background: {SURFACE} !important; color: {TEXT} !important; }}
div[data-testid="stExpander"] summary:hover {{ background: {SURFACE_2} !important; }}
div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{ color: {MUTED} !important; }}
div[data-testid="stExpander"] summary:hover p, div[data-testid="stExpander"] summary:hover span {{
  color: {ACCENT} !important; }}
div[data-testid="stRadio"] p {{ color: {TEXT}; }}
div[data-testid="stTable"] td, div[data-testid="stTable"] th {{ border-color: {BORDER} !important; }}
{_DF_INVERT_CSS}[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background: rgba(5,150,105,0.10) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background: rgba(217,119,6,0.12) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background: rgba(2,132,199,0.10) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background: rgba(220,38,38,0.10) !important; }}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentSuccess"] * {{ color: #065F46 !important; }}
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentWarning"] * {{ color: #92400E !important; }}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] * {{ color: #075985 !important; }}
[data-testid="stAlertContentError"], [data-testid="stAlertContentError"] * {{ color: #991B1B !important; }}
div[data-testid="stForm"] {{ border-color: {BORDER} !important; }}
[data-testid="stElementToolbar"], [data-testid="stElementToolbar"] * {{
  background: {SURFACE} !important; color: {MUTED} !important; }}
.js-plotly-plot .modebar-btn path {{ fill: {MUTED} !important; }}
.js-plotly-plot .modebar-btn:hover path, .js-plotly-plot .modebar-btn.active path {{ fill: {ACCENT} !important; }}
.js-plotly-plot .modebar {{ background: transparent !important; }}
.js-plotly-plot .legendtext, .js-plotly-plot .legendtitletext, .js-plotly-plot .xtick text,
.js-plotly-plot .ytick text, .js-plotly-plot .xtitle, .js-plotly-plot .ytitle {{ fill: {TEXT} !important; }}
[role="tooltip"], [data-testid="stTooltipContent"] {{ background: {BG} !important; color: {TEXT} !important;
  border: 1px solid {BORDER} !important; box-shadow: 0 4px 14px rgba(15,23,42,0.15) !important; }}
[data-testid="stTooltipContent"] * {{ color: {TEXT} !important; background: transparent !important; }}
""" if LIGHT_MODE else ""

# Dark mode: pin label, input, select, list, tab, toggle, alert and chart text colors to the palette
# so text stays readable whatever the Streamlit widget theme is (e.g. default light with no
# --theme).
_DARK_WIDGET_CSS = f"""
.stApp, header[data-testid="stHeader"] {{ background: {BG} !important; color: {TEXT}; }}
[role="tooltip"], [data-testid="stTooltipContent"] {{ background: {SURFACE} !important; color: {TEXT} !important;
  border: 1px solid {BORDER} !important; }}
[data-testid="stTooltipContent"] * {{ color: {TEXT} !important; background: transparent !important; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] span {{ color: {TEXT} !important; }}
div[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] p,
.stSelectbox label, .stMultiSelect label, .stNumberInput label, .stTextInput label, .stRadio label {{
  color: {TEXT} !important; }}
div[data-testid="stMarkdownContainer"], div[data-testid="stExpander"] summary p, h1, h2, h3, h4 {{ color: {TEXT}; }}
div[data-testid="stTooltipIcon"] svg {{ color: {MUTED} !important; }}
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-testid="stNumberInputContainer"], [data-testid="stTextInputRootElement"] {{
  background: {PAL['INPUT']} !important; border-color: {BORDER} !important; }}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input,
[data-testid="stNumberInputField"], [data-testid="stTextInputField"] {{
  background: {PAL['INPUT']} !important; color: {TEXT} !important; -webkit-text-fill-color: {TEXT} !important;
  caret-color: {TEXT} !important; }}
.stApp input::placeholder, .stApp textarea::placeholder {{ color: {MUTED} !important;
  -webkit-text-fill-color: {MUTED} !important; opacity: 1; }}
[data-testid="stSelectbox"] [role="group"] svg, [data-testid="stMultiSelect"] [role="group"] svg {{ color: {MUTED} !important; }}
[data-testid="stMultiSelectTagsContainer"] [data-tag] {{ background: {ACCENT_DIM} !important; color: #fff !important; }}
[data-testid="stMultiSelectTagsContainer"] [data-tag] * {{ color: #fff !important; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ color: {TEXT} !important; }}
[data-testid="stNumberInputStepDown"]:disabled, [data-testid="stNumberInputStepUp"]:disabled {{
  color: {MUTED} !important; opacity: 0.45; }}
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stMultiSelectVirtualDropdown"],
div[role="listbox"] {{ background: {BG} !important; border-color: {BORDER} !important; }}
div[role="listbox"] [role="option"], div[role="listbox"] [role="option"] * {{ color: {TEXT} !important; }}
div[role="listbox"] [role="option"][data-focused="true"], div[role="listbox"] [role="option"]:hover {{
  background: {SURFACE_2} !important; }}
[data-testid="stCheckbox"] label > span + div {{ background: {BORDER} !important; }}
[data-testid="stCheckbox"] label:has(input:checked) > span + div {{ background: {ACCENT} !important; }}
[data-testid="stCheckbox"] label > span + div > div {{ background: {TEXT} !important; }}
.stTabs [role="tab"] p {{ color: {TEXT} !important; }}
.stTabs .react-aria-SelectionIndicator {{ background: {ACCENT} !important; }}
div[data-testid="stExpander"] details {{ background: {SURFACE} !important; border-color: {BORDER} !important; }}
div[data-testid="stExpander"] summary {{ color: {TEXT} !important; }}
div[data-testid="stExpander"] summary [data-testid="stIconMaterial"] {{ color: {MUTED} !important; }}
div[data-testid="stRadio"] p {{ color: {TEXT}; }}
div[data-testid="stTable"] td, div[data-testid="stTable"] th {{ border-color: {BORDER} !important; }}
{_DF_INVERT_CSS}[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{ background: rgba(52,211,153,0.15) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{ background: rgba(245,165,36,0.15) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{ background: rgba(56,189,248,0.15) !important; }}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{ background: rgba(248,113,113,0.15) !important; }}
[data-testid="stAlertContentSuccess"], [data-testid="stAlertContentSuccess"] * {{ color: #A7F3D0 !important; }}
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentWarning"] * {{ color: #FDE68A !important; }}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] * {{ color: #BAE6FD !important; }}
[data-testid="stAlertContentError"], [data-testid="stAlertContentError"] * {{ color: #FECACA !important; }}
.js-plotly-plot .modebar-btn path {{ fill: {MUTED} !important; }}
.js-plotly-plot .modebar-btn:hover path, .js-plotly-plot .modebar-btn.active path {{ fill: {ACCENT} !important; }}
.js-plotly-plot .modebar {{ background: transparent !important; }}
.js-plotly-plot .legendtext, .js-plotly-plot .legendtitletext, .js-plotly-plot .xtick text,
.js-plotly-plot .ytick text, .js-plotly-plot .xtitle, .js-plotly-plot .ytitle {{ fill: #CBD5E1 !important; }}
[data-testid="stElementToolbar"], [data-testid="stElementToolbar"] * {{
  background: {SURFACE} !important; color: {MUTED} !important; }}
""" if not LIGHT_MODE else ""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Geist:wght@400;500;600&display=swap');
:root {{
  --bg: {BG}; --surface: {SURFACE}; --surface2: {SURFACE_2}; --border: {BORDER};
  --text: {TEXT}; --muted: {MUTED}; --accent: {ACCENT}; --good: {GOOD};
  --warn: {WARN}; --bad: {BAD};
  --btn: {ACCENT if LIGHT_MODE else ACCENT_DIM};   /* filled buttons + tags: white text needs a deep blue in dark mode */
}}
html {{ font-size: 19px; }}
body, .stApp {{ background: var(--bg); color: var(--text); }}
* {{ font-family: Inter, -apple-system, "Segoe UI", Pretendard, sans-serif; }}

/* typography — :not([style]) so custom-sized inline elements (brand header,
   badges) aren't force-reset to body size */
.stMarkdown p, .stMarkdown li, div[data-testid="stText"],
.stMarkdown span:not([style]):not([class*="kx-"]):not(.kx-head *):not(.kx-status *):not(.kx-kpi *) {{
  font-size: 1.05rem !important; line-height: 1.55; color: var(--text); }}
/* grey explanatory text (captions + ? tooltips) — one style everywhere, Geist */
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p,
div[data-testid="stCaptionContainer"] li, div[data-testid="stCaptionContainer"] span, .stCaption {{
  font-family: Geist, Inter, -apple-system, "Segoe UI", sans-serif !important;
  font-size: 0.9rem !important; line-height: 1.6 !important; color: var(--muted) !important;
  letter-spacing: 0; }}
[data-testid="stTooltipContent"], [data-testid="stTooltipContent"] * {{
  font-family: Geist, Inter, -apple-system, "Segoe UI", sans-serif !important; line-height: 1.6; }}
/* section 5 regimen line — important: black (light) / white (dark), not grey */
.st-key-kx_regimen div[data-testid="stCaptionContainer"],
.st-key-kx_regimen div[data-testid="stCaptionContainer"] * {{ color: {"#000000" if LIGHT_MODE else "#FFFFFF"} !important; }}
h1 {{ font-size: 2.1rem !important; font-weight: 600; letter-spacing: -0.01em; }}
h2, .stSubheader p {{ font-size: 1.3rem !important; font-weight: 600;
  letter-spacing: 0; border-top: 1px solid var(--border); padding-top: 22px; margin-top: 8px; }}
h3 {{ font-size: 1.12rem !important; font-weight: 600; letter-spacing: 0; }}

/* app header — wordmark + version + model status */
.kx-head {{ display: flex; align-items: center; justify-content: space-between;
  gap: 10px 16px; flex-wrap: wrap; padding: 14px 0 12px; margin-bottom: 14px;
  border-bottom: 1px solid var(--border); }}
.kx-brand {{ display: flex; align-items: center; gap: 16px; }}
.kx-titles {{ display: flex; flex-direction: column; justify-content: center; gap: 5px; }}
.kx-title {{ display: flex; align-items: center; gap: 10px; }}
.kx-word {{ font-size: 34px; font-weight: 700; letter-spacing: 0.01em;
  color: var(--text); line-height: 1.1; }}
.kx-word2 {{ font-size: 30px; font-weight: 700; letter-spacing: 0.01em;
  color: var(--text); line-height: 1.1; }}
.kx-ver {{ font-size: 13px; font-weight: 600; letter-spacing: 0.06em;
  color: var(--accent); border: 1px solid {PAL["VER_BORDER"]};
  border-radius: 4px; padding: 2px 6px; line-height: 1.2; }}
.kx-sub {{ font-size: 13.5px; font-weight: 500; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); line-height: 1.4; }}
/* right margin = room for the Light mode toggle at the right end of the logo row */
.kx-chips {{ display: flex; gap: 8px; flex-wrap: wrap; margin-right: 212px; }}
.kx-chip {{ font-size: 11.5px; color: var(--muted); background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 9px;
  font-variant-numeric: tabular-nums; white-space: nowrap; }}
.kx-chip b {{ color: var(--text); font-weight: 600; }}

/* status bar — fixed to the bottom of the viewport */
.kx-status {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
  display: flex; gap: 18px; align-items: center; padding: 5px 18px;
  background: {PAL["STATUS_BG"]}; border-top: 1px solid var(--border);
  font-size: 11.5px; color: var(--muted); letter-spacing: 0.02em;
  font-variant-numeric: tabular-nums; }}
.kx-status .dot {{ width: 7px; height: 7px; border-radius: 50%;
  background: var(--good); display: inline-block; margin-right: 6px; }}
.kx-status .sp {{ flex: 1; }}
.block-container {{ padding-bottom: 64px !important; }}

/* KPI cards */
.kx-kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px; margin: 6px 0 14px; }}
.kx-kpi {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 12px 9px; min-width: 0; }}
.kx-kpi .k {{ font-size: 12px; font-weight: 600; letter-spacing: 0.02em;
  color: var(--muted); white-space: nowrap; }}
.kx-kpi .v {{ font-size: 1.35rem; font-weight: 600; color: var(--text);
  font-variant-numeric: tabular-nums; margin-top: 2px; white-space: nowrap; }}
.kx-kpi .u {{ font-size: 0.78rem; color: var(--muted); margin-left: 4px; font-weight: 500; }}
@media (max-width: 900px) {{ .kx-kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }} }}

/* download buttons — same look as Run prediction, a little below the chart */
[class*="st-key-kxdl_"] {{ margin-top: 14px; }}
.stDownloadButton button {{ background: var(--btn) !important; color: #fff !important;
  border: none !important; border-radius: 8px !important;
  font-size: 0.85rem !important; font-weight: 600 !important; padding: 6px 16px !important; }}
.stDownloadButton button * {{ color: #fff !important; }}
.stDownloadButton button:hover {{ filter: brightness(1.12); }}

/* lines under the CL / V cards — one block, even spacing */
.kx-below {{ margin: -4px 0 18px; display: flex; flex-direction: column; gap: 3px;
  font-size: 13px; line-height: 1.35; color: var(--muted); font-variant-numeric: tabular-nums; }}
.kx-below > div {{ margin: 0; }}

/* metrics */
div[data-testid="stMetric"] {{
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px 10px; }}
div[data-testid="stMetricValue"] {{ font-size: 1.9rem !important; font-weight: 650; color: var(--text); }}
div[data-testid="stMetricLabel"] {{ font-size: 0.86rem !important; color: var(--muted) !important;
  text-transform: uppercase; letter-spacing: 0.04em; }}
div[data-testid="stMetricDelta"] {{ font-size: 0.92rem !important; }}

/* tables / dataframes */
div[data-testid="stTable"] table, div[data-testid="stDataFrame"] {{
  border: 1px solid var(--border) !important; border-radius: 8px; overflow: hidden; }}
div[data-testid="stTable"] table, div[data-testid="stDataFrame"] * {{
  font-size: 0.97rem !important; color: var(--text); }}
div[data-testid="stTable"] thead th {{
  background: var(--surface2) !important; color: var(--muted) !important;
  text-transform: uppercase; font-size: 0.78rem !important; letter-spacing: 0.03em;
  font-weight: 600 !important; }}
div[data-testid="stTable"] tbody tr:nth-child(odd) {{ background: var(--surface); }}
div[data-testid="stTable"] tbody tr:hover {{ background: var(--surface2); }}

/* inputs */
.stSelectbox label, .stMultiSelect label, .stNumberInput label,
.stTextInput label, .stRadio label {{ font-size: 0.92rem !important; color: var(--muted) !important;
  font-weight: 500; }}
div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{
  background: var(--surface) !important; border-color: var(--border) !important;
  border-radius: 8px !important; font-size: 1rem !important; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid var(--border); }}
.stTabs [data-baseweb="tab"] {{ background: transparent; border-radius: 8px 8px 0 0; padding: 8px 4px; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.05rem !important; font-weight: 550; color: var(--muted); }}
.stTabs [aria-selected="true"] p {{ color: var(--text) !important; }}
.stTabs [aria-selected="true"] {{ border-bottom: 2px solid var(--accent); }}

/* buttons */
.stButton button, .stFormSubmitButton button {{
  background: var(--btn) !important; color: #fff !important; border: none !important;
  border-radius: 8px !important; font-weight: 600 !important; font-size: 0.98rem !important; }}
.stButton button *, .stFormSubmitButton button * {{ color: #fff !important; }}
.stButton button:hover, .stFormSubmitButton button:hover {{ filter: brightness(1.12); }}

/* expander */
div[data-testid="stExpander"] {{
  background: var(--surface); border: 1px solid var(--border) !important; border-radius: 10px; }}
div[data-testid="stExpander"] summary p {{ font-size: 1.02rem !important; font-weight: 600; }}

/* alerts */
div[data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid var(--border); font-size: 0.98rem; }}

/* section divider between drug groups */
hr {{ border-color: var(--border) !important; }}

/* best-algorithm pill — replaces the old "★" glyph with a plain, labeled tag */
.best-pill {{
  display: inline-flex; align-items: center; gap: 6px; margin-top: 6px;
  padding: 3px 10px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
  background: {PAL["PILL_BG"]}; color: var(--accent);
  border: 1px solid {PAL["PILL_BORDER"]}; }}
.best-pill::before {{ content: ""; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); display: inline-block; }}
{_LIGHT_WIDGET_CSS}{_DARK_WIDGET_CSS}</style>
""", unsafe_allow_html=True)


APP_VERSION = "0.90"


def _logo_html(height):
    """Mark + KINENTRIX wordmark logo (lockup).

    assets/kinentrix_lockup_dark.png (dark mode: light text) / _light.png (light mode: navy
    text). Both have transparent backgrounds. Falls back to the mark alone if the file is missing.
    """
    import base64
    f = _ASSETS / f"kinentrix_lockup_{'light' if LIGHT_MODE else 'dark'}.png"
    if not f.exists():
        f = _LOGO
    if f is None or f.suffix != ".png":
        return ""
    b64 = base64.b64encode(f.read_bytes()).decode()
    return (f"<img class='kx-lockup' alt='KINENTRIX' src='data:image/png;base64,{b64}' "
            f"style='height:{height}px;width:auto;display:block'>")


# --- Deployment (Render) ---
# If KINENTRIX_PASSWORD is set, the password screen is shown first. Otherwise behavior is unchanged.
# If KINENTRIX_DEMO_TOP5="1", only Top-5 drugs can be selected in Existing Drugs (the full list
# stays visible).
_PASSWORD = os.environ.get("KINENTRIX_PASSWORD", "")
# Public build (Top-5 data only): appdata carries a "public" block -> the Top-5 lock is always on.
PUBLIC = appdata.load().get("public")
DEMO_TOP5 = os.environ.get("KINENTRIX_DEMO_TOP5", "") == "1" or bool(PUBLIC)
TOP5_INN = ("Pembrolizumab", "Canakinumab", "Dupilumab", "Nivolumab", "Avelumab")

if _PASSWORD and not st.session_state.get("auth_ok"):
    st.html("<style>[data-testid='InputInstructions'] { display: none; }</style>")
    _gate = st.columns([1, 1, 1])[1]
    with _gate:
        st.markdown("<div style='display:flex;justify-content:center;margin:12vh 0 18px'>"
                    f"{_logo_html(96)}</div>", unsafe_allow_html=True)
        with st.form("kx_login"):
            _pw = st.text_input("Password", type="password", placeholder="Password",
                                label_visibility="collapsed")
            _go = st.form_submit_button("Enter", width="stretch")
        if _go:
            if hmac.compare_digest(_pw.encode("utf-8"), _PASSWORD.encode("utf-8")):
                st.session_state["auth_ok"] = True
                st.rerun()
            st.error("Incorrect password")
    st.stop()


def _model_summary():
    """Confirmed model's training drug count, cross-validated R2 and GMFE. Shared by the header and status bar."""
    out = {}
    for tg in ("CL", "V"):
        r = RES[tg]
        m = r["metrics"].get(r["algo"], {})
        out[tg] = {"n": RES["n_train_by_target"][tg], "r2": m.get("r2"),
                   "gmfe": m.get("gmfe"), "label": model_label(r["algo"])}
    return out


def brand_header():
    """Wordmark + version badge + one-line description, with confirmed-model performance chips on the right."""
    ms = _model_summary()
    chips = "".join(
        f"<span class='kx-chip'>{tg} · {ms[tg]['label']} · n <b>{ms[tg]['n']}</b>"
        f" · GMFE <b>{ms[tg]['gmfe']:.2f}×</b></span>"
        for tg in ("CL", "V"))
    st.markdown(
        f"<div class='kx-head'>"
        f"<div class='kx-brand'>{_logo_html(104)}"
        f"<div class='kx-titles'><div class='kx-title'>"
        f"<span class='kx-word'>KINENTRIX</span>"
        f"<span class='kx-word2'>Biologics</span>"
        f"<span class='kx-ver'>v{APP_VERSION}</span></div>"
        f"<span class='kx-sub'>HUMAN PK PREDICTION FOR ANTIBODY THERAPEUTICS</span></div>"
        f"</div><div class='kx-chips'>{chips}</div></div>",
        unsafe_allow_html=True)


def status_bar():
    """Fixed bottom status bar: version, model build date, training size."""
    ms = _model_summary()
    st.markdown(
        f"<div class='kx-status'><span><span class='dot'></span>Ready</span>"
        f"<span>KINENTRIX Biologics v{APP_VERSION}</span>"
        f"<span>DB drugs {PUBLIC['n_db'] if PUBLIC else len(ds['base'])}</span>"
        f"<span class='sp'></span><span>© 2026 APLUS Simulation. All rights reserved.</span></div>",
        unsafe_allow_html=True)

FLAG_COLOR = {"green": GOOD, "yellow": WARN, "red": BAD, "gray": MUTED}
# Traffic-light thresholds differ per target (out-of-fold fold-error quantiles stored in the
# bundle).
# CL observations span a 1507-fold range and V only 29-fold, so one shared yardstick kept CL
# permanently red. Color now means "predicted well relative to that target".
# A single colored badge is used instead of an emoji circle, so text and color do not say the same
# thing twice.
FLAG_LABEL = {"green": "GREEN", "yellow": "YELLOW", "red": "RED", "gray": "N/A"}

# Display labels for candidate models (internal key -> readable name).
# Names share one format: `Algorithm (Hybrid)` / `Algorithm (PureML)`.
_ALGO_LABEL = {"RandomForest": "RandomForest", "ExtraTrees": "ExtraTrees",
               "CatBoost": "CatBoost", "XGBoost": "XGBoost",
               "LightGBM": "LightGBM", "HistGB": "HistGB",
               "GradientBoosting": "GradientBoosting",
               "Ridge": "Ridge", "Lasso": "Lasso", "ElasticNet": "ElasticNet"}


def model_label(name):
    """Internal key to display name. allometry_only stays as Allometry."""
    if not name:
        return "—"
    if name == "allometry_only":
        return "Allometry"
    pure = name.endswith("_pure")
    algo = name[:-5] if pure else name
    algo = _ALGO_LABEL.get(algo, algo)
    return f"{algo} ({'PureML' if pure else 'Hybrid'})"


class _LabelMap(dict):
    """Accepts existing MODEL_LABEL.get(name, name)-style calls unchanged."""

    def get(self, k, default=None):
        return model_label(k)

    def __getitem__(self, k):
        return model_label(k)


MODEL_LABEL = _LabelMap()

# Display labels for method families (baseline = formula only / residual = formula + ML correction /
# direct = ML predicts directly).
KIND_LABEL = {"baseline": "Allometry", "residual": "ML (residual correction)",
              "direct": "ML (Pure/direct)"}


def style_axes(fig):
    """Shared PK chart axis style: log ticks as real numbers, horizontal/vertical grid, border.

    Two fixes together.
      1. Log-axis numbers: major ticks at 10x spacing (0.1, 1, 10, 100...) are labeled with real
         numbers (tickformat="~g", no SI abbreviation); 2-9 get short unlabeled ticks. 1-2-5 ticks
         (dtick="D2") put 20 at 30% between 10 and 100, which looks uneven. Grid lines only on major ticks.
      2. Grid + border: without grid lines the curve looks like it floats. Both axes get a faint
         grid, and a border (showline+mirror) frames the plot.
    """
    # Major ticks at 10x spacing only; 2-9 get short unlabeled ticks.
    fig.update_yaxes(
        type="log", dtick=1, tickformat="~g",
        showgrid=True, gridcolor=PAL["GRID"], gridwidth=1,
        minor=dict(dtick="D1", showgrid=False, ticks="outside", ticklen=3,
                   tickcolor=PAL["AXIS_LINE"]),
        showline=True, linecolor=PAL["AXIS_LINE"], mirror=True, zeroline=False)
    fig.update_xaxes(
        showgrid=True, gridcolor=PAL["GRID"], gridwidth=1,
        showline=True, linecolor=PAL["AXIS_LINE"], mirror=True, zeroline=False,
        ticks="outside", ticklen=4, tickcolor=PAL["AXIS_LINE"])
    fig.update_layout(
        plot_bgcolor=CHART_BG, paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family='Inter, -apple-system, "Segoe UI", sans-serif', size=13, color=TEXT),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=BORDER,
                        font=dict(color=TEXT, size=12)),
        hovermode="x unified",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=ACCENT),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13.5)))
    return fig


_INTER_TTF = _ASSETS / "fonts" / "Inter.ttf"


def _legend_text_w(s, size=13.5):
    """Legend text width in px, measured with the same Inter font file as the screen; estimated from character count if missing."""
    try:
        from matplotlib.ft2font import FT2Font
        f = FT2Font(str(_INTER_TTF))
        f.set_size(size, 72)
        f.set_text(s)
        return f.get_width_height()[0] / 64 * 1.02      # browser rendering is about 2 % wider
    except Exception:
        return len(s) * 6.9


def space_legend(fig):
    """Space horizontal legend items a uniform 50px apart (names unchanged, only cell width). The last item of each
    legend uses its own width so centering does not drift to one side."""
    last = {}
    for tr in fig.data:
        if tr.name and tr.showlegend is not False:
            # plotly item cell = legendwidth + 45px, holding a 35px symbol + text -> gap = width +
            # 10 - text
            tr.legendwidth = int(round(_legend_text_w(str(tr.name)) + 40))
            last[tr.legend or "legend"] = tr
    for tr in last.values():
        tr.legendwidth = None
    return fig


def set_linear_y(fig):
    """Log-axis toggle off: linear y axis starting at 0."""
    fig.update_yaxes(type="linear", dtick=None, tickformat="~g", rangemode="tozero",
                     range=None, autorange=True, minor=dict(dtick=None, ticks=""))
    return fig


def kpi_row(cl, v, regimen, model_row=None, newdrug_tab=False):
    """Compute steady-state exposure metrics from predicted CL and V and show them as six cards.

    CL, V, t1/2, AUCtau, Cmax, Ctrough. Uses the same 1-compartment parameters as the curve.
    """
    if not cl or not v or cl <= 0 or v <= 0:
        return None
    if newdrug_tab:      # same settings as the curve (src/newdrug: SC F, 1-compartment)
        m = newdrug.exposure(cl, v, regimen)
    else:
        p = _with_ka(pksim.params_from_prediction(cl, v, model_row), regimen)
        m = pksim.exposure_metrics(p, regimen["dose_mg"], regimen["interval_days"],
                                   regimen.get("infusion_days", 0.0), regimen["route"],
                                   single_dose=bool(regimen.get("single_dose")))
    ss = m["kind"] == "ss"
    approx = "≈" if m.get("capped") else ""

    def _kv(x):
        """Card number: thousands separators instead of exponent notation (1.74e+03) for large values."""
        return f"{x:,.0f}" if (x is not None and np.isfinite(x) and x >= 1000) else _sig(x)
    # New Drug tab uses newdrug.terminal_thalf (equals ln2*V/CL for 1-compartment).
    th = (newdrug.terminal_thalf(cl, v, regimen["route"]) if newdrug_tab
          else predicted_thalf(cl, v))
    cards = [("CL", _sig(cl), "L/day"), ("V", _sig(v), "L"),
             ("t½", _sig(th), "day"),
             ("AUCτ,ss" if ss else "AUC0–last", approx + _kv(m["AUC"]), "µg·day/mL"),
             ("Cmax,ss" if ss else "Cmax", approx + _kv(m["Cmax"]), "µg/mL"),
             ("Ctrough,ss" if ss else "Ctrough",
              approx + _sig(m["Ctrough"]) if m["Ctrough"] is not None else "—", "µg/mL")]
    st.markdown("<div class='kx-kpis'>" + "".join(
        f"<div class='kx-kpi'><div class='k'>{k}</div>"
        f"<div class='v'>{val}<span class='u'>{u}</span></div></div>"
        for k, val, u in cards) + "</div>", unsafe_allow_html=True)
    return m


_PLOT_CFG = {"displaylogo": False, "displayModeBar": True,   # always show the Reset icon
             "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
             "toImageButtonOptions": {"format": "png", "scale": 2,
                                      "filename": "kinentrix_profile"}}


def chart_block(fig, key, params=None, title=""):
    """One chart + log-axis toggle + CSV/PNG export. Shared by both tabs."""
    top = st.columns([3, 1])
    log = top[1].toggle("Log scale", value=True, key=f"{key}_log")
    if not log:
        set_linear_y(fig)
    st.plotly_chart(fig, width="stretch", config=_PLOT_CFG, key=f"{key}_chart")
    dl = st.container(key=f"kxdl_{key}", horizontal=True, horizontal_alignment="center",
                      gap="medium")   # three buttons side by side, centered
    csv_fig = fig
    if DEMO_TOP5 and not PUBLIC:   # full data behind the Top-5 lock: do not export reference-curve values
        csv_fig = go.Figure(fig)
        csv_fig.data = [tr for tr in csv_fig.data if tr.legendgroup not in ("mol", "pk")]
    dl.download_button("Profile CSV", exporting.profile_csv(csv_fig),
                       file_name=f"{key}_profile.csv", mime="text/csv",
                       key=f"{key}_csv")
    if params is not None:
        dl.download_button("Parameters CSV", exporting.params_csv(params),
                           file_name=f"{key}_parameters.csv", mime="text/csv",
                           key=f"{key}_pcsv")
    png_fig = fig
    if LIGHT_MODE:   # PNG uses the same colors regardless of theme: revert curve/band colors to the dark palette
        to_dark = {v: PALETTES["dark"][k] for k, v in PALETTES["light"].items()}
        png_fig = go.Figure(fig)
        for tr in png_fig.data:
            if isinstance(getattr(tr, "fillcolor", None), str):
                tr.fillcolor = to_dark.get(tr.fillcolor, tr.fillcolor)
            if isinstance(getattr(getattr(tr, "line", None), "color", None), str):
                tr.line.color = to_dark.get(tr.line.color, tr.line.color)
    dl.download_button("Figure PNG", exporting.figure_png(png_fig, title=title, log=log),
                       file_name=f"{key}_profile.png", mime="image/png",
                       key=f"{key}_png")


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "—"


def _sig(x):
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x):
        return "—"
    return f"{x:.3g}"


def _loga(x):
    """Log of an allometry value. Returns NaN (blank) when monkey data is missing.

    About half of the 230 training drugs have no monkey CL/V. The bundle was trained with those
    left blank, so they are passed on blank here too. Hybrid needs allometry, so those drugs
    automatically get a pure-ML answer.
    """
    if x is None or not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0:
        return np.nan
    return float(np.log(x))


def _pos(x):
    """Whether finite and positive (not NaN, None or 0)."""
    try:
        return bool(np.isfinite(float(x)) and float(x) > 0)
    except (TypeError, ValueError):
        return False


def _gmfe(x):
    return f"{x:.2f}×" if isinstance(x, (int, float)) and x is not None else "—"


def _fold(pred, obs):
    """Predicted:Observed fold-error for this drug = max(p/o, o/p)."""
    if pred is None or obs is None or not np.isfinite(pred) or not np.isfinite(obs) \
            or pred <= 0 or obs <= 0:
        return "—"
    return f"{max(pred / obs, obs / pred):.2f}×"


@st.cache_resource(show_spinner="Loading models…")
def load_all():
    """Load the precomputed data and models. Nothing is trained here.

    The results are fixed once per build, so they are generated ahead of time.
    """
    ds = {"base": appdata.base(), "iiv_by_num": appdata.iiv_by_num()}
    res, pred_df = predict.train(ds)
    sim_feats, sim_labels = appdata.similarity_index()
    # public build: the OOD reference is a summary (scaler + threshold) of the full training set
    ood = PUBLIC["ood"] if PUBLIC else predict.build_ood_index(sim_feats)
    return (ds, res, pred_df, appdata.model_idx(), appdata.invitro_idx(),
            sim_feats, sim_labels, ood)


ds, RES, PRED_DF, MODEL_IDX, INV_IDX, SIM_FEATS, SIM_LABELS, OOD = load_all()
# Training-set NUM -> row position (to exclude the drug itself in OOD distance).
TRAIN_ROW = {l.get("NUM"): i for i, l in enumerate(SIM_LABELS)}
OOD_LABEL = "distance from training centre" if PUBLIC else "3-NN distance"


@st.cache_resource(show_spinner=False)
def model_profile_nums():
    """Set of NUMs whose profile can be drawn from published compartment parameters.

    Same condition as the orange dashed line ("Model parameters") overlaid by profile_figure():
    drugs with valid CL and V1 in the literature model table. Shown up front as a (model) tag
    in the dropdown.
    """
    out = set()
    for num, row in MODEL_IDX.items():
        if pksim.params_from_model(row):
            out.add(str(num))
    return out
# Per-drug observed inter-individual variability (IIV) log-SD (from vpop_summary). Drugs without it
# fall back to global constants.
IIV_SD = ds.get("iiv_by_num", {})


def iiv_sd_for(num):
    """IIV log-SD (cl_sd, v_sd) for this drug. Uses the observed CV in vpop if present,
    otherwise falls back to the global C.IIV_CL_SD / C.IIV_V_SD."""
    cl_sd, v_sd = IIV_SD.get(num, (None, None))
    return (cl_sd if cl_sd else C.IIV_CL_SD,
            v_sd if v_sd else C.IIV_V_SD)


def iiv_is_measured(num):
    """Whether this drug's band is based on observed CV (True) or the global fallback (False)."""
    sd = IIV_SD.get(num)
    return bool(sd and (sd[0] or sd[1]))


def ood_check(base_row, inv_row, num=None):
    """Whether this drug is outside the training distribution. Returns (is_ood, distance, threshold)."""
    feat = predict.build_sim_features(base_row, inv_row)
    if PUBLIC:
        # Top-5: exact full-set 3-NN verdicts, precomputed. New inputs: standardised distance
        # from the full training-set centre against its 95th-percentile threshold.
        pre = PUBLIC["ood_existing"].get(str(num)) if num is not None else None
        if pre is not None:
            return bool(pre[0]), pre[1], pre[2]
        if not getattr(C, "OOD_FALLBACK", False):
            return False, None, None
        z = OOD["pipe"].transform(pd.DataFrame([feat]))[0]
        d = float(np.sqrt(np.sum(z ** 2)))
        return bool(d > OOD["threshold"]), d, OOD["threshold"]
    return predict.is_ood(OOD, feat, exclude_idx=TRAIN_ROW.get(num))


def apply_ood_fallback(target, chosen, entries, flagged, is_auto):
    """Fall back to allometry only when OOD and the user did not change the method (default selection).

    If the user manually picked ML, that choice is respected and no fallback happens (warning only).
    Returns (name used, entry, fell_back).
    """
    if not entries:
        return None, {"pred": None, "kind": "baseline", "adj": None}, False
    if chosen not in entries:
        # The dropdown can be empty, giving chosen=None, when that target has no candidates at all.
        # Fall back to the first available one.
        chosen = ("allometry_only" if "allometry_only" in entries
                  else next(iter(entries)))
    if flagged and is_auto and chosen != "allometry_only" and "allometry_only" in entries:
        return "allometry_only", entries["allometry_only"], True
    return chosen, entries[chosen], False


def shrink_to_allo(pred, allo, kind):
    """Shrink the ML prediction toward allometry in log space: exp(w*log(ML) + (1-w)*log(allo)).

    Returned unchanged for baseline (allometry selected) or when the switch is off.
    Returns (final value, shrink_applied). Set config.SHRINK_TO_ALLO=False to disable.
    """
    if not getattr(C, "SHRINK_TO_ALLO", False) or kind == "baseline":
        return pred, False
    if pred is None or allo is None or pred <= 0 or allo <= 0 or not np.isfinite(pred):
        return pred, False
    w = C.SHRINK_W
    return float(np.exp(w * np.log(pred) + (1 - w) * np.log(allo))), True


# Grouping criterion for the list; each drug gets exactly one of four groups.
#   full     both CL and V present
#   partial  not both (only one, or none)
G1 = "1. Full set: Monkey[full data], Human[full data]"
G2 = "2. Partial set: Monkey[full data], Human[partial data]"
G3 = "3. Partial set: Monkey[partial data], Human[full data]"
G4 = "4. Other set: Monkey[full data], Human[no data]"
GROUP_ORDER = (G1, G2, G3, G4)



# ------------------------------------------------------------- (approved) badge
# What it does: appends "(approved)" after the drug name only while the selection list is open;
#     the closed box shows just the drug name.
#     "(approved)" = regimen taken from the reviewed Regimen set; the others come from clinical
#     trial records and may not be approved regimens.
#
# Why JavaScript: st.selectbox's format_func uses the **same text** for the list and the closed
#     box, with no option to differ. So text is appended only when a list item is rendered;
#     the closed box and the selected value are untouched.
#
# If it breaks, the app is fine: if Streamlit's internals change, the badge simply does not
#     appear; drug selection and prediction still work.
# Payload planted in the parent document. It lives outside the frame, so it survives re-renders.
_BADGE_CORE = """
(function () {
  var src = null, map = {};

  function tags() {
    if (window.__bbNames !== src) {          // new name list when the drug set changes
      src = window.__bbNames;
      try { map = JSON.parse(src || '{}') || {}; } catch (e) { map = {}; }
    }
    return map;
  }

  // Re-tag on every mutation: the list recycles option elements, so tags are added or
  // removed to match the current name. Only changed nodes are touched, so the
  // observer does not re-trigger itself.
  function paint() {
    var ok = tags();
    var q = '[role="listbox"][aria-label="' + window.__bbLabel + '"]'
          + ' [role="option"]';
    document.querySelectorAll(q).forEach(function (li) {
      var tag = li.querySelector('.bb-approved');
      var name = (tag ? li.textContent.replace(tag.textContent, '')
                      : li.textContent || '').trim();
      var want = Object.prototype.hasOwnProperty.call(ok, name) ? ok[name] : null;
      if (want && (!tag || tag.textContent !== want)) {
        if (!tag) {
          tag = document.createElement('span');
          tag.className = 'bb-approved';
          tag.style.opacity = '0.55';
          tag.style.fontSize = '0.85em';
          li.appendChild(tag);
        }
        tag.textContent = want;
      } else if (!want && tag) {
        tag.remove();
      }
    });
  }

  window.__bbPaint = paint;
  new MutationObserver(paint).observe(document.body,
      {childList: true, subtree: true, characterData: true});
  paint();
})();
"""

_BADGE_JS = """
<script>
(function () {
  var win, doc;
  try { win = window.parent; doc = win.document; } catch (e) { return; }
  if (!doc || !doc.head) { return; }

  // names and label are passed as plain strings (see note above).
  win.__bbNames = __NAMES__;
  win.__bbLabel = __LABEL__;

  if (!doc.getElementById('bb-approved')) {
    var sc = doc.createElement('script');
    sc.id = 'bb-approved';
    sc.textContent = __CORE__;
    doc.head.appendChild(sc);
  }
  if (win.__bbPaint) { win.__bbPaint(); }
})();
</script>
"""


def approved_badge(tag_map, label="Select drug (drop-down)", extra_html=""):
    """Append a tag only to items of the open list.

    tag_map: {drug name: "  (approved · model)"}; the text is appended as-is. Passing a
    set/list of names (legacy) tags all of them with "  (approved)".
    """
    import json as _json
    import streamlit.components.v1 as _components
    if not isinstance(tag_map, dict):
        tag_map = {n: "  (approved)" for n in tag_map}
    html = (_BADGE_JS
            .replace("__NAMES__", _json.dumps(_json.dumps(tag_map)))
            .replace("__LABEL__", _json.dumps(label))
            .replace("__CORE__", _json.dumps(_BADGE_CORE))) + extra_html
    _components.html(html, height=0)


# Top-5 demo lock: same mechanism as the badge (payload planted in the parent document). Marks list
# items whose name is not allowed with aria-disabled + bb-disabled and blocks press/click/Enter in
# the capture phase.
# Even if blocked here, the server-side check (TAB 1 below) reverts it once more.
_LOCK_CORE = """
(function () {
  var src = null, ok = {};
  function allowed() {
    if (window.__bbAllowed !== src) {
      src = window.__bbAllowed;
      try { ok = JSON.parse(src || '{}') || {}; } catch (e) { ok = {}; }
    }
    return ok;
  }
  function q() { return '[role="listbox"][aria-label="' + window.__bbLabel + '"] [role="option"]'; }
  function nameOf(li) {
    var tag = li.querySelector('.bb-approved');
    return (tag ? li.textContent.replace(tag.textContent, '') : li.textContent || '').trim();
  }
  function paint() {
    var a = allowed();
    document.querySelectorAll(q()).forEach(function (li) {
      var off = !Object.prototype.hasOwnProperty.call(a, nameOf(li));
      if (off !== li.classList.contains('bb-disabled')) {
        li.classList.toggle('bb-disabled', off);
        if (off) { li.setAttribute('aria-disabled', 'true'); } else { li.removeAttribute('aria-disabled'); }
      }
    });
  }
  function block(e) {
    var li = e.target && e.target.closest && e.target.closest('[role="option"].bb-disabled');
    if (li) { e.preventDefault(); e.stopImmediatePropagation(); }
  }
  ['pointerdown', 'pointerup', 'mousedown', 'mouseup', 'click', 'touchstart', 'touchend']
    .forEach(function (t) { window.addEventListener(t, block, true); });
  window.addEventListener('keydown', function (e) {
    if (e.key !== 'Enter' && e.key !== ' ' && e.key !== 'Tab') { return; }
    var li = document.querySelector(q() + '.bb-disabled[data-focused="true"]');
    if (li) { e.preventDefault(); e.stopImmediatePropagation(); }
  }, true);
  var st = document.createElement('style');
  var d = 'div[role="listbox"] div[role="option"].bb-disabled';
  st.textContent = d + '{opacity:.42;cursor:not-allowed !important;filter:grayscale(1)}'
    + d + ',' + d + ':hover,' + d + '[data-focused="true"],' + d + '[data-hovered="true"],'
    + d + ' > div,' + d + ':hover > div,' + d + '[data-focused="true"] > div'
    + '{background:transparent !important}'
    + d + ' *{cursor:not-allowed !important}';
  document.head.appendChild(st);
  window.__bbLockPaint = paint;
  new MutationObserver(paint).observe(document.body,
      {childList: true, subtree: true, characterData: true, attributes: true,
       attributeFilter: ['data-focused']});
  paint();
})();
"""

_LOCK_JS = """
<script>
(function () {
  var win, doc;
  try { win = window.parent; doc = win.document; } catch (e) { return; }
  if (!doc || !doc.head) { return; }
  win.__bbAllowed = __ALLOWED__;
  win.__bbLabel = __LABEL__;
  if (!doc.getElementById('bb-lock')) {
    var sc = doc.createElement('script');
    sc.id = 'bb-lock';
    sc.textContent = __CORE__;
    doc.head.appendChild(sc);
  }
  if (win.__bbLockPaint) { win.__bbLockPaint(); }
})();
</script>
"""


def top5_lock_html(allowed_labels, label="Select drug (drop-down)"):
    """Demo mode: script that greys out list items other than the allowed names and makes them unselectable.

    Attached to the same frame as the badge; an extra frame would push the layout down.
    """
    import json as _json
    return (_LOCK_JS
            .replace("__ALLOWED__", _json.dumps(_json.dumps({n: 1 for n in allowed_labels})))
            .replace("__LABEL__", _json.dumps(label))
            .replace("__CORE__", _json.dumps(_LOCK_CORE)))


def drug_catalog():
    """Drugs that can be shown on screen + the group each belongs to.

    Two kinds are merged.
      - drugs the model was trained on (union of CL 100 + V 214 = 230)
      - drugs with monkey CL/V that did not make it into training (60)

    Why the second kind: training requires an observed human value, so drugs without human data
    are left out of training, yet those are exactly the ones whose human values must be predicted.
    With monkey data, both allometry and ML work.
    """
    B = predict._load()[0]
    trained = set()
    for tg in ("CL", "V"):
        trained |= {str(x) for x in B[tg]["nums"]}
    reg = appdata.regimen_nums()

    b = ds["base"].copy()
    b["NUM"] = b["NUM"].astype(str)

    def pos(col):
        return pd.to_numeric(b.get(col), errors="coerce").gt(0).fillna(False)

    mk_cl, mk_v = pos("monkey_CL_Lday"), pos("monkey_V_L")
    mk_full = mk_cl & mk_v
    in_train = b["NUM"].isin(trained)

    # Drugs to list: those in training, or those with both monkey CL and V. Drugs with only one
    # monkey value that are also not in training (10) have too thin a basis for prediction and are
    # not listed.
    # Drugs excluded for data errors or duplicates (the reason is recorded with each).
    excluded = set(appdata.load().get("excluded", {}))
    keep = (in_train | mk_full) & ~b["NUM"].isin(excluded)
    b = b[keep].copy()
    mk_cl, mk_v, mk_full, in_train = (mk_cl[keep], mk_v[keep],
                                      mk_full[keep], in_train[keep])

    # Observed value: bundle (training target) first, otherwise screen data - same as observed_of.
    def _obs(num, tg, col, row):
        if observed_of(num, tg) is not None:
            return True
        v = pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0]
        return bool(np.isfinite(v) and v > 0)

    grp = []
    for (_, row), tr, mc, mv in zip(b.iterrows(), in_train, mk_cl, mk_v):
        num = row["NUM"]
        o_cl = _obs(num, "CL", "human_CL_Lday", row)
        o_v = _obs(num, "V", "human_V_L", row)
        if not tr:
            grp.append(G4)
        elif not (mc and mv):
            grp.append(G3)
        elif o_cl and o_v:
            grp.append(G1)
        else:
            grp.append(G2)
    b["_group"] = grp
    b["_trained"] = list(in_train)
    # Drugs sharing a name (e.g. Infliximab 53 IV / 166 SC) get a number appended in the list name;
    # others keep the plain name.
    _dup = b["INN"].duplicated(keep=False)
    b["_label"] = [f"{i} [#{n}]" if d else i
                   for i, n, d in zip(b["INN"], b["NUM"], _dup)]
    # Stable sort - drugs with the same name keep source order.
    if PUBLIC:
        # public build: the other drugs are names only (greyed out and locked in the drop-down)
        have = set(b["_label"])
        extra = pd.DataFrame([{"NUM": "", "INN": inn, "_label": lab, "_group": None,
                               "_trained": False}
                              for lab, inn in PUBLIC["catalog"] if lab not in have])
        b = pd.concat([b, extra], ignore_index=True)
        pos = {lab: i for i, (lab, _) in enumerate(PUBLIC["catalog"])}
        return b.sort_values("_label", key=lambda s: s.map(pos), kind="mergesort")
    return b.sort_values("INN", kind="mergesort")


def predictable_drugs():
    """Legacy name: returns everything without grouping."""
    return drug_catalog()


def target_source(num, target):
    """Where this target's prediction comes from.

    "oof"     answer from a model trained without that drug (drug was in that target's training)
    "fullfit" drug not used in that target's training: answer from a model fit on all drugs
    """
    B = predict._load()[0]
    return "oof" if str(num) in {str(x) for x in B[target]["nums"]} \
        else "fullfit"


SOURCE_TAG = {
    "oof": ("out-of-fold", GOOD,
            "held out of this model's own training before being predicted"),
    "fullfit": ("not in training set", ACCENT,
                "never in the training set for this target, so the prediction "
                "comes from a model fitted on all training drugs"),
}


def source_note(num, target):
    """Small source label shown under the prediction."""
    label, color, why = SOURCE_TAG[target_source(num, target)]
    return f"<div style='color:{color}' title='{why}'>{label}</div>"


def flag_badge(fl, names=None):
    """Light CL and V separately, so a bad one does not hide the other's green.

    Passing names={"CL": "ExtraTrees (PureML) ★", "V": ...} also prints the model name inside
    the box, so the source is visible without scrolling to the table above.
    """
    names = names or {}
    cols = st.columns(2)
    for col, tg, unit in ((cols[0], "CL", "L/day"), (cols[1], "V", "L")):
        d = fl.get(tg) or {}
        c = FLAG_COLOR[d.get("flag", "gray")]
        txt = FLAG_LABEL[d.get("flag", "gray")]
        f = d.get("fold")
        cut = d.get("cut") or {}
        sub = (f"{f:.2f}× off the observed value" if f
               else "no observed value to compare")
        rule = (f"green &lt; {cut.get('green', 1.5):.2g}× · "
                f"red &gt; {cut.get('red', 3.0):.2g}×" if cut else "")
        who = names.get(tg)
        who_html = (f"<span style='font-weight:600;font-size:14px'>{who}"
                    f"</span><br>" if who else "")
        col.markdown(
            f"<div style='padding:12px;border-radius:8px;background:{c};"
            f"color:white;font-weight:600;font-size:16px'>{txt} · {tg}<br>"
            f"{who_html}"
            f"<span style='font-weight:400;font-size:14px'>{sub}</span><br>"
            f"<span style='font-weight:400;font-size:12px;opacity:.85'>{rule}"
            f"</span></div>", unsafe_allow_html=True)


def _with_ka(p, regimen):
    """Fill in the absorption rate (ka) for SC when missing.

    Without ka, pksim assumes no drug moves from the subcutaneous depot to blood and sets
    concentration to 0. Many SC drugs in the literature model table lack ka, so their profiles
    were entirely 0 and the log-axis ticks broke.

    Antibody SC absorption half-life is typically 2-4 days, so ka 0.25 /day is the default.
    The screen notes that the default was used.
    """
    if not regimen or not str(regimen.get("route", "")).upper().startswith("S"):
        return p
    if p and not p.get("ka"):
        p = dict(p)
        p["ka"] = getattr(C, "DEFAULT_KA_SC", 0.25)
        p["_ka_default"] = True
    return p


def dose_interval_text(days):
    """Convert the dosing interval to human-readable text.

    Interval in the regimen data is in weeks, so a once-daily drug is 1/7. Rounding that to
    Q{n}W would give Q0W.
    """
    try:
        d = float(days)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(d) or d <= 0:
        return "—"
    if d < 0.95:                                   # several times a day
        h = max(1, int(round(d * 24)))
        return f"every {h} h"
    if d < 1.5:
        return "QD (once daily)"
    if d < 6.5:
        return f"every {int(round(d))} days"
    w = d / 7.0
    if abs(w - round(w)) < 0.06:                   # only when it is an exact number of weeks
        return f"Q{int(round(w))}W"
    return f"every {int(round(d))} days"


def ka_is_default(num, regimen):
    """Whether this drug's profile uses the default absorption rate.

    For SC with no ka in the literature model table, _with_ka() fills in the default.
    Checked ahead of time so the screen can say so.
    """
    if not regimen or not str(regimen.get("route", "")).upper().startswith("S"):
        return False
    p = pksim.params_from_prediction(1.0, 1.0, MODEL_IDX.get(num))
    return not (p and p.get("ka"))


def bundle_allometry(num, target):
    """Allometry prediction for the target as actually used in the bundle's training.

    The bundle builds the allometry base value from the **human body weight stated per drug in
    its source document** (65 kg if absent). The left-side `predict.allometry_*` uses a fixed
    70 kg, so for training-set drugs whose stated weight is not 70, the left Allometry value, the
    Allometry row of the model comparison table (and the value when Allometry is chosen in the
    dropdown) would differ. Training-set drugs use this value to keep them consistent.

    None if the drug is not in the training set (new drug etc.) or the base value is empty
    because monkey data is missing.
    """
    B = predict._load()[0]
    t = B.get(target) or {}
    nums = [str(x) for x in t.get("nums", [])]
    key = str(num)
    if key in nums:
        la = t["allo"][nums.index(key)]
        if la is not None and np.isfinite(la):
            return float(np.exp(la))
    return None


def run_prediction(base_row, num):
    """Allometry+ML prediction for a single drug."""
    a_cl = predict.allometry_cl(base_row.get("monkey_CL_Lday"), base_row.get("monkey_bw"))
    a_v = predict.allometry_v(base_row.get("monkey_V_L"), base_row.get("monkey_bw"))
    # Training-set drugs use the bundle's base value (stated body weight), so the left metric and
    # the model comparison Allometry row show the same value.
    a_cl = bundle_allometry(num, "CL") or a_cl
    a_v = bundle_allometry(num, "V") or a_v
    feat_cl = predict.build_features(base_row, INV_IDX.get(num), _loga(a_cl))
    feat_v = predict.build_features(base_row, INV_IDX.get(num), _loga(a_v))
    corr = predict.apply_correction(RES, feat_cl, feat_v, a_cl, a_v)
    return {"allo_CL": a_cl, "allo_V": a_v,
            "feat_CL": feat_cl, "feat_V": feat_v, **corr}


def target_predictions(target, feat, allo, num=None, noseq=False):
    """If num is in the training set (oof_table), use the out-of-fold prediction (fair, no memorization);
    otherwise (new drug etc. not used in training) use the prediction of the model trained on all data.
    The latter never trained on that target, so it is fair as is.
    """
    if num is not None and num in RES.get("oof_table", {}):
        out = predict.oof_model_predictions(RES, target, num, allo)
        # CL and V have different training drugs (CL 100, V 214). Many drugs exist for only one, so
        # an out-of-fold prediction may be missing for that target; then predict with the model
        # trained on all data. The drug was not used in that target's training, so this is still
        # fair.
        if out:
            return out
    return predict.model_predictions(RES, target, feat, allo, noseq=noseq)


# Core columns as readable names; used to explain why the hybrid does not run.
CORE_LABEL = {"log_allo": "the allometry baseline", "mk_V": "monkey V",
              "log_Kd": "binding affinity (Kd)", "MW": "molecular weight"}


def hybrid_block_reason(feat, target):
    """Inputs blocking the hybrid for this target. Empty list if all are present.

    The build adds allometry only when all four core columns are measured. If any is missing it
    falls back to the direct model, so the screen must follow the same rule.
    """
    cols = RES[target]["columns"]
    out = []
    for c in predict.CORE:
        if c not in cols:
            continue
        v = pd.to_numeric(pd.Series([feat.get(c)]), errors="coerce").iloc[0]
        if not np.isfinite(v):
            out.append(CORE_LABEL.get(c, c))
    return out


def live_hybrids(pred_map):
    """Candidate list keeping only hybrids that actually include allometry.

    A hybrid uses allometry as its base and learns the residual on top. Without a base (no monkey
    CL/V or a core column is empty) it falls back to the direct model by the same rule as the build,
    so its values equal pure ML exactly.

    Showing the same number under two names makes it look like there are thirteen choices.
    Such hybrids are dropped. Returns (kept candidates, dropped names).
    """
    out, dropped = {}, []
    for name, d in pred_map.items():
        twin = pred_map.get(name + "_pure")
        if (d.get("kind") == "residual" and twin
                and d.get("pred") and twin.get("pred")
                and abs(np.log(d["pred"] / twin["pred"])) < 1e-12):
            dropped.append(name)
            continue
        out[name] = d
    return out, dropped


# Plausible range for human CL/V. Candidate predictions outside it (e.g. outliers from linear
# models) are removed from the choices. predict.py is left as is, since the regression_predictions
# reference values live there.
PLAUSIBLE = {"CL": (1e-3, 1e3), "V": (0.1, 1e3)}


def implausible(d, target):
    """Whether this candidate's prediction is finite and positive but outside the PLAUSIBLE range.

    0 and None are not filtered here; they are reported separately as "no value".
    """
    p = d.get("pred")
    lo, hi = PLAUSIBLE[target]
    return bool(isinstance(p, (int, float)) and np.isfinite(p) and p > 0
                and not (lo <= p <= hi))


def plausible_only(pred_map, target):
    """{name: entry} without out-of-range candidates."""
    return {k: d for k, d in pred_map.items() if not implausible(d, target)}


def algo_of(name):
    """Algorithm name only from a model key. allometry_only stays as is."""
    if not name or name == "allometry_only":
        return "allometry_only"
    return name[:-5] if name.endswith("_pure") else name


# Algorithm families. Uses the one carried in the bundle, otherwise this list.
_FAMILY_FALLBACK = {
    "ensemble": ("CatBoost", "XGBoost", "ExtraTrees", "RandomForest",
                 "LightGBM", "HistGB", "GradientBoosting"),
    "linear": ("Ridge", "Lasso", "ElasticNet"),
}


def families():
    """Ensemble / linear family, read as decided by the build."""
    try:
        f = predict._load()[0]["meta"].get("families")
    except Exception:
        f = None
    return f or _FAMILY_FALLBACK


def table_order(names):
    """Table order: Allometry on top, then ensembles, then linear.

    The same algorithm takes two adjacent rows (hybrid first, then pure ML).
    Within a family, ordered by cross-validated R2 (mean of CL and V), best first.
    """
    def r2(nm):
        v = [RES[t]["metrics"].get(nm, {}).get("r2") for t in ("CL", "V")]
        v = [x for x in v if isinstance(x, (int, float))]
        return sum(v) / len(v) if v else -9.0

    fam = families()
    rank = {}
    for i, group in enumerate(("ensemble", "linear")):
        for a in fam.get(group, ()):
            rank[a] = i

    algos, seen = [], set()
    for nm in names:
        a = algo_of(nm)
        if a != "allometry_only" and a not in seen:
            seen.add(a)
            algos.append(a)
    # Family first, then by R2 within the family. Unknown families go last.
    algos.sort(key=lambda a: (rank.get(a, 9), -max(r2(a), r2(a + "_pure"))))

    order = ["allometry_only"] if "allometry_only" in names else []
    for a in algos:
        for nm in (a, a + "_pure"):        # hybrid first, then pure ML
            if nm in names:
                order.append(nm)
    return order


@st.cache_resource(show_spinner=False)
def matched_r2():
    """R2 re-measured for all models using only drugs for which allometry is scored.

    The R2 in the table is over a different drug count per row: allometry exists only for drugs
    with monkey data while ML covers all drugs. Compared directly, allometry looks better than
    it is. Re-measuring on the same drugs reverses the order.
    """
    B = predict._load()[0]
    out = {}
    for tg in ("CL", "V"):
        t = B[tg]
        if "matched_r2" in t:        # public build: precomputed on the full training set
            out[tg] = dict(t["matched_r2"])
            continue
        vt = "esm" if "esm" in t["variants"] else "noesm"
        V = t["variants"][vt]
        y = np.asarray(t["y"], float)
        allo = np.asarray(t["allo"], float)
        m = np.isfinite(allo) & np.isfinite(y)
        d = {}
        for key, p in V["oof"].items():
            k = m & np.isfinite(p)
            if k.sum() < 10:
                continue
            denom = float(np.sum((y[k] - y[k].mean()) ** 2))
            if denom <= 0:
                continue
            d[predict.jsname(key)] = 1 - float(np.sum((y[k] - p[k]) ** 2)) / denom
        out[tg] = d
    return out


def best_by_r2(pred_map, target):
    """Candidate with the highest cross-validated R2 among those selectable for this drug.

    R2 itself is the model's score and does not change with the drug. But the **candidate list
    differs per drug**: without monkey data the hybrid is absent, and a drug not used in that
    target's training has no out-of-fold prediction. So the star moves.

    It is not chosen by comparing to observed values: that would be picking with the answer in
    view and could not be used for new drugs (the confirmed model wins only 11.9% of the time
    for CL, i.e. closer to chance).
    """
    # If allometry is among the candidates, compare on the same drugs; otherwise all values were
    # measured on all drugs, so compare directly.
    use_matched = "allometry_only" in pred_map
    mr = matched_r2().get(target, {}) if use_matched else {}
    best, best_r2 = None, None
    for name, d in pred_map.items():
        p = d.get("pred")
        if p is None or not np.isfinite(p) or p <= 0:
            continue
        r2 = (mr.get(name) if use_matched
              else RES[target]["metrics"].get(name, {}).get("r2"))
        if not isinstance(r2, (int, float)) or not np.isfinite(r2):
            continue
        if best_r2 is None or r2 > best_r2:
            best, best_r2 = name, r2
    return best


def observed_of(num, target):
    """Observed human value for this target. For training drugs, uses the bundle's training target.

    If the screen and training built observed values by different rules, a drug used in training
    would show "no observation" on screen. The bundle holds the value the model actually tried to
    predict, so it is the reference.
    """
    B = predict._load()[0]
    t = B.get(target) or {}
    nums = [str(x) for x in t.get("nums", [])]
    key = str(num)
    if key in nums:
        v = t["obs"][nums.index(key)]
        if v is not None and np.isfinite(v) and v > 0:
            return float(v)
    return None


def _score_of(d):
    """Sort-key score. Goes last if score is missing (candidate training failed etc.)."""
    s = d.get("score")
    return s if isinstance(s, (int, float)) else float("inf")


def _sorted_names(pred_list):
    """Names sorted ascending by LOO-CV composite score (best method first)."""
    return [d["name"] for d in sorted(pred_list, key=_score_of)]


# Cross-validation metrics available in the comparison table; toggled on screen.
#   GMFE      average fold-error; 1.00 is perfect
#   RMSE      scatter in log space; unaffected by the variance of the truth, so good for comparing different drug sets
#   R2        primary metric; but low for targets where the truth is tightly clustered (V)
#   Spearman  whether the ranking is right; measures something different from R2
#   2-fold    fraction within 2-fold; the number most readable in practice
CV_METRICS = {
    "GMFE (typical error)": ("gmfe", lambda x: _gmfe(x)),
    "RMSE (log)": ("rmse", lambda x: _fmt(x)),
    "R²": ("r2", lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else "—"),
    "Spearman ρ": ("spearman",
                   lambda x: f"{x:+.3f}" if isinstance(x, (int, float)) else "—"),
    "within 2-fold": ("within2",
                      lambda x: f"{x:.0f}%" if isinstance(x, (int, float)) else "—"),
    # R2 re-measured on drugs scored for allometry only. The only column that compares allometry and
    # ML side by side (other columns differ in drug count).
    "R² (same drugs)": ("r2_matched",
                        lambda x: f"{x:+.3f}" if isinstance(x, (int, float))
                        else "—"),
}
CV_DEFAULT = ["GMFE (typical error)", "RMSE (log)", "R²", "Spearman ρ"]


def model_comparison_df(feat_cl, feat_v, allo_cl, allo_v, obs_cl=None, obs_v=None,
                        num=None, show=None, star_cl=None, star_v=None, noseq=False):
    """Per-model (each ML calibration + allometry-only) CL/V predictions & metrics.

    Rows are sorted by the average of CL+V composite score, best-performing first.
    CV metrics are 5-fold cross-validation computed once when the models were built —
    the same for every drug. When num is a training-set drug, "Predicted CL/V" and
    "fold vs obs" use out-of-fold predictions (that drug excluded from training) so
    they're a fair check, not memorization.

    star_cl / star_v: method closest to the observed value for this drug. A ★ is added next to the
    fold-error on that row. It is chosen with the answer in view, so it is a post-hoc check only.

    show: which CV metrics to display. None = CV_DEFAULT.
    """
    show = [s for s in (show or CV_DEFAULT) if s in CV_METRICS]
    pc = {d["name"]: d for d in target_predictions("CL", feat_cl, allo_cl, num, noseq)}
    pv = {d["name"]: d for d in target_predictions("V", feat_v, allo_v, num, noseq)}
    # Out-of-range predictions (PLAUSIBLE) cannot be chosen, so the table also shows `—` in the
    # value cell.
    for _m, _tg in ((pc, "CL"), (pv, "V")):
        for _k, _d in list(_m.items()):
            if implausible(_d, _tg):
                _m[_k] = dict(_d, pred=None)
    # Decided only by whether an allometry base value exists.
    #   neither target has one: drop both the Allometry row and the hybrid row (not applicable)
    #   only one target lacks it: show `—` in that target's cell
    def _ok(x):
        return x is not None and np.isfinite(x) and x > 0

    live_cl, live_v = _ok(allo_cl), _ok(allo_v)
    if not (live_cl or live_v):
        for _m in (pc, pv):
            for _k in [k for k, d in _m.items()
                       if d.get("kind") in ("residual", "baseline")]:
                _m.pop(_k)

    _mr = matched_r2()

    def _with_matched(d, tg):
        """Add same-drug-basis R2 to the row."""
        d = dict(d)
        d["r2_matched"] = _mr.get(tg, {}).get(d.get("name"))
        return d

    def _blank(nm, tg):
        """Row with no prediction for that target. Cross-validation metrics belong to the model, so they are kept."""
        m = dict(RES[tg]["metrics"].get(nm, {}))
        m.update(name=nm, kind=RES[tg]["model_kind"].get(nm, "baseline"),
                 pred=None)
        return m

    # Allometry on top, then per algorithm two rows: hybrid and pure.
    # If either target has a base value, the Allometry row is shown (the other becomes `—`). If
    # neither has one it was dropped earlier and is not revived here.
    _keep = {"allometry_only"} if (live_cl or live_v) else set()
    names = table_order(sorted(set(pc) | set(pv) | _keep))
    pc = [_with_matched(pc.get(nm) or _blank(nm, "CL"), "CL") for nm in names]
    rows = []
    for d in pc:
        name = d["name"]
        v = _with_matched(pv.get(name) or _blank(name, "V"), "V")
        # No Type column; the name already carries (Hybrid)/(PureML).
        # The "best for this drug" marker is a cell background highlight at the call site, not a
        # star.
        r = {"Model": MODEL_LABEL.get(name, name)}
        # Rows that rely on allometry (Allometry, hybrid) blank both prediction and metrics when the
        # target has no base value; the method itself does not exist.
        _needs_allo = (d.get("kind") or RES["CL"]["model_kind"].get(name, "")) \
            in ("residual", "baseline")
        _cl_on = live_cl or not _needs_allo
        for lbl in show:                       # CL metrics
            key, fmt = CV_METRICS[lbl]
            r[f"CL {lbl}"] = fmt(d.get(key)) if _cl_on else "—"
        _p = d["pred"] if _cl_on else None
        r["Predicted CL (L/day)"] = _sig(_p)
        r["CL fold vs obs"] = _fold(_p, obs_cl)
        _vk = v.get("kind") or RES["V"]["model_kind"].get(name, "")
        _v_on = live_v or _vk not in ("residual", "baseline")
        for lbl in show:                       # V metrics
            key, fmt = CV_METRICS[lbl]
            r[f"V {lbl}"] = fmt(v.get(key)) if _v_on else "—"
        _q = v.get("pred") if _v_on else None
        r["Predicted V (L)"] = _sig(_q)
        r["V fold vs obs"] = _fold(_q, obs_v)
        rows.append(r)
    return pd.DataFrame(rows)


def iiv_band(cl_pred, v_pred, num, regimen, ref_t, n_doses, horizon):
    """Inter-individual variability (IIV) Monte Carlo band. Samples log-normal individual variability
    around predicted CL/V, draws many profiles, and returns P5-P95 at each time point on the ref_t grid.

    This band is not prediction accuracy but "the spread between individuals when the same drug is given to a population".
    Returns (lo, hi), same length as ref_t; (None, None) on failure.
    """
    if cl_pred is None or v_pred is None or cl_pred <= 0 or v_pred <= 0:
        return None, None
    cl_sd, v_sd = iiv_sd_for(num)  # observed CV for that drug (if any), otherwise global fallback
    rng = np.random.default_rng(int(float(num)) if str(num).replace(".", "").isdigit() else 0)
    mdl = MODEL_IDX.get(num)
    curves = []
    for _ in range(C.IIV_N):
        cl_i = cl_pred * float(np.exp(rng.normal(0, cl_sd)))
        v_i = v_pred * float(np.exp(rng.normal(0, v_sd)))
        p_i = _with_ka(pksim.params_from_prediction(cl_i, v_i, mdl), regimen)
        ti, ci = pksim.simulate(p_i, regimen["dose_mg"], regimen["interval_days"],
                                n_doses, regimen["infusion_days"], regimen["route"], horizon)
        curves.append(np.interp(ref_t, ti, ci))
    arr = np.vstack(curves)
    lo = np.percentile(arr, C.IIV_BAND_LO, axis=0)
    hi = np.percentile(arr, C.IIV_BAND_HI, axis=0)
    return lo, hi


def model_band(model_row, num, regimen, ref_t, n_doses, horizon):
    """Population band around the literature model (orange dashed, 'Model parameters').

    If the literature model has weight covariates (CL_WT etc.), a virtual weight population is run
    and CL/V/Q are scaled per individual; this uses the paper model's own covariate structure, so it
    takes priority. Drugs with no covariates would get no band at all; instead the same global/observed
    IIV% (log-normal) as for the prediction line is used.
    Returns (lo, hi, method); method is 'weight covariate' | 'global IIV' | None.
    """
    p0 = pksim.params_from_model(model_row)
    if p0 is None:
        return None, None, None
    seed = int(float(num)) if str(num).replace(".", "").isdigit() else 0
    rng = np.random.default_rng(seed)
    curves = []
    if pksim.has_weight_covariate(model_row):
        # Body weight alone falls well short of the reported between-subject variability (BSV/omega)
        # (e.g. Nivolumab CL: weight only ~10% CV vs paper BSV 42% CV). The remaining variability
        # (eta) not explained by covariates is multiplied in using the drug's observed IIV%
        # (vpop_iiv.csv, global fallback if absent).
        method = "weight covariate"
        wt_logsd = float(np.sqrt(np.log(1 + C.POP_WT_CV ** 2)))
        cl_sd, v_sd = iiv_sd_for(num)
        for _ in range(C.IIV_N):
            wt_i = C.POP_WT_REF * float(np.exp(rng.normal(0, wt_logsd)))
            p_i = pksim.covariate_scaled_params(model_row, wt_i, C.POP_WT_REF)
            p_i["CL"] = p_i["CL"] * float(np.exp(rng.normal(0, cl_sd)))
            if "V1" in p_i:
                p_i["V1"] = p_i["V1"] * float(np.exp(rng.normal(0, v_sd)))
            p_i = _with_ka(p_i, regimen)
            ti, ci = pksim.simulate(p_i, regimen["dose_mg"], regimen["interval_days"],
                                    n_doses, regimen["infusion_days"], regimen["route"], horizon)
            curves.append(np.interp(ref_t, ti, ci))
    else:
        method = "global IIV"
        cl_sd, v_sd = iiv_sd_for(num)
        for _ in range(C.IIV_N):
            p_i = dict(p0)
            p_i["CL"] = p0["CL"] * float(np.exp(rng.normal(0, cl_sd)))
            p_i["V1"] = p0["V1"] * float(np.exp(rng.normal(0, v_sd)))
            p_i = _with_ka(p_i, regimen)
            ti, ci = pksim.simulate(p_i, regimen["dose_mg"], regimen["interval_days"],
                                    n_doses, regimen["infusion_days"], regimen["route"], horizon)
            curves.append(np.interp(ref_t, ti, ci))
    arr = np.vstack(curves)
    lo = np.percentile(arr, C.IIV_BAND_LO, axis=0)
    hi = np.percentile(arr, C.IIV_BAND_HI, axis=0)
    return lo, hi, method


# Intravitreal (eye) injection drugs: systemic blood profiles are not clinically meaningful.
# Identified by INN stem (short, reliable list). Indication keywords are limited to those clearly
# meaning intravitreal injection; "eye" alone would catch systemic drugs such as Tebentafusp (uveal
# melanoma, IV) and Teprotumumab (thyroid eye disease, IV).
_INTRAVITREAL_STEMS = ("ranibizumab", "brolucizumab", "faricimab", "aflibercept",
                       "pegaptanib", "conbercept", "abicipar", "bevasiranib",
                       "dexamethasone intravitreal")
_INTRAVITREAL_IND = ("macular degeneration", "macular edema", "retinal vein "
                     "occlusion", "diabetic retinopathy", "neovascular age",
                     "intravitreal")


def is_intravitreal(row, regimen):
    """Whether this drug is given by intravitreal (eye) injection."""
    inn = str(row.get("INN", "")).strip().lower()
    if any(inn.startswith(s) for s in _INTRAVITREAL_STEMS):
        return True
    ind = str((regimen or {}).get("indication", "")).lower()
    return any(w in ind for w in _INTRAVITREAL_IND)


def predicted_thalf(cl_pred, v_pred):
    """1-compartment terminal half-life (day) = ln2*V/CL."""
    if not cl_pred or not v_pred or cl_pred <= 0 or v_pred <= 0:
        return None
    return float(np.log(2) * v_pred / cl_pred)


def sim_window(regimen):
    """Simulation window (n_doses, horizon_days).

    Covers at least 16 weeks and extends to approximate steady state, so daily/weekly dosing is not
    cut off before reaching steady state (long-half-life antibodies would otherwise look as if the
    curve never declines).
    """
    if regimen.get("single_dose"):
        return 1, float(C.SIM_DAYS_DEFAULT)
    tau = max(float(regimen["interval_days"]), 0.5)
    target = max(float(C.SIM_DAYS_DEFAULT), 112.0)   # at least 16 weeks
    n = min(60, max(2, int(np.ceil(target / tau)) + 1))
    return n, tau * n


def profile_figure(cl_pred, v_pred, num, regimen, observed=None,
                   pred_label="Predicted (allometry+ML)", show_band=True, matched=None):
    """Predicted profile + Model-parameter profile (+ population band) overlay.

    The band attaches to the literature model (orange dashed) if present: it shows that model's own
    population variability using the paper model's weight covariates (if any) or global IIV (if not).
    Only drugs without a literature model get it around the prediction line (blue solid).
    """
    fig = go.Figure()
    n_doses, horizon = sim_window(regimen)

    # 1) Predicted profile (the line is always drawn)
    p_pred = _with_ka(pksim.params_from_prediction(
        cl_pred, v_pred, MODEL_IDX.get(num)), regimen)
    t, c = pksim.simulate(p_pred, regimen["dose_mg"], regimen["interval_days"],
                          n_doses, regimen["infusion_days"], regimen["route"], horizon)

    # 2) Model-parameter profile: if present, the band is attached here first
    band_info = None
    if num in MODEL_IDX:
        mdl_row = MODEL_IDX[num]
        # For SC, fill ka: IV popPK rows have no ka, so no drug leaves the subcutaneous depot and
        # the orange line was entirely 0. The band is already filled.
        p_mdl = _with_ka(pksim.params_from_model(mdl_row), regimen)
        if p_mdl:
            tm, cm = pksim.simulate(p_mdl, regimen["dose_mg"], regimen["interval_days"],
                                    n_doses, regimen["infusion_days"], regimen["route"], horizon)
            if show_band:
                lo, hi, method = model_band(mdl_row, num, regimen, tm, n_doses, horizon)
                if lo is not None:
                    band_info = {"on": "model", "method": method}
                    # Amber lightened with white: keeps a translucent feel while keeping the edge
                    # clear against the navy background.
                    fig.add_trace(go.Scatter(
                        x=np.concatenate([tm, tm[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                        fill="toself", fillcolor=PAL["BAND_MODEL"],
                        line=dict(width=0), hoverinfo="skip",
                        name=f"Population variability ({C.IIV_BAND_LO}–{C.IIV_BAND_HI}%)"))
            fig.add_trace(go.Scatter(x=tm, y=cm, name="Model parameters",
                                     line=dict(color="#FB923C", width=2, dash="dash")))

    # 1b) Prediction-line band: only when no literature model existed above to attach it (fallback)
    if show_band and band_info is None:
        lo, hi = iiv_band(cl_pred, v_pred, num, regimen, t, n_doses, horizon)
        if lo is not None:
            band_info = {"on": "predicted", "method": "global IIV"}
            fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]),
                                     y=np.concatenate([hi, lo[::-1]]),
                                     fill="toself", fillcolor=PAL["BAND_PRED"],
                                     line=dict(width=0), hoverinfo="skip",
                                     name=f"Population variability ({C.IIV_BAND_LO}–{C.IIV_BAND_HI}%)"))
    fig.add_trace(go.Scatter(x=t, y=c, name=pred_label,
                             line=dict(color=ACCENT, width=3)))

    # Fast-clearing drugs (non-IgG formats etc.) show only as a spike on an 84-day axis. Narrow the
    # x axis to the meaningful range (also used for the regimen-matched steady-state point position,
    # so compute it first).
    x_hi = _view_window(t, c, regimen, horizon,
                        thalf=predicted_thalf(cl_pred, v_pred))

    # 3) Observed exposure points
    # SC absorption is slow, so Tmax is several days. Placing the point at 0.04 d lands where the
    # curve is near 0 and looked off by tens of fold.
    # Place it at the time of the peak within the first dosing interval of the drawn prediction
    # curve. IV is unchanged.
    x_cmax = regimen["infusion_days"] or 0.04
    if str(regimen.get("route", "")).upper().startswith("S"):
        k1 = t <= min(float(regimen["interval_days"]), horizon)
        if k1.any() and np.isfinite(c[k1]).any():
            x_cmax = float(t[k1][int(np.nanargmax(c[k1]))])
    if matched:
        # Regimen-matched observed points (Top-5). The first-dose value goes at the same position as
        # above; the steady-state value uses the last dosing interval visible on screen: trough at
        # its end, Cmax at the peak within it. Placing it at the end of the first interval (tau)
        # would compare it with the 1-cycle curve and look off by 2x.
        tau = float(regimen["interval_days"])
        x_end = min(x_hi if x_hi is not None else horizon, horizon)
        x_ss = max(1, int(np.floor(x_end / tau + 1e-9))) * tau
        kss = (t >= x_ss - tau - 1e-9) & (t <= x_ss + 1e-9)
        x_ss_max = (float(t[kss][int(np.nanargmax(c[kss]))]) if kss.any() else x_ss - tau)
        for p in matched["points"]:
            ss = p["when"] == "ss"
            if ss:      # steady-state observed points are not drawn; first-dose value only
                continue
            if p["stat"] == "Cmax":
                x, col = (x_ss_max if ss else x_cmax), "green"
            else:
                x, col = (x_ss if ss else tau), "red"
            fig.add_trace(go.Scatter(x=[x], y=[p["value"]], mode="markers",
                                     name=f"Observed {p['stat']}" + (" (steady state)" if ss else ""),
                                     marker=dict(color=col, size=11,
                                                 symbol="diamond-open" if ss else "diamond")))
    elif observed:
        if observed.get("Cmax"):
            fig.add_trace(go.Scatter(x=[x_cmax],
                                     y=[observed["Cmax"]], mode="markers",
                                     name="Observed Cmax", marker=dict(color="green", size=11, symbol="diamond")))
        if observed.get("Ctrough"):
            fig.add_trace(go.Scatter(x=[regimen["interval_days"]],
                                     y=[observed["Ctrough"]], mode="markers",
                                     name="Observed Ctrough", marker=dict(color="red", size=11, symbol="diamond")))

    # Move the horizontal legend well below the graph so it does not overlap the x-axis title.
    fig.update_layout(xaxis_title="Time (day)", yaxis_title="Concentration (µg/mL)",
                      height=500,
                      legend=dict(orientation="h", yanchor="top", y=-0.22,
                                  x=0.5, xanchor="center"),
                      # observed points (markers) on their own line below the line legend
                      legend2=dict(orientation="h", yanchor="top", y=-0.295,
                                   x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)",
                                   font=dict(size=13.5)),
                      margin=dict(t=30, b=135))
    for tr in fig.data:
        if tr.name and tr.mode == "markers":
            tr.legend = "legend2"
    space_legend(fig)
    if x_hi is not None:
        fig.update_xaxes(range=[0, x_hi])
    return style_axes(fig), band_info


def _view_window(t, c, regimen, horizon, thalf=None):
    """x-axis upper bound that contains the whole predicted curve with the empty tail trimmed. None if no narrowing is needed.

    For rapidly cleared drugs (t1/2 ~ 0.1-2 days):
      - single dose: soon 0 after the last dose -> the baseline takes 99% of the plot
      - repeated: steady state within a day -> the same sawtooth repeats for 16 weeks
    Showing only the early part is enough for both.
    """
    t = np.asarray(t); c = np.asarray(c)
    if t.size == 0 or not np.isfinite(c).any():
        return None
    cmax = float(np.nanmax(c))
    if cmax <= 0:
        return None
    single = regimen.get("single_dose")
    tau = 0.0 if single else float(regimen["interval_days"])
    n_doses = 1 if single else max(1, int(round(horizon / max(tau, 1e-6))))
    t_last = (n_doses - 1) * tau

    if single:
        tail = np.where(c > cmax * 5e-4)[0]
        need = float(t[tail[-1]]) * 1.1 if tail.size else horizon
    elif thalf is not None and thalf < 0.6 * max(tau, 1.0):
        # Repeated dosing + fast clearance: time to steady state (~5 t1/2) plus two more intervals
        doses_to_ss = int(np.ceil(5 * thalf / tau)) + 2
        need = min(doses_to_ss, n_doses) * tau
    else:
        tail = np.where((t >= t_last) & (c > cmax * 5e-4))[0]
        t_decay = float(t[tail[-1]]) if tail.size else t_last
        need = max(t_last + max(tau, 1.0), t_decay * 1.1)

    need = min(need, horizon)
    return need if need < 0.65 * horizon else None


# ================================================================ UI
brand_header()
status_bar()
# Light-mode toggle: right end of the header logo row (visible on both tabs). Value is
# session_state["light_mode"].
# Taken out of flow (absolute) so it does not push the header/tabs. The chip group leaves a 212px
# slot for the toggle on the right, so they never overlap at any screen width.
st.html("<style>"
        "[data-testid='stVerticalBlock']:has(> .st-key-light_mode) { position: relative; }"
        ".st-key-light_mode { position: absolute !important; top: 52px; right: 0; z-index: 50;"
        " width: auto !important; transform: scale(1.75); transform-origin: right center; }"
        ".st-key-light_mode label p { font-size: 11.5px !important; font-weight: 600;"
        " color: var(--text) !important; white-space: nowrap; }"
        ".st-key-light_mode label > span + div { background: var(--border) !important; }"
        ".st-key-light_mode label:has(input:checked) > span + div { background: var(--accent) !important; }"
        # Phone width: to avoid overlapping the header, move the toggle onto its own row below the
        # header; title and subtitle may wrap
        "@media (max-width: 640px) {"
        " .st-key-light_mode { position: static !important; transform: none; margin: -6px 0 4px auto; }"
        " .kx-chips { margin-right: 0; }"
        " .kx-brand { align-items: flex-start; gap: 12px; }"
        " .kx-titles { min-width: 0; }"
        " .kx-sub { white-space: normal; letter-spacing: 0.06em; }"
        " .kx-title { flex-wrap: wrap; }"
        # Keep the status bar on one line; the app name and DB counts are already in the header, so
        # hide them
        " .kx-status { gap: 10px; padding: 4px 12px; font-size: 10.5px; white-space: nowrap; }"
        " .kx-status > span:nth-child(2), .kx-status > span:nth-child(3) { display: none; }"
        "}"
        "</style>")
st.toggle("Light mode", value=False, key="light_mode")

tab1, tab2 = st.tabs(["Existing Drugs", "New Drug"])

# ---------------------------------------------------------------- TAB 1
with tab1:
    catalog = drug_catalog()
    if catalog.empty:
        st.warning("No predictable drug available.")
    else:
        counts = catalog["_group"].value_counts()
        opts = ["All drugs"] + [g for g in GROUP_ORDER if counts.get(g, 0)]
        pick = st.selectbox(
            "Show",
            opts,
            format_func=lambda g: (f"All drugs ({len(catalog)})"
                                   if g == "All drugs"
                                   else f"{g} ({counts.get(g, 0)})"),
            help="Grouped by data coverage: full = both CL and V on file. "
                 "'Training set' = used to fit at least one model.")
        drugs = (catalog if pick == "All drugs"
                 else catalog[catalog["_group"] == pick])
        # Two dropdown tags.
        #   (approved)  regimen taken from the reviewed Regimen set (the rest are clinical trial records and may not be approved regimens).
        #   (model)     the literature model table has compartment parameters for the drug, so the literature model profile (orange dashed) can be overlaid on the curve.
        _reg = appdata.load()["regimen"]
        _mdl_nums = model_profile_nums()
        _tag_map = {}
        for _, r in drugs.iterrows():
            _tags = []
            if (_reg.get(str(r["NUM"])) or {}).get("source") == "regimen":
                _tags.append("approved")
            if str(r["NUM"]) in _mdl_nums:
                _tags.append("model")
            if _tags:
                _tag_map[r["_label"]] = "  (" + " · ".join(_tags) + ")"
        # List names (_label) are unique; the row is found via name -> NUM.
        if not DEMO_TOP5:
            sel = st.selectbox("Select drug (drop-down)", drugs["_label"].tolist())
            approved_badge(_tag_map)
        else:
            # Top-5 lock: the whole list is visible but only Top-5 can be chosen. Re-checked on the
            # server: a non-Top-5 value reverts to the previous Top-5 (Pembrolizumab if none).
            _top = [l for l, i in zip(catalog["_label"], catalog["INN"]) if i in TOP5_INN]
            _first = next((l for l, i in zip(catalog["_label"], catalog["INN"])
                           if i == "Pembrolizumab"), _top[0] if _top else None)
            _last = st.session_state.get("_top5_last", _first)
            _opts = drugs["_label"].tolist()
            if not any(l in _top for l in _opts):     # if the filter has no Top-5, put the previous Top-5 first
                _opts = [_last] + [l for l in _opts if l != _last]
            if st.session_state.get("drug_sel") not in _top or st.session_state["drug_sel"] not in _opts:
                st.session_state["drug_sel"] = _last if _last in _opts else next(
                    l for l in _opts if l in _top)
            sel = st.selectbox("Select drug (drop-down)", _opts, key="drug_sel")
            st.session_state["_top5_last"] = sel
            approved_badge(_tag_map, extra_html=top5_lock_html(_top))
        _src = catalog if DEMO_TOP5 else drugs    # demo: also find the previous Top-5 outside the filter
        _num_of = dict(zip(_src["_label"], _src["NUM"]))
        row = _src[_src["NUM"] == _num_of[sel]].iloc[0]
        num = row["NUM"]
        sel_inn = row["INN"]           # captions and exports use the INN as is
        if not row["_trained"]:
            st.caption("Not in the training set — prediction uses models fitted on all training drugs.")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("1. Drug information")
            info = {k: row.get(k) for k in ["INN", "Target", "Isotype", "MW",
                    "MOLECULAR CATEGORY", "DAR", "LINKER", "PAYLOAD",
                    "THERMAL STABILITY", "HYDROPHOBICITY"] if k in row.index}
            _lab = {"MW": "Molecular Weight", "MOLECULAR CATEGORY": "Molecular Category", "LINKER": "Linker",
                    "PAYLOAD": "Payload", "THERMAL STABILITY": "Thermal Stability",
                    "HYDROPHOBICITY": "Hydrophobicity"}
            info_rows = [(_lab.get(k, k), "—" if pd.isna(v) else str(v)) for k, v in info.items()]
            st.table(pd.DataFrame(info_rows, columns=["Field", "Value"]))
        with c2:
            st.subheader("2. Monkey CL and V")
            st.table(pd.DataFrame({
                "Field": ["Body weight (kg)", "CL (L/day, normalized)", "V (L, normalized)"],
                "Value": [_sig(row.get("monkey_bw")),
                          _sig(row.get("monkey_CL_Lday")),
                          _sig(row.get("monkey_V_L"))]}))
            _mk_cl = pd.to_numeric(row.get("monkey_CL_Lday"), errors="coerce")
            _mk_v = pd.to_numeric(row.get("monkey_V_L"), errors="coerce")
            _gone = [t for t, v in (("CL", _mk_cl), ("V", _mk_v))
                     if not (pd.notna(v) and v > 0)]
            if _gone:
                st.caption(f"No monkey {' or '.join(_gone)} on file — pure-ML "
                          f"models carry the prediction for {' or '.join(_gone)}.")

        pr = run_prediction(row, num)
        algo_cl, algo_v = RES["CL"]["algo"], RES["V"]["algo"]
        pc_map = {d["name"]: d for d in target_predictions("CL", pr["feat_CL"], pr["allo_CL"], num)}
        pv_map = {d["name"]: d for d in target_predictions("V", pr["feat_V"], pr["allo_V"], num)}
        # Methods relying on allometry (Allometry, hybrid) are selectable only when the target has a
        # base value; without one the value itself does not exist.
        def _ok_allo(x):
            return x is not None and np.isfinite(x) and x > 0

        _, dropped_cl = live_hybrids(pc_map)
        _, dropped_v = live_hybrids(pv_map)
        if not _ok_allo(pr["allo_CL"]):
            pc_map = {k: v for k, v in pc_map.items()
                      if v.get("kind") not in ("residual", "baseline")}
            dropped_cl = []
        if not _ok_allo(pr["allo_V"]):
            pv_map = {k: v for k, v in pv_map.items()
                      if v.get("kind") not in ("residual", "baseline")}
            dropped_v = []
        # Out-of-range predictions are removed from the choices (PLAUSIBLE). The ★ and the default
        # are chosen among the remaining candidates.
        pc_map = plausible_only(pc_map, "CL")
        pv_map = plausible_only(pv_map, "V")
        is_oof = num in RES.get("oof_table", {})

        # Observed value: bundle (training target) first, otherwise screen data.
        obs_cl = observed_of(num, "CL")
        obs_v = observed_of(num, "V")
        if obs_cl is None:
            obs_cl = row.get("human_CL_Lday")
        if obs_v is None:
            obs_v = row.get("human_V_L")

        # ★ = candidate with the highest cross-validated R2 among those selectable for this drug.
        # The candidate list differs per drug, so the star moves per drug.
        star_cl = best_by_r2(pc_map, "CL")
        star_v = best_by_r2(pv_map, "V")
        default_cl = star_cl or algo_cl
        default_v = star_v or algo_v

        # Append the drug number to the selector key.
        #
        # With a fixed key ("tab1_cl_model") cleared from session_state on drug change, Streamlit
        # keeps the widget value for the same key, so the reset would apply one rerun late and the
        # previous drug's method would stay until clicked.
        #
        # A per-drug key makes the widget new for each drug, so it starts from index (= ★). A
        # manually chosen method persists while on that drug and when returning to it.
        _kc, _kv = f"tab1_cl_model_{num}", f"tab1_v_model_{num}"

        st.subheader("3. Predicted Human CL and V",
                     help="The best-scoring method (by cross-validation R²) is "
                          "pre-selected below; override from the dropdowns. "
                          "● dot below each value = fold-error vs. observed "
                          f"(CL green <{C.FLAG_CUT['CL']['green']:g}×, red "
                          f">{C.FLAG_CUT['CL']['red']:g}× · V green "
                          f"<{C.FLAG_CUT['V']['green']:g}×, red "
                          f">{C.FLAG_CUT['V']['red']:g}×).")

        _no_allo = not any(x is not None and np.isfinite(x) and x > 0
                           for x in (pr["allo_CL"], pr["allo_V"]))
        if _no_allo:
            st.caption("No monkey CL/V on file — PureML predicts directly.")
        elif dropped_cl or dropped_v:
            _why = []
            for _tg, _dd, _ft in (("CL", dropped_cl, pr["feat_CL"]),
                                  ("V", dropped_v, pr["feat_V"])):
                if not _dd:
                    continue
                _miss = hybrid_block_reason(_ft, _tg)
                _why.append(f"**{_tg}** (missing "
                            + ", ".join(_miss or ["a core input"]) + ")")
            st.caption("Hybrid unavailable for " + " and ".join(_why)
                      + " — falls back to the plain tree model.")

        cc = st.columns(2)
        cl_names = table_order(_sorted_names(pc_map.values()))
        v_names = table_order(_sorted_names(pv_map.values()))
        cl_choice = cc[0].selectbox(
            "CL method", cl_names, format_func=lambda n: MODEL_LABEL.get(n, n),
            index=cl_names.index(default_cl) if default_cl in cl_names else 0,
            key=_kc)
        if star_cl:
            cc[0].markdown(f"<span class='best-pill'>Best Method for this Drug — "
                           f"{MODEL_LABEL.get(star_cl, star_cl)}</span>",
                           unsafe_allow_html=True)
        v_choice = cc[1].selectbox(
            "V method", v_names, format_func=lambda n: MODEL_LABEL.get(n, n),
            index=v_names.index(default_v) if default_v in v_names else 0,
            key=_kv)
        if star_v:
            cc[1].markdown(f"<span class='best-pill'>Best Method for this Drug — "
                           f"{MODEL_LABEL.get(star_v, star_v)}</span>",
                           unsafe_allow_html=True)

        # -- OOD fallback: out-of-distribution drugs use allometry instead of ML.
        #    If the user changed the method manually, that choice is respected (warning only); fallback happens only for the auto-selected method.
        flagged, ood_dist, ood_thr = ood_check(row, INV_IDX.get(num), num)
        cl_used, cl_entry, cl_fb = apply_ood_fallback("CL", cl_choice, pc_map, flagged,
                                                      is_auto=(cl_choice == default_cl))
        v_used, v_entry, v_fb = apply_ood_fallback("V", v_choice, pv_map, flagged,
                                                   is_auto=(v_choice == default_v))
        cl_pred, v_pred = cl_entry["pred"], v_entry["pred"]
        # baseline shrink: pull the ML prediction slightly toward allometry (no-op for OOD fallback
        # / allometry)
        cl_pred, cl_sh = shrink_to_allo(cl_pred, pr["allo_CL"], cl_entry["kind"])
        v_pred, v_sh = shrink_to_allo(v_pred, pr["allo_V"], v_entry["kind"])
        cl_lbl, v_lbl = MODEL_LABEL.get(cl_used, cl_used), MODEL_LABEL.get(v_used, v_used)
        if cl_sh or v_sh:
            sh_of = " and ".join([x for x, on in [("CL", cl_sh), ("V", v_sh)] if on])
            st.caption(f"{sh_of} shrunk toward allometry "
                      f"({int(C.SHRINK_W*100)}% ML + {int((1-C.SHRINK_W)*100)}% allometry).")
        if cl_fb or v_fb:
            fb_of = " and ".join([x for x, on in
                                  [("CL", cl_fb), ("V", v_fb)] if on])
            st.info(f"Out-of-distribution — allometry fallback applied to {fb_of} "
                    f"(3-NN distance {ood_dist:.2f} > threshold {ood_thr:.2f}). "
                    f"Pick a method above to override.")
        elif flagged and (cl_choice != "allometry_only" or v_choice != "allometry_only"):
            st.warning(f"Out-of-distribution (3-NN distance {ood_dist:.2f} > threshold "
                       f"{ood_thr:.2f}) — ML shown as manually selected, but allometry is safer here.")

        def _acc_badge(pred, obs, tg="CL"):
            """Show fold-error vs observed as a traffic-light color. Thresholds are per target.

            Unlike % vs allometry (change in value), the color reflects real prediction accuracy.
            Empty string if there is no observation (new drug etc.).
            """
            if pred is None or obs is None or pd.isna(obs) or obs <= 0 or pred <= 0 \
                    or not np.isfinite(pred):
                return ""
            _cut = getattr(C, "FLAG_CUT", {}).get(
                tg, {"green": C.FLAG_GREEN, "red": C.FLAG_RED})
            fold = max(pred / obs, obs / pred)
            if fold < _cut["green"]:
                col, txt = GOOD, "close"
            elif fold > _cut["red"]:
                col, txt = BAD, "far"
            else:
                col, txt = WARN, "moderate"
            return (f"<div style='color:{TEXT}'>"
                    f"<span style='color:{col};font-weight:700'>●</span> "
                    f"<b>{fold:.2f}×</b> vs observed "
                    f"<span style='color:{col};font-weight:600'>({txt})</span></div>")

        # % vs allometry (change) is shown in neutral gray (delta_color='off'); the accuracy badge
        # below (fold vs observed) carries the traffic-light color.
        def _iiv_range(pred, sd):
            """Individual variability P5-P95 range text around the predicted typical value."""
            if pred is None or not np.isfinite(pred) or pred <= 0:
                return ""
            z = 1.645  # P5/P95 (standard normal)
            lo, hi = pred * np.exp(-z * sd), pred * np.exp(z * sd)
            return f"<div>population 5–95%: {lo:.3g} – {hi:.3g}</div>"

        def _delta_badge(adj):
            # st.metric's own delta= is not used: Allometry/Observed cards have no delta, which made
            # card heights differ. All three cards hold only the value, and "vs allometry" goes on
            # the helper text line below.
            if adj is None:
                return ""
            pct = (np.exp(adj) - 1) * 100
            return f"<div>{pct:+.0f}% vs allometry</div>"

        def _below(col, *lines):
            # Helper lines under the card as one block; drawing each line separately makes the
            # spacing uneven
            body = "".join(x for x in lines if x)
            if body:
                col.markdown(f"<div class='kx-below'>{body}</div>", unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Allometry CL (L/day)", _sig(pr["allo_CL"]))
        _below(m1, _acc_badge(pr["allo_CL"], obs_cl, "CL"))
        m2.metric(f"Predicted CL — {cl_lbl}", _sig(cl_pred))
        _below(m2, _delta_badge(cl_entry["adj"]), _acc_badge(cl_pred, obs_cl, "CL"),
               _iiv_range(cl_pred, iiv_sd_for(num)[0]), source_note(num, "CL"))
        m3.metric("Observed Human CL", f"{obs_cl:.3g}" if pd.notna(obs_cl) else "—")
        m1, m2, m3 = st.columns(3)
        m1.metric("Allometry V (L)", _sig(pr["allo_V"]))
        _below(m1, _acc_badge(pr["allo_V"], obs_v, "V"))
        m2.metric(f"Predicted V — {v_lbl}", _sig(v_pred))
        _below(m2, _delta_badge(v_entry["adj"]), _acc_badge(v_pred, obs_v, "V"),
               _iiv_range(v_pred, iiv_sd_for(num)[1]), source_note(num, "V"))
        m3.metric("Observed Human V", f"{obs_v:.3g}" if pd.notna(obs_v) else "—")

        _src = {tg: target_source(num, tg) for tg in ("CL", "V")}
        if all(v == "oof" for v in _src.values()):
            st.caption("Both out-of-fold — fair check, not memorization.")
        elif any(v == "oof" for v in _src.values()):
            _o = "CL" if _src["CL"] == "oof" else "V"
            _f = "V" if _o == "CL" else "CL"
            st.caption(f"{_o} is out-of-fold; {_f} was never in its own training set — both out-of-sample.")
        else:
            st.caption("Not in either training set — both predictions are out-of-sample.")

        def _how(chosen, algo, fb, star=None):
            """How this method was chosen."""
            if fb:
                return "OOD fallback"
            if star and chosen == star:
                return "best cross-validation R² among this drug's candidates"
            if chosen == algo:
                return "cross-validation default"
            return "manually chosen"

        def _cvm(target, name, key):
            """Cross-validation value of the model currently in use, not the confirmed model's.

            Since the method now varies per drug, printing the confirmed model's numbers would show another
            model's score as if it were this model's.
            """
            return RES[target]["metrics"].get(name, {}).get(key)

        st.caption(f"In use — CL: **{cl_lbl}** ({KIND_LABEL.get(cl_entry['kind'], cl_entry['kind'])}, "
                   f"{_how(cl_choice, algo_cl, cl_fb, star_cl)}, "
                   f"typical error={_gmfe(_cvm('CL', cl_used, 'gmfe'))}, "
                   f"RMSE={_fmt(_cvm('CL', cl_used, 'rmse'))}) · "
                   f"V: **{v_lbl}** ({KIND_LABEL.get(v_entry['kind'], v_entry['kind'])}, "
                   f"{_how(v_choice, algo_v, v_fb, star_v)}, "
                   f"typical error={_gmfe(_cvm('V', v_used, 'gmfe'))}, "
                   f"RMSE={_fmt(_cvm('V', v_used, 'rmse'))})")

        # ── Per-model comparison table (allometry / residual-ML / pure-ML, all three families)
        with st.expander("Model comparison — Allometry vs. ML (residual correction) vs. ML (Pure/direct)"):
            shown = st.multiselect(
                "Cross-validation metrics to show",
                list(CV_METRICS), default=CV_DEFAULT, key="cvm1",
                help="GMFE = typical fold-gap · RMSE = log-space spread · "
                     "R² = variance explained · Spearman = rank accuracy")
            _cmp_df = model_comparison_df(
                pr["feat_CL"], pr["feat_V"], pr["allo_CL"], pr["allo_V"],
                obs_cl if pd.notna(obs_cl) else None,
                obs_v if pd.notna(obs_v) else None, num=num, show=shown,
                star_cl=star_cl, star_v=star_v)
            _best_cl_lbl = MODEL_LABEL.get(star_cl, star_cl) if star_cl else None
            _best_v_lbl = MODEL_LABEL.get(star_v, star_v) if star_v else None

            def _highlight_best(row):
                # Instead of a star, highlight the best-performing row for that target with a faint
                # marker-style background.
                styles = [""] * len(row)
                if _best_cl_lbl and row["Model"] == _best_cl_lbl:
                    for i, col in enumerate(row.index):
                        if col == "Model" or col.startswith("CL"):
                            styles[i] = "background-color: " + PAL["HL_CL"]
                if _best_v_lbl and row["Model"] == _best_v_lbl:
                    for i, col in enumerate(row.index):
                        if col == "Model" or col.startswith("V"):
                            styles[i] = "background-color: " + PAL["HL_V"]
                return styles

            st.dataframe(_cmp_df.style.apply(_highlight_best, axis=1).hide(axis="index"),
                        width="stretch")
            if star_cl or star_v:
                st.caption("Highlighted cells = best cross-validation R² for this drug "
                           "(blue = CL, green = V).")
            st.caption("Allometry's R² is scored on a smaller drug set and isn't directly "
                       "comparable — use **R² (same drugs)** for a fair comparison.")
            st.caption("Typical error (GMFE): 1.00× = perfect, 2× = typically off two-fold. "
                       "RMSE penalizes big misses more. fold vs obs = this drug's own error"
                       f"{' (out-of-fold)' if is_oof else ''}. "
                       f"Composite score = {C.SEL_W_CLOSENESS:.2f}·typical error + {C.SEL_W_RMSE:.2f}·RMSE.")
            if "allometry_only" in (algo_cl, algo_v):
                st.info(f"Neither ML approach beat allometry under cross-validation here "
                        f"(small sample, N={RES['n_train']}).")

        st.subheader("4. Traffic-light flag (predicted vs observed Human)")
        fl = predict.flag(cl_pred, obs_cl if pd.notna(obs_cl) else None,
                          v_pred, obs_v if pd.notna(obs_v) else None)
        # Take the model name actually used in step 3 as is. Whether it is the best was already
        # shown by the best-pill in step 3, so it is not marked again here.
        flag_badge(fl, {"CL": cl_lbl, "V": v_lbl})
        st.caption("Out-of-fold check — reflects real predictive accuracy, not memorization." if is_oof
                   else "No observed Human data in the training set — prediction is out-of-sample.")

        st.subheader("5. PK profile simulation (approved regimen)")
        reg = appdata.regimen_for(num)
        if reg is None:
            st.warning("No approved regimen data for this drug; simulation not available.")
        elif is_intravitreal(row, reg):
            st.warning("Intravitreal drug — injected into the eye, not the bloodstream. "
                      "A systemic profile isn't clinically meaningful, so it isn't drawn.")
        else:
            # Offer a route choice only for drugs with both IV and SC. The default is the current
            # route, so the screen is unchanged unless another is chosen; choosing another uses that
            # route's regimen and observed exposure.
            _routes = appdata.regimen_routes(num)
            _route_switched = False
            if set(_routes) == {"IV", "SC"}:
                _rt_opts = ["IV", "SC"]
                _rt = st.radio("Route", _rt_opts, index=_rt_opts.index(reg["route"]),
                               horizontal=True, key=f"route_{num}")
                if _rt != reg["route"]:
                    reg, _route_switched = _routes[_rt], True
            _sched = ("single dose" if reg.get("single_dose")
                      else dose_interval_text(reg["interval_days"]))
            _from_human = reg.get("source") == "human"
            _tag = ("  ·  from clinical-study records, not an approved label"
                    if (_from_human and not reg.get("approved")) else "")
            _dose = reg["dose_mg"]
            _dtxt = (f"{_dose:.0f}" if _dose >= 10 else
                     f"{_dose:.3g}")            # so doses below 1 mg do not show as "0 mg"
            # If an SC formulation is registered separately (e.g. Pembrolizumab -> 449), name that
            # product
            _link = ""
            if reg.get("linked_from"):
                _lb = appdata.base()
                _ln = _lb.loc[_lb["NUM"].astype(str) == str(reg["linked_from"]), "INN"]
                _link = f" · SC product: {_ln.iloc[0] if len(_ln) else reg['linked_from']}"
            st.container(key="kx_regimen").caption(f"{sel_inn} · Regimen: {_dtxt} mg · {_sched} · "
                       f"{reg['route']}" + _link
                       + (f" · Indication: {reg['indication']}"
                          if reg.get("indication") else "") + _tag,
                       help=("From human PK study records, not the approved-regimen file"
                             + ("" if reg.get("approved") else " (dose-finding study, not necessarily approved)")
                             + " — CL/V predictions are unaffected.") if _from_human else None)
            _notes = []
            if reg.get("dose_basis") == "mg/kg":
                _notes.append(f"per body weight, converted at {C.HUMAN_BW:.0f} kg")
            elif reg.get("dose_basis") == "mg/m^2":
                _notes.append(f"per body-surface area, converted at {getattr(C, 'HUMAN_BSA', 1.73):.2f} m²")
            if reg.get("single_dose"):
                _notes.append("no interval on file — shown as a single dose over 12 weeks")
            if _notes:
                st.caption("Note — " + "; ".join(_notes) + ".")
            if ka_is_default(num, reg):
                st.caption(
                    f"Absorption rate not reported — using default "
                    f"ka = {getattr(C, 'DEFAULT_KA_SC', 0.25):.2f}/day "
                    f"(t½ ≈ {np.log(2)/getattr(C, 'DEFAULT_KA_SC', 0.25):.1f} d).",
                    help="Cmax/Tmax are indicative; steady-state AUC is unaffected.")
            _th = predicted_thalf(cl_pred, v_pred)
            _cat = str(row.get("MOLECULAR CATEGORY", "")).lower()
            _mw = pd.to_numeric(row.get("MW"), errors="coerce")
            _nonigg = ("frag" in _cat or "fusion" in _cat or "bispecific" in _cat
                       or "bsab" in _cat or (pd.notna(_mw) and _mw < 110))
            if _th is not None and _th < 3:
                st.caption(
                    f"Short predicted half-life (~{_th:.1f} day"
                    + ("s" if _th >= 1.5 else "") + ")"
                    + (" — non-IgG format, fast clearance expected." if _nonigg else ".")
                    + " X-axis zoomed to the informative part of the curve.")
            obs = (appdata.exposure_for(num, reg["route"]) if _route_switched
                   else appdata.human_exposure(num))
            # If a regimen-matched observed value exists (Top-5, that route), use it; otherwise as
            # before.
            _matched = (appdata.load().get("exposure_matched", {})
                        .get(str(num), {}).get(reg["route"]))
            kpi_row(cl_pred, v_pred, reg, MODEL_IDX.get(num))
            fig, band_info = profile_figure(
                cl_pred, v_pred, num, reg, None if _matched else obs,
                pred_label=f"Predicted (CL:{cl_lbl} · V:{v_lbl})", matched=_matched)
            chart_block(fig, f"{sel}".replace(" ", "_"), title=f"{sel_inn} — {_dtxt} mg {_sched} {reg['route']}",
                        params=[{"drug": sel_inn, "CL_method": cl_lbl, "V_method": v_lbl,
                                 "CL_pred_Lday": cl_pred, "V_pred_L": v_pred,
                                 "CL_allometry_Lday": pr["allo_CL"], "V_allometry_L": pr["allo_V"],
                                 "CL_observed_Lday": obs_cl if pd.notna(obs_cl) else None,
                                 "V_observed_L": obs_v if pd.notna(obs_v) else None,
                                 "dose_mg": _dose, "interval_day": reg["interval_days"],
                                 "route": reg["route"], "model_build": RES.get("built")}])
            if band_info is None:
                pass
            elif band_info["on"] == "model" and band_info["method"] == "weight covariate":
                _cl_sd_m, _v_sd_m = iiv_sd_for(num)
                _cvp_m = lambda sd: (np.exp(sd ** 2) - 1) ** 0.5 * 100
                _src_m = ("this drug's own published virtual population" if iiv_is_measured(num)
                          else "the median across ~40 published virtual populations")
                st.caption(
                    f"Population variability ({C.IIV_BAND_LO}–{C.IIV_BAND_HI}%): "
                    f"CL CV {_cvp_m(_cl_sd_m):.0f}% · V CV {_cvp_m(_v_sd_m):.0f}%",
                    help=(f"Body weight CV {C.POP_WT_CV*100:.0f}% around {C.POP_WT_REF:.0f} kg, "
                          f"scaled by the model's own weight exponent, plus CL/V CV from "
                          f"{_src_m}. Not prediction accuracy."))
            else:
                _cl_sd_u, _v_sd_u = iiv_sd_for(num)
                _cvp = lambda sd: (np.exp(sd ** 2) - 1) ** 0.5 * 100  # log-SD -> CV%
                _src = ("this drug's own published virtual population" if iiv_is_measured(num)
                        else "the median across ~40 published virtual populations")
                st.caption(
                    f"Population variability ({C.IIV_BAND_LO}–{C.IIV_BAND_HI}%): "
                    f"CL CV {_cvp(_cl_sd_u):.0f}% · V CV {_cvp(_v_sd_u):.0f}%",
                    help=f"From {_src}. Not prediction accuracy.")

# ---------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("New Drug Input → Human PK Prediction",
                 help="Drug, in vitro and monkey CL/V inputs → predicted profile.")
    with st.form("newdrug"):
        cc = st.columns(3)
        inn = cc[0].text_input("INN (name)", "New-mAb")
        isotype = cc[1].selectbox("Isotype", ["IgG1", "IgG2", "IgG4", "missing"])
        molcat = cc[2].selectbox("Molecular Category",
                                 ["mAb [human]", "mAb [humanized]", "mAb [chimeric]",
                                  "ADC", "bispecific", "missing"])
        cc = st.columns(3)
        # Negative values are not accepted (min_value=0.0). Defaults are all >= 0 and plain floats.
        mw = cc[0].number_input("MW (kDa)", value=150.0, min_value=0.0)
        dar = cc[1].number_input("DAR (ADC only)", value=0.0, min_value=0.0)
        mk_bw = cc[2].number_input("Monkey body weight (kg)", value=3.0, min_value=0.0)
        cc = st.columns(3)
        mk_cl = cc[0].number_input("Monkey CL (L/day, absolute)", value=0.02, min_value=0.0,
                                   format="%.4f")
        mk_v = cc[1].number_input("Monkey V (L, absolute)", value=0.15, min_value=0.0,
                                  format="%.4f")
        mk_t12 = cc[2].number_input("Monkey t½ (day, 0 = unknown)", value=0.0, min_value=0.0,
                                    format="%.2f",
                                    help="Used by the CL model (the V model does not use it); "
                                         "0 = unknown, imputed.")
        cc = st.columns(3)
        kd = cc[0].number_input("Kd (nM)", value=1.0, min_value=0.0)
        ic50 = cc[1].number_input("IC50 (nM)", value=0.0, min_value=0.0)
        ec50 = cc[2].number_input("EC50 (nM)", value=0.0, min_value=0.0)
        cc = st.columns(3)
        dose_mg = cc[0].number_input("Dose (mg)", value=400.0, min_value=0.0)
        interval_w = cc[1].number_input("Interval (weeks)", value=3.0, min_value=0.0)
        route = cc[2].selectbox("Route", ["IV", "SC"])
        inf_h = st.number_input("Infusion time (hours, IV)", value=1.0, min_value=0.0)
        target_conc = st.number_input(
            "Target concentration (µg/mL, optional)", value=0.0, min_value=0.0,
            help="Drawn as a red dashed line on the chart. 0 = not set.")
        submitted = st.form_submit_button("Run prediction")

    # Inputs live inside the form, so values are fixed only on submit: store them in session_state,
    # and redraw from the stored inputs even when a change in the method selectbox (outside the
    # form) triggers a rerun.
    if submitted:
        st.session_state["nd"] = {
            "inn": inn, "isotype": isotype, "molcat": molcat, "mw": mw, "dar": dar,
            "mk_cl": mk_cl, "mk_v": mk_v, "mk_bw": mk_bw, "mk_t12": mk_t12,
            "kd": kd, "ic50": ic50, "ec50": ec50,
            "dose_mg": dose_mg, "interval_w": interval_w, "route": route, "inf_h": inf_h,
            "target_conc": target_conc,
        }

    if "nd" in st.session_state:
        nd = st.session_state["nd"]
        base_row = pd.Series({"INN": nd["inn"], "Isotype": nd["isotype"],
                              "MOLECULAR CATEGORY": nd["molcat"], "MW": nd["mw"], "DAR": nd["dar"],
                              "monkey_CL_Lday": nd["mk_cl"], "monkey_V_L": nd["mk_v"],
                              "monkey_bw": nd["mk_bw"],
                              # 0/empty -> NaN -> replaced by the training-set median (imputer)
                              "monkey_t_half": nd.get("mk_t12") or np.nan,
                              "SCIV": nd["route"]})
        inv_row = {"Kd": nd["kd"] or np.nan, "IC50": nd["ic50"] or np.nan,
                   "EC50": nd["ec50"] or np.nan}
        a_cl = predict.allometry_cl(nd["mk_cl"], nd["mk_bw"])
        a_v = predict.allometry_v(nd["mk_v"], nd["mk_bw"])
        # Model features are built by converting the class name to the DB format. With the form
        # labels ("ADC", "bispecific") as is, mc_conjugate and mc_bsAb were always 0, giving the
        # same value as "missing". mAb-family and missing give the same features either way.
        feat_row = base_row.copy()
        feat_row["MOLECULAR CATEGORY"] = newdrug.db_category(base_row["MOLECULAR CATEGORY"])
        feat_cl = predict.build_features(feat_row, inv_row, _loga(a_cl))
        feat_v = predict.build_features(feat_row, inv_row, _loga(a_v))

        # New drugs have no sequence: predict with the variant trained without sequence. Filling the
        # sequence columns of the original model with the median biased CL 1.32x too high.
        pc = {d["name"]: d for d in predict.model_predictions(RES, "CL", feat_cl, a_cl, noseq=True)}
        pv = {d["name"]: d for d in predict.model_predictions(RES, "V", feat_v, a_v, noseq=True)}
        # Out-of-range predictions are removed from the choices (PLAUSIBLE, same as tab 1)
        pc, pv = plausible_only(pc, "CL"), plausible_only(pv, "V")
        ND_CL, ND_V = predict.noseq_block(RES, "CL"), predict.noseq_block(RES, "V")

        # -- OOD check: if the new drug is outside the training distribution, ML cannot be trusted.
        # Similar-drug and OOD searches use the input in DB format (class names matched, DAR 0 = not
        # applicable). Otherwise a plain antibody would be matched close to ADCs because of DAR 0
        # and raise a false OOD warning.
        sim_row = newdrug.similarity_row(base_row)
        nd_flagged, nd_dist, nd_thr = ood_check(sim_row, inv_row)
        if nd_flagged:
            st.warning(f"Out of the training distribution ({OOD_LABEL} {nd_dist:.2f} > "
                       f"threshold {nd_thr:.2f}) — allometry is the safer choice here.")
        elif nd_dist is not None:
            st.success(f"Within the training distribution ({OOD_LABEL} {nd_dist:.2f} ≤ "
                       f"threshold {nd_thr:.2f}).")

        with st.expander("Model comparison — Allometry vs. ML (residual correction) vs. ML (Pure/direct)"):
            shown2 = st.multiselect(
                "Cross-validation metrics to show",
                list(CV_METRICS), default=CV_DEFAULT, key="cvm2")
            # no observed → 'fold vs obs' = —
            st.table(model_comparison_df(feat_cl, feat_v, a_cl, a_v,
                                         show=shown2, noseq=True))
            st.caption(f"Typical error (GMFE): 1.00× = perfect. RMSE penalizes big misses more. "
                       f"Composite = {C.SEL_W_CLOSENESS:.2f}·typical error + {C.SEL_W_RMSE:.2f}·RMSE.")

        st.markdown("**Choose method(s) to simulate** — select one or more per target to overlay")
        cl_names = _sorted_names(pc.values())
        v_names = _sorted_names(pv.values())
        cl_default = [ND_CL["algo"]] if ND_CL["algo"] in cl_names else cl_names[:1]
        v_default = [ND_V["algo"]] if ND_V["algo"] in v_names else v_names[:1]
        cc = st.columns(2)
        cl_sel = cc[0].multiselect(
            "CL method(s)", cl_names, default=cl_default,
            format_func=lambda n: MODEL_LABEL.get(n, n), key="nd_cl_models")
        v_sel = cc[1].multiselect(
            "V method(s)", v_names, default=v_default,
            format_func=lambda n: MODEL_LABEL.get(n, n), key="nd_v_models")

        MAX_LINES = 6
        combos = ([(cl_n, v_n) for cl_n in cl_sel for v_n in v_sel]
                  if (cl_sel and v_sel) else [])
        _trimmed = len(combos)
        combos = combos[:MAX_LINES]

        # Find similar drugs **before** the graph, because they are overlaid on the same plot.
        #   B (predicted PK) uses the CL and V predicted for the first combination actually drawn.
        #   Changing the method updates both the list and the curve.
        ref_cl_n, ref_v_n = (combos[0] if combos
                             else (ND_CL["algo"], ND_V["algo"]))
        pk_cl = pc.get(ref_cl_n, {}).get("pred")
        pk_v = pv.get(ref_v_n, {}).get("pred")
        sim_feat = predict.build_sim_features(sim_row, inv_row)
        mol_nb = predict.nearest_drugs(SIM_FEATS, SIM_LABELS, sim_feat, top_n=3)
        pk_nb = predict.nearest_drugs_by_pk(SIM_LABELS, pk_cl, pk_v, top_n=3)

        if not combos:
            st.info("Select at least one CL method and one V method to simulate.")
        else:
            if _trimmed > MAX_LINES:
                st.warning(f"{_trimmed} CL×V combinations selected — plotting only the first "
                           f"{MAX_LINES} to keep the chart readable. Narrow your selection to see the rest.")

            st.table(pd.DataFrame([{
                "CL method": MODEL_LABEL.get(cl_n, cl_n), "V method": MODEL_LABEL.get(v_n, v_n),
                "Predicted CL (L/day)": _sig(pc[cl_n]["pred"]), "Predicted V (L)": _sig(pv[v_n]["pred"]),
            } for cl_n, v_n in combos]))
            st.caption(f"Allometry reference — CL: {_sig(a_cl)} L/day · V: {_sig(a_v)} L."
                       if (a_cl or a_v) else
                       "Allometry reference — not available (no monkey CL/V entered).")

            # Interval <= 0 means a single dose. As in tab 1, put 12 weeks in the interval field.
            single = nd["interval_w"] <= 0
            reg = {"dose_mg": nd["dose_mg"],
                   "interval_days": float(C.SIM_DAYS_DEFAULT) if single else nd["interval_w"] * 7,
                   "route": nd["route"], "infusion_days": (nd["inf_h"] / 24.0) if nd["route"] == "IV" else 0.0,
                   "indication": "", "single_dose": single}
            _sched_nd = "single dose" if single else f"Q{nd['interval_w']:g}W"
            n_doses, horizon = sim_window(reg)
            # Colors split by group. New drug blue; similar drugs orange (molecular properties) and
            # green (predicted PK). Similar drugs are reference lines, so thin and dashed.
            BLUES = ["#1f77b4", "#4c9fd4", "#0d4f7a",
                     "#7fc4ea", "#12608f", "#a8d8f0"]
            MOLC = ["#E8873A", "#B85C10", "#F5C08A"]
            PKC = ["#2E8B57", "#166B3C", "#7FC79B"]

            def _draw(p_cl, p_v, label, color, dash, width, group):
                """Draw one curve from a CL/V pair. The regimen is the same for all.
                Legend is one line below the graph; instead of a group title, (closest: ...) is appended to the name."""
                if not (_pos(p_cl) and _pos(p_v)):
                    return
                t, c = newdrug.simulate(p_cl, p_v, reg, n_doses, horizon)
                # Demo build: similar-drug curves show shape only; hover values are hidden (prevents
                # back-calculating observed CL/V)
                _hide = DEMO_TOP5 and not PUBLIC and group != "sim"   # public build: neighbours are Top-5, values are public
                fig.add_trace(go.Scatter(
                    x=t, y=c, name=label, legendgroup=group,
                    hoverinfo="skip" if _hide else None,
                    line=dict(color=color, dash=dash, width=width)))

            fig = go.Figure()
            for i, (cl_n, v_n) in enumerate(combos):
                cl_lbl = MODEL_LABEL.get(cl_n, cl_n)
                v_lbl = MODEL_LABEL.get(v_n, v_n)
                _draw(pc[cl_n]["pred"], pv[v_n]["pred"],
                      f"Predicted (CL:{cl_lbl} · V:{v_lbl})", BLUES[i % len(BLUES)],
                      "solid", 2.6, "sim")

            # Similar drugs: drawn with their **observed** human CL/V under the same regimen.
            # Only the **closest one** of each list is drawn; drawing all 3 each gives seven lines
            # and hides the new drug's curve. The table still lists all 3.
            # The curve uses the closest drug that has both human CL and V (33 of the 231 reference
            # drugs lack CL or V).
            _mol_draw = [n for n in mol_nb
                         if _pos(n.get("human_CL")) and _pos(n.get("human_V"))][:1]
            for i, nb in enumerate(_mol_draw):
                _draw(nb.get("human_CL"), nb.get("human_V"), f"{nb['INN']} (closest: molecular)",
                      MOLC[i % len(MOLC)], "dash", 1.8, "mol")
            # If the first drug of both lists is the same, the same curve would be drawn twice (e.g.
            # Avelumab input). Keep the molecular-side line; the PK side uses the next drawable
            # drug.
            _mol_inn = {str(n["INN"]) for n in _mol_draw}
            _pk_ok = [n for n in pk_nb if _pos(n.get("human_CL")) and _pos(n.get("human_V"))]
            _pk_draw = [n for n in _pk_ok if str(n["INN"]) not in _mol_inn][:1]
            for i, nb in enumerate(_pk_draw):
                _draw(nb.get("human_CL"), nb.get("human_V"), f"{nb['INN']} (closest: PK)",
                      PKC[i % len(PKC)], "dot", 1.8, "pk")

            fig.update_layout(xaxis_title="Time (day)", yaxis_title="Concentration (µg/mL)",
                              height=540, margin=dict(t=30, b=120),
                              legend=dict(orientation="h", yanchor="top", y=-0.2, x=0.5, xanchor="center",
                                          groupclick="toggleitem"))
            space_legend(fig)       # same item spacing as the Existing Drugs legend
            style_axes(fig)
            # Cut the bottom of the log axis at 1/10^4 of this drug's curve peak. Fast-clearing
            # reference drugs (ADC etc.) reaching 1e-10 stretch the axis and flatten the curves. The
            # top covers the reference curves and the target concentration line too.
            _tops = [float(np.nanmax(tr.y)) for tr in fig.data
                     if tr.legendgroup == "sim" and tr.y is not None and len(tr.y)]
            _refs = [float(np.nanmax(tr.y)) for tr in fig.data
                     if tr.legendgroup in ("mol", "pk") and tr.y is not None and len(tr.y)]
            if _tops and max(_tops) > 0:
                _hi = max([max(_tops) * 3]
                          + [x * 1.5 for x in _refs if np.isfinite(x) and x > 0]
                          + ([float(nd["target_conc"]) * 1.5] if nd.get("target_conc") else []))
                fig.update_yaxes(range=[np.log10(max(_tops) * 1e-4), np.log10(_hi)])
            # Fast clearance: narrow the x axis to the meaningful range (same as tab 1)
            if pk_cl and pk_v:
                _t0, _c0 = newdrug.simulate(pk_cl, pk_v, reg, n_doses, horizon)
                _xh = _view_window(_t0, _c0, {"single_dose": reg["single_dose"],
                                              "interval_days": reg["interval_days"]},
                                   horizon, thalf=predicted_thalf(pk_cl, pk_v))
                if _xh is not None:
                    fig.update_xaxes(range=[0, _xh])
                    _th0 = predicted_thalf(pk_cl, pk_v)
                    st.caption(f"Short predicted half-life (~{_th0:.1f} day) — "
                               f"x-axis zoomed to the informative part of the curve.")
            # Target concentration baseline: only when entered (0 = unused). add_hline breaks the
            # axis auto-range with log axis + dtick="D2" (verified), so it is drawn as a data trace
            # (horizontal dotted line), which is always safe on a log axis.
            if nd.get("target_conc"):
                fig.add_trace(go.Scatter(
                    x=[0, horizon], y=[nd["target_conc"]] * 2, mode="lines",
                    line=dict(color="red", dash="dash", width=2),
                    name=f"Target {nd['target_conc']:g} µg/mL", hoverinfo="skip"))
            kpi_row(pk_cl, pk_v, reg, newdrug_tab=True)
            chart_block(fig, "newdrug_" + str(nd["inn"]).replace(" ", "_"),
                        title=f"{nd['inn']} — {nd['dose_mg']:g} mg {_sched_nd} {nd['route']}",
                        params=[{"drug": nd["inn"], "CL_method": MODEL_LABEL.get(c_, c_),
                                 "V_method": MODEL_LABEL.get(v_, v_),
                                 "CL_pred_Lday": pc[c_]["pred"], "V_pred_L": pv[v_]["pred"],
                                 "CL_allometry_Lday": a_cl, "V_allometry_L": a_v,
                                 "MW_kDa": nd["mw"], "monkey_CL_Lday": nd["mk_cl"],
                                 "monkey_V_L": nd["mk_v"], "monkey_bw_kg": nd["mk_bw"],
                                 "dose_mg": nd["dose_mg"], "interval_week": nd["interval_w"],
                                 "route": nd["route"], "model_build": RES.get("built")}
                                for c_, v_ in combos])
            st.caption(f"Sequence-free models (CL {MODEL_LABEL.get(ND_CL['algo'], ND_CL['algo'])}, "
                       f"V {MODEL_LABEL.get(ND_V['algo'], ND_V['algo'])})."
                       + (f" Curves use SC bioavailability F = {newdrug.F_SC:.2f}."
                          if newdrug.is_sc(nd["route"]) else ""),
                       help="Medians of published popPK models; tested against observed "
                            "profiles of five marketed antibodies.")
            if mol_nb or pk_nb:
                st.caption("Dashed/dotted lines = closest known drug from each list below, on the "
                          "same regimen. Context only, not a prediction.")

        # -- Similar-drug candidates: the same drugs already overlaid on the graph above (computed
        # before the graph; see the comment above).
        if mol_nb or pk_nb:
            st.markdown("### Closest existing drugs (reference only)")
            both = {n["INN"] for n in mol_nb} & {n["INN"] for n in pk_nb}

            def _nb_name(n):
                # appears in both lists -> strongest similar case, marked with ◎
                return f"◎ {n['INN']}" if n["INN"] in both else str(n["INN"])

            st.caption(f"Nearest neighbours among {len(SIM_LABELS)} training-set drugs, ranked two ways. "
                       "Context only."
                       + ("  ◎ = appears in both lists." if both else ""))
            cc = st.columns(2)
            with cc[0]:
                st.markdown("**A. By Molecular Features - Drug Characteristics Similarity**")
                if mol_nb and DEMO_TOP5 and not PUBLIC:
                    # demo build: names only; do not show similar drugs' values (distance, observed CL/V)
                    st.table(pd.DataFrame([{"#": i, "Drug": _nb_name(n)}
                                           for i, n in enumerate(mol_nb, 1)]).set_index("#"))
                elif mol_nb:
                    st.table(pd.DataFrame([{
                        "#": i, "Drug": _nb_name(n), "Distance": f"{n['distance']:.2f}",
                        "Obs CL (L/day)": _sig(n["human_CL"]), "Obs V (L)": _sig(n["human_V"]),
                    } for i, n in enumerate(mol_nb, 1)]).set_index("#"))
                    st.caption("Distance = standardized molecular-feature distance "
                              "(MW, DAR, isotype, category, Kd, IC50, EC50); lower = more similar.")
                else:
                    st.info("No molecular-feature neighbours available.")
            with cc[1]:
                st.markdown("**B. By Predicted PK — PK Profile Similarity**")
                if pk_nb and DEMO_TOP5 and not PUBLIC:
                    st.table(pd.DataFrame([{"#": i, "Drug": _nb_name(n)}
                                           for i, n in enumerate(pk_nb, 1)]).set_index("#"))
                elif pk_nb:
                    st.table(pd.DataFrame([{
                        "#": i, "Drug": _nb_name(n),
                        "CL fold": f"{n['fold_CL']:.2f}×", "V fold": f"{n['fold_V']:.2f}×",
                        "Obs CL (L/day)": _sig(n["human_CL"]), "Obs V (L)": _sig(n["human_V"]),
                    } for i, n in enumerate(pk_nb, 1)]).set_index("#"))
                    st.caption(f"Ranked by distance in log(CL)–log(V) space vs. this drug's prediction "
                               f"(CL {_sig(pk_cl)} L/day · V {_sig(pk_v)} L). fold = 1.00× is identical.")
                else:
                    st.info("No PK neighbours — a predicted CL/V is needed first.")
