"""Visual design layer: global CSS plus small HTML components (hero, section, chips).

Colours come from the same tokens as the charts (common.py) so the whole app reads
as one system. All user-provided text is HTML-escaped before it is rendered.
"""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
:root {
  --bg: #0a0f1e; --card: rgba(19, 27, 48, 0.72); --card-solid: #131b30;
  --line: rgba(144, 133, 233, 0.16); --line-strong: rgba(57, 135, 229, 0.55);
  --ink: #e8ecf6; --ink-2: #a4adc4; --ink-3: #7c87a3;
  --blue: #3987e5; --violet: #9085e9; --cyan: #5fd4e8;
  --grad: linear-gradient(100deg, #3987e5 0%, #9085e9 100%);
}

/* ---- canvas: deep navy with two soft light sources --------------------- */
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(57, 135, 229, 0.16), transparent 62%),
    radial-gradient(900px 480px at 102% 4%, rgba(144, 133, 233, 0.14), transparent 58%),
    radial-gradient(700px 400px at 50% 110%, rgba(95, 212, 232, 0.06), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"], .block-container { padding-top: 2.4rem; max-width: 1440px; }

/* ---- entrance animation ------------------------------------------------- */
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
[data-testid="stMainBlockContainer"] > div { animation: rise .45s ease-out both; }

/* ---- sidebar ------------------------------------------------------------ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1124 0%, #070b16 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebarNav"] a { border-radius: 10px; transition: background .15s ease; }
[data-testid="stSidebarNav"] a:hover { background: rgba(144, 133, 233, 0.10); }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: linear-gradient(90deg, rgba(57,135,229,.22), rgba(144,133,233,.14));
  box-shadow: inset 2px 0 0 var(--violet);
}

/* ---- glass cards: metrics and bordered containers ----------------------- */
[data-testid="stMetric"] {
  background: linear-gradient(160deg, rgba(25, 36, 64, 0.78), rgba(14, 20, 38, 0.78));
  border: 1px solid var(--line) !important; border-radius: 16px !important;
  padding: 16px 18px !important; backdrop-filter: blur(8px);
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255,255,255,0.04);
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-2px); border-color: var(--line-strong) !important;
  box-shadow: 0 14px 36px rgba(57, 135, 229, 0.16);
}
[data-testid="stMetricLabel"] p {
  color: var(--ink-2) !important; text-transform: uppercase; letter-spacing: .04em;
  font-size: .7rem !important; font-weight: 600;
}
[data-testid="stMetricValue"] { font-family: "Space Grotesk", Inter, sans-serif; font-weight: 600;
  font-size: 2rem !important; }

[data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stVerticalBlock"][style*="border"] {
  border-radius: 16px;
}
[class*="st-key-glass"], [class*="st-key-card"] {
  background: linear-gradient(160deg, rgba(25, 36, 64, 0.70), rgba(14, 20, 38, 0.70));
  border: 1px solid var(--line); border-radius: 16px; padding: 18px 20px;
  transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
}
[class*="st-key-card"] { min-height: 196px; }
[class*="st-key-card"] h5 { font-size: 1.08rem; margin-bottom: 2px; }
[class*="st-key-card"]:hover {
  transform: translateY(-3px); border-color: var(--line-strong);
  box-shadow: 0 16px 40px rgba(57, 135, 229, 0.18);
}

/* ---- buttons, tabs, pills, inputs --------------------------------------- */
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
  background: var(--grad); border: 0; color: #fff; font-weight: 600;
  box-shadow: 0 8px 22px rgba(96, 110, 233, 0.32);
}
.stButton > button:hover, .stDownloadButton > button:hover { filter: brightness(1.08); }
[data-baseweb="tab-list"] { gap: 6px; }
[data-baseweb="tab"] { border-radius: 999px !important; padding: 6px 14px !important; }
[data-baseweb="tab"][aria-selected="true"] { background: rgba(144, 133, 233, 0.16); }
[data-testid="stPageLink"] a { border-radius: 10px; }

/* ---- tables, expanders, code, alerts ------------------------------------ */
[data-testid="stDataFrame"] {
  border-radius: 14px; overflow: hidden; border: 1px solid var(--line);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
}
[data-testid="stExpander"] details { border-radius: 14px; border-color: var(--line); background: rgba(19,27,48,.45); }
[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stChatMessage"] { background: rgba(19, 27, 48, 0.55); border: 1px solid var(--line); border-radius: 16px; }

/* ---- hero ---------------------------------------------------------------- */
.hero { position: relative; padding: 30px 34px 26px; margin: -6px 0 22px; border-radius: 22px;
  background:
    radial-gradient(600px 200px at 0% 0%, rgba(57,135,229,.28), transparent 70%),
    radial-gradient(500px 220px at 100% 0%, rgba(144,133,233,.24), transparent 70%),
    linear-gradient(160deg, rgba(22, 32, 58, .92), rgba(12, 18, 36, .92));
  border: 1px solid var(--line); overflow: hidden;
  box-shadow: 0 20px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.05); }
.hero::after { content: ""; position: absolute; inset: 0; pointer-events: none;
  background-image: linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px),
                    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px);
  background-size: 34px 34px; mask-image: linear-gradient(180deg, rgba(0,0,0,.9), transparent 85%); }
.hero-eyebrow { display: inline-flex; align-items: center; gap: 8px; font-size: .72rem; font-weight: 600;
  letter-spacing: .14em; text-transform: uppercase; color: var(--cyan); margin-bottom: 10px; }
.hero-title { font-family: "Space Grotesk", Inter, sans-serif; font-size: 2.35rem; line-height: 1.12;
  font-weight: 700; margin: 0; letter-spacing: -.01em;
  background: linear-gradient(92deg, #f4f6fc 0%, #b9d4ff 45%, #c4bbff 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero-sub { color: var(--ink-2); font-size: 1rem; line-height: 1.55; margin: 10px 0 0; max-width: 980px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px;
  font-size: .78rem; color: var(--ink); background: rgba(255,255,255,.05); border: 1px solid var(--line); }
.chip b { color: #b9d4ff; font-weight: 600; }
.pulse { width: 8px; height: 8px; border-radius: 50%; background: #2ee6a8; box-shadow: 0 0 0 0 rgba(46,230,168,.6);
  animation: pulse 2s infinite; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(46,230,168,.55); } 70% { box-shadow: 0 0 0 9px rgba(46,230,168,0); }
  100% { box-shadow: 0 0 0 0 rgba(46,230,168,0); } }

/* ---- section header ----------------------------------------------------- */
.section { display: flex; align-items: baseline; gap: 12px; margin: 26px 0 10px; }
.section-bar { width: 4px; height: 20px; border-radius: 4px; background: var(--grad); align-self: center; }
.section-title { font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.22rem; font-weight: 600; color: var(--ink); }
.section-sub { color: var(--ink-3); font-size: .86rem; }

/* ---- callout ------------------------------------------------------------- */
.callout { display: flex; gap: 14px; align-items: flex-start; padding: 16px 18px; border-radius: 16px;
  background: linear-gradient(100deg, rgba(57,135,229,.14), rgba(144,133,233,.10));
  border: 1px solid rgba(57,135,229,.35); color: var(--ink); line-height: 1.55; margin: 6px 0 14px; }
.callout .icon { font-size: 1.2rem; line-height: 1.3; }
.callout b { color: #cfe0ff; }

/* ---- status chips -------------------------------------------------------- */
.status { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600; }
.status-good { color: #7ee3a0; background: rgba(12,163,12,.14); border: 1px solid rgba(12,163,12,.35); }
.status-warn { color: #ffc27a; background: rgba(236,131,90,.14); border: 1px solid rgba(236,131,90,.35); }
.status-bad  { color: #ff9a9a; background: rgba(208,59,59,.16); border: 1px solid rgba(208,59,59,.40); }

/* ---- next-best-action cards ---------------------------------------------- */
.nba { height: 100%; padding: 16px 18px; border-radius: 16px; border: 1px solid var(--line);
  background: linear-gradient(160deg, rgba(25, 36, 64, 0.78), rgba(14, 20, 38, 0.78));
  transition: transform .18s ease, border-color .18s ease; }
.nba:hover { transform: translateY(-2px); border-color: var(--line-strong); }
.nba.best { border-color: rgba(46, 230, 168, .45); box-shadow: 0 0 0 1px rgba(46,230,168,.15), 0 12px 30px rgba(46,230,168,.10); }
.nba-rank { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--ink-3); font-weight: 600; }
.nba.best .nba-rank { color: #2ee6a8; }
.nba-title { font-family: "Space Grotesk", Inter, sans-serif; font-weight: 600; font-size: 1.02rem; color: var(--ink); margin: 4px 0 10px; min-height: 2.6em; }
.nba-risk { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.nba-from { color: var(--ink-3); font-size: 1.05rem; text-decoration: line-through; }
.nba-to { font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.9rem; font-weight: 600; color: var(--ink); }
.nba-drop { font-size: .75rem; font-weight: 600; color: #7ee3a0; background: rgba(12,163,12,.14);
  border: 1px solid rgba(12,163,12,.35); padding: 2px 8px; border-radius: 999px; }
.nba-money { margin-top: 10px; font-size: .8rem; color: var(--ink-2); line-height: 1.6; }
.nba-money .pos { color: #7ee3a0; font-weight: 600; } .nba-money .neg { color: #ff9a9a; font-weight: 600; }

/* ---- sidebar brand ------------------------------------------------------- */
.brand { display: flex; align-items: center; gap: 10px; padding: 4px 2px 12px; }
.brand-mark { width: 34px; height: 34px; border-radius: 10px; background: var(--grad);
  display: grid; place-items: center; font-weight: 700; color: #fff; font-family: "Space Grotesk", sans-serif;
  box-shadow: 0 6px 18px rgba(96,110,233,.45); }
.brand-name { font-family: "Space Grotesk", sans-serif; font-weight: 600; font-size: 1rem; color: var(--ink); line-height: 1.1; }
.brand-sub { font-size: .7rem; color: var(--ink-3); letter-spacing: .08em; text-transform: uppercase; }
.side-foot { font-size: .75rem; color: var(--ink-3); line-height: 1.5; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _e(text: str) -> str:
    return html.escape(str(text))


def hero(title: str, subtitle: str = "", eyebrow: str = "", chips: list[str] | None = None,
         live: bool = False) -> None:
    """Page header. `chips` may contain simple <b> markup (trusted, developer-written)."""
    pulse = '<span class="pulse"></span>' if live else ""
    eyebrow_html = f'<div class="hero-eyebrow">{pulse}{_e(eyebrow)}</div>' if eyebrow else ""
    sub_html = f'<p class="hero-sub">{_e(subtitle)}</p>' if subtitle else ""
    chips_html = ("<div class='chips'>" + "".join(f"<span class='chip'>{c}</span>" for c in chips)
                  + "</div>") if chips else ""
    st.markdown(f"<div class='hero'>{eyebrow_html}<div class='hero-title'>{_e(title)}</div>"
                f"{sub_html}{chips_html}</div>", unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    sub = f"<span class='section-sub'>{_e(subtitle)}</span>" if subtitle else ""
    st.markdown(f"<div class='section'><span class='section-bar'></span>"
                f"<span class='section-title'>{_e(title)}</span>{sub}</div>",
                unsafe_allow_html=True)


def callout(html_text: str, icon: str = "💡") -> None:
    """Highlighted insight. `html_text` is developer-written and may use <b>."""
    st.markdown(f"<div class='callout'><span class='icon'>{icon}</span><div>{html_text}</div></div>",
                unsafe_allow_html=True)


def action_card(label: str, title: str, before: float, after: float, saved: float, cost: float,
                net: float, highlight: bool = False) -> None:
    """Next-best-action card: risk before -> after and the money if the offer is accepted."""
    def usd(v: float) -> str:
        return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"
    st.markdown(
        f"<div class='nba{' best' if highlight else ''}'>"
        f"<div class='nba-rank'>{_e(label)}</div><div class='nba-title'>{_e(title)}</div>"
        f"<div class='nba-risk'><span class='nba-from'>{before:.0%}</span>"
        f"<span class='nba-to'>{after:.0%}</span>"
        f"<span class='nba-drop'>&minus;{(before - after) * 100:.0f} pts</span></div>"
        f"<div class='nba-money'>Margin saved {usd(saved)} &middot; offer {usd(cost)}<br>"
        f"Net if accepted <span class='{'pos' if net >= 0 else 'neg'}'>{usd(net)}</span></div>"
        f"</div>", unsafe_allow_html=True)


def status_chip(status: str) -> str:
    kind = {"Stable": "good", "LOW": "good", "Watch": "warn", "MEDIUM": "warn"}.get(status, "bad")
    return f"<span class='status status-{kind}'>{_e(status)}</span>"


def sidebar_brand() -> None:
    st.sidebar.markdown(
        "<div class='brand'><div class='brand-mark'>C</div><div>"
        "<div class='brand-name'>Churn Intelligence</div>"
        "<div class='brand-sub'>Prediction &amp; BI</div></div></div>",
        unsafe_allow_html=True)
