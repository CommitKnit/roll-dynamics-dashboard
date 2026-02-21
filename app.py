# ── 1. ALL IMPORTS FIRST ──────────────────────
import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Output, Input
from datetime import datetime, timedelta
import pytz

# ── 2. CONFIG ─────────────────────────────────
ACCESS_TOKEN    = os.environ.get("ACCESS_TOKEN", "")
KEY_CURRENT     = "NSE_FO|59464"      # ← your key
KEY_NEXT        = "NSE_FO|52028"      # ← your key
SYMBOL          = "SBICARD"
REFRESH_SECONDS = 60
LOOKBACK_SHORT  = 12
LOOKBACK_LONG   = 37
HISTORY_DAYS    = 5                   # ← change this to get more/less days

IST          = pytz.timezone("Asia/Kolkata")
MARKET_OPEN  = pd.Timestamp("09:15").time()
MARKET_CLOSE = pd.Timestamp("15:30").time()

HEADERS = {
    "Content-Type": "application/json",
    "Accept":       "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}"
}

NSE_HOLIDAYS_2025 = {
    "2025-01-26", "2025-02-26", "2025-03-14", "2025-03-31",
    "2025-04-10", "2025-04-14", "2025-04-18", "2025-05-01",
    "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21",
    "2025-10-22", "2025-11-05", "2025-12-25"
}


# ── 3. HELPER — Market status ──────────────────
def get_market_status():
    now      = datetime.now(IST)
    date_str = now.strftime("%Y-%m-%d")
    is_holiday = date_str in NSE_HOLIDAYS_2025
    is_weekend = now.weekday() >= 5
    is_open    = (
        not is_holiday and
        not is_weekend and
        MARKET_OPEN <= now.time() <= MARKET_CLOSE
    )
    return now, date_str, is_open, is_holiday, is_weekend


# ── 4. FETCH FUNCTIONS ────────────────────────

def fetch_historical(instrument_key: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    """Fetch last N days of 1-min historical candles."""
    to_date   = datetime.now(IST).strftime("%Y-%m-%d")
    from_date = (datetime.now(IST) - timedelta(days=days + 4)).strftime("%Y-%m-%d")
    encoded   = instrument_key.replace("|", "%7C")
    url = (
        f"https://api.upstox.com/v3/historical-candle/"
        f"{encoded}/minutes/1/{to_date}/{from_date}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        candles = resp.json().get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(
            candles,
            columns=["timestamp","open","high","low","close","volume","oi"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"Historical fetch error: {e}")
        return pd.DataFrame()


def fetch_intraday(instrument_key: str) -> pd.DataFrame:
    """Fetch today's live 1-min intraday candles."""
    encoded = instrument_key.replace("|", "%7C")
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{encoded}/minutes/1"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return pd.DataFrame()
        candles = resp.json().get("data", {}).get("candles", [])
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(
            candles,
            columns=["timestamp","open","high","low","close","volume","oi"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        return df.sort_values("timestamp").reset_index(drop=True)
    except Exception as e:
        print(f"Intraday fetch error: {e}")
        return pd.DataFrame()


def merge_historical_intraday(hist: pd.DataFrame, intra: pd.DataFrame) -> pd.DataFrame:
    """Combine historical + intraday, deduplicate, sort."""
    if hist.empty and intra.empty:
        return pd.DataFrame()
    if hist.empty:
        return intra
    if intra.empty:
        return hist
    df = (
        pd.concat([hist, intra], ignore_index=True)
          .drop_duplicates(subset="timestamp")
          .sort_values("timestamp")
          .reset_index(drop=True)
    )
    return df


# ── 5. RESAMPLE + PIVOT + ROLL PRESSURE ───────

def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy().set_index("timestamp")
    df_5 = df.resample("5min", closed="left", label="left").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        "oi":     "last"
    }).dropna(subset=["close"]).reset_index()

    # Keep only trading hours
    df_5 = df_5[
        (df_5["timestamp"].dt.time >= MARKET_OPEN) &
        (df_5["timestamp"].dt.time <= MARKET_CLOSE) &
        (df_5["timestamp"].dt.weekday < 5) &
        (~df_5["timestamp"].dt.strftime("%Y-%m-%d").isin(NSE_HOLIDAYS_2025))
    ].reset_index(drop=True)

    return df_5


def build_pivot(df_cur: pd.DataFrame, df_nxt: pd.DataFrame) -> pd.DataFrame:
    df_cur = df_cur.rename(columns={
        "open":"open_current","high":"high_current","low":"low_current",
        "close":"close_current","volume":"volume_current","oi":"oi_current"
    })
    df_nxt = df_nxt.rename(columns={
        "open":"open_next","high":"high_next","low":"low_next",
        "close":"close_next","volume":"volume_next","oi":"oi_next"
    })
    df = pd.merge(df_cur, df_nxt, on="timestamp", how="inner")
    df = df.dropna(subset=["close_current","close_next"]).reset_index(drop=True)
    df["roll_spread"] = df["close_current"] - df["close_next"]
    return df


def calculate_roll_pressure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for lookback, suffix in [(LOOKBACK_SHORT,"short"),(LOOKBACK_LONG,"long")]:
        oi_chg_cur = df["oi_current"].diff(lookback)
        oi_chg_nxt = df["oi_next"].diff(lookback)
        total_oi   = df["oi_current"] + df["oi_next"]
        both_legs  = (oi_chg_cur < 0) & (oi_chg_nxt > 0)
        raw        = -oi_chg_cur + oi_chg_nxt
        df[f"roll_pressure_{suffix}"]            = raw.where(both_legs, other=0)
        df[f"roll_pressure_{suffix}_normalized"] = (
            df[f"roll_pressure_{suffix}"] / total_oi.replace(0, pd.NA)
        )
        df[f"roll_confirmed_{suffix}"] = both_legs.astype(int)

    total_oi = df["oi_current"] + df["oi_next"]
    df["roll_completion_pct"] = (df["oi_next"] / total_oi.replace(0, pd.NA)) * 100
    df["roll_signal"] = (
        (df["roll_pressure_short"] > 0) &
        (df["roll_pressure_long"]  > 0) &
        (df["roll_confirmed_short"] == 1) &
        (df["roll_confirmed_long"]  == 1)
    ).astype(int)
    return df


# ── 6. MASTER PIPELINE ────────────────────────

def fetch_and_build() -> pd.DataFrame:
    now, date_str, is_open, is_holiday, is_weekend = get_market_status()

    print(f"  [{now.strftime('%H:%M:%S')}] Market {'OPEN' if is_open else 'CLOSED'} — fetching data...")

    # Always fetch historical (last N days)
    hist_cur = fetch_historical(KEY_CURRENT, HISTORY_DAYS)
    hist_nxt = fetch_historical(KEY_NEXT,    HISTORY_DAYS)

    if is_open:
        # Market open — also fetch live intraday and merge
        intra_cur = fetch_intraday(KEY_CURRENT)
        intra_nxt = fetch_intraday(KEY_NEXT)
        raw_cur   = merge_historical_intraday(hist_cur, intra_cur)
        raw_nxt   = merge_historical_intraday(hist_nxt, intra_nxt)
        print(f"  Live merge: {len(raw_cur)} cur bars, {len(raw_nxt)} nxt bars")
    else:
        # Market closed — use historical only
        raw_cur = hist_cur
        raw_nxt = hist_nxt
        print(f"  Historical only: {len(raw_cur)} cur bars, {len(raw_nxt)} nxt bars")

    if raw_cur.empty or raw_nxt.empty:
        print("  ⚠ No data available")
        return pd.DataFrame()

    df_cur   = resample_to_5min(raw_cur)
    df_nxt   = resample_to_5min(raw_nxt)
    df_pivot = build_pivot(df_cur, df_nxt)

    if df_pivot.empty:
        print("  ⚠ Pivot is empty")
        return pd.DataFrame()

    df_pivot = calculate_roll_pressure(df_pivot)
    print(f"  ✅ {len(df_pivot)} 5-min bars | {df_pivot['timestamp'].min()} → {df_pivot['timestamp'].max()}")
    return df_pivot


# ── 7. BUILD FIGURE ───────────────────────────

def build_figure(df: pd.DataFrame, status_label: str = "") -> go.Figure:

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark", paper_bgcolor="#0d0d1a",
            plot_bgcolor="#111122",
            title=dict(text="⏳  Waiting for data — check API token and instrument keys",
                       font=dict(color="white", size=14)),
            height=400
        )
        return fig

    x      = list(range(len(df)))
    labels = df["timestamp"].dt.strftime("%d %b  %H:%M").tolist()

    def bar_colors(series, pos_col, neg_col):
        return [pos_col if v > 0 else neg_col for v in series.fillna(0)]

    # Day boundary lines
    day_boundaries = []
    prev = None
    for i, ts in enumerate(df["timestamp"]):
        d = ts.date()
        if prev is not None and d != prev:
            day_boundaries.append(i)
        prev = d

    # X-axis day labels
    tick_vals, tick_text = [], []
    for d, grp in df.groupby(df["timestamp"].dt.date):
        idxs = grp.index.tolist()
        tick_vals.append(idxs[len(idxs) // 2])
        tick_text.append(pd.Timestamp(d).strftime("%d %b"))

    updated_at = datetime.now(IST).strftime("%H:%M:%S IST")

    fig = make_subplots(
        rows=7, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.22, 0.13, 0.13, 0.13, 0.11, 0.14, 0.14],
        specs=[
            [{"secondary_y": True}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
        subplot_titles=(
            f"{SYMBOL}  |  Price — Current vs Next  +  Roll Spread",
            "Roll Pressure (Raw)  —  Short=60min  |  Long=3hr",
            "Normalized Roll Pressure — SHORT (60 min)  |  Blue=rolling  Red=reversing",
            "Normalized Roll Pressure — LONG  (3 hr)   |  Gold zone = both positive",
            f"Roll Completion %  —  <35% Early  |  35–55% Active  |  >55% Late",
            "Open Interest  —  Blue falling + Red rising = roll confirmed",
            "Volume  —  Red overtaking Blue = liquidity shifting"
        )
    )

    def add_boundaries(row):
        for b in day_boundaries:
            fig.add_vline(
                x=b,
                line=dict(color="rgba(255,255,255,0.07)", width=1, dash="dot"),
                row=row, col=1
            )

    # ── Panel 1: Price + Spread ───────────────
    fig.add_trace(go.Scatter(
        x=x, y=df["close_current"], customdata=labels,
        name="Current Expiry", line=dict(color="#60A5FA", width=1.8),
        hovertemplate="%{customdata}<br>Current: ₹%{y:.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x, y=df["close_next"], customdata=labels,
        name="Next Expiry", line=dict(color="#F87171", width=1.8, dash="dot"),
        hovertemplate="%{customdata}<br>Next: ₹%{y:.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=x, y=df["roll_spread"], customdata=labels,
        name="Roll Spread (₹)", line=dict(color="#FFD700", width=1.2),
        fill="tozeroy", fillcolor="rgba(255,215,0,0.07)",
        hovertemplate="%{customdata}<br>Spread: ₹%{y:.2f}<extra></extra>"
    ), row=1, col=1, secondary_y=True)

    sig = df[df["roll_signal"] == 1]
    if not sig.empty:
        fig.add_trace(go.Scatter(
            x=sig.index.tolist(), y=sig["close_current"],
            mode="markers", name="Roll Signal 🔔",
            marker=dict(symbol="triangle-up", size=9, color="#FFD700",
                        line=dict(color="white", width=1)),
            hoverinfo="skip"
        ), row=1, col=1, secondary_y=False)

    add_boundaries(1)

    # ── Panel 2: Raw Roll Pressure ────────────
    fig.add_trace(go.Bar(
        x=x, y=df["roll_pressure_long"], customdata=labels,
        name="Long Pressure (3hr)",
        marker_color=bar_colors(df["roll_pressure_long"],
                                "rgba(52,211,153,0.45)", "rgba(248,113,113,0.45)"),
        hovertemplate="%{customdata}<br>Long: %{y:,.0f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=df["roll_pressure_short"], customdata=labels,
        name="Short Pressure (60min)", line=dict(color="#A78BFA", width=1.8),
        hovertemplate="%{customdata}<br>Short: %{y:,.0f}<extra></extra>"
    ), row=2, col=1)

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1), row=2, col=1)
    add_boundaries(2)

    # ── Panel 3: NRP Short ────────────────────
    short_nrp = df["roll_pressure_short_normalized"].fillna(0)

    fig.add_trace(go.Bar(
        x=x, y=short_nrp, customdata=labels, name="Short NRP",
        marker_color=bar_colors(short_nrp,
                                "rgba(96,165,250,0.75)", "rgba(248,113,113,0.75)"),
        hovertemplate="%{customdata}<br>Short NRP: %{y:.5f}<extra></extra>"
    ), row=3, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=[short_nrp.mean()] * len(x),
        name=f"Short Mean ({short_nrp.mean():.5f})",
        line=dict(color="rgba(96,165,250,0.35)", width=1, dash="dash"),
        hoverinfo="skip"
    ), row=3, col=1)

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1), row=3, col=1)
    add_boundaries(3)

    # ── Panel 4: NRP Long ─────────────────────
    long_nrp = df["roll_pressure_long_normalized"].fillna(0)
    both_pos = short_nrp.where((short_nrp > 0) & (long_nrp > 0), other=0)

    fig.add_trace(go.Scatter(
        x=x, y=both_pos, fill="tozeroy",
        fillcolor="rgba(255,215,0,0.10)", line=dict(width=0),
        name="Both NRP Positive", hoverinfo="skip"
    ), row=4, col=1)

    fig.add_trace(go.Bar(
        x=x, y=long_nrp, customdata=labels, name="Long NRP",
        marker_color=bar_colors(long_nrp,
                                "rgba(52,211,153,0.75)", "rgba(251,146,60,0.75)"),
        hovertemplate="%{customdata}<br>Long NRP: %{y:.5f}<extra></extra>"
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=[long_nrp.mean()] * len(x),
        name=f"Long Mean ({long_nrp.mean():.5f})",
        line=dict(color="rgba(52,211,153,0.35)", width=1, dash="dash"),
        hoverinfo="skip"
    ), row=4, col=1)

    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1), row=4, col=1)
    add_boundaries(4)

    # ── Panel 5: Roll Completion ──────────────
    fig.add_trace(go.Scatter(
        x=x, y=df["roll_completion_pct"], customdata=labels,
        name="Roll Completion %", line=dict(color="#F59E0B", width=2),
        fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
        hovertemplate="%{customdata}<br>Completion: %{y:.1f}%<extra></extra>"
    ), row=5, col=1)

    fig.add_hline(y=35,
        line=dict(color="rgba(52,211,153,0.55)", width=1.2, dash="dash"),
        annotation_text="35% Early", annotation_position="top right",
        annotation_font=dict(color="#34D399", size=9), row=5, col=1)
    fig.add_hline(y=55,
        line=dict(color="rgba(248,113,113,0.55)", width=1.2, dash="dash"),
        annotation_text="55% Late", annotation_position="top right",
        annotation_font=dict(color="#F87171", size=9), row=5, col=1)

    add_boundaries(5)

    # ── Panel 6: OI ───────────────────────────
    fig.add_trace(go.Scatter(
        x=x, y=df["oi_current"], customdata=labels,
        name="OI Current", line=dict(color="#60A5FA", width=1.8),
        fill="tozeroy", fillcolor="rgba(96,165,250,0.08)",
        hovertemplate="%{customdata}<br>OI Current: %{y:,.0f}<extra></extra>"
    ), row=6, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=df["oi_next"], customdata=labels,
        name="OI Next", line=dict(color="#F87171", width=1.8),
        fill="tozeroy", fillcolor="rgba(248,113,113,0.08)",
        hovertemplate="%{customdata}<br>OI Next: %{y:,.0f}<extra></extra>"
    ), row=6, col=1)

    add_boundaries(6)

    # ── Panel 7: Volume ───────────────────────
    fig.add_trace(go.Bar(
        x=x, y=df["volume_current"], customdata=labels,
        name="Volume Current", marker_color="rgba(96,165,250,0.6)",
        hovertemplate="%{customdata}<br>Vol Current: %{y:,.0f}<extra></extra>"
    ), row=7, col=1)

    fig.add_trace(go.Bar(
        x=x, y=df["volume_next"], customdata=labels,
        name="Volume Next", marker_color="rgba(248,113,113,0.6)",
        hovertemplate="%{customdata}<br>Vol Next: %{y:,.0f}<extra></extra>"
    ), row=7, col=1)

    add_boundaries(7)

    # ── Layout ────────────────────────────────
    fig.update_layout(
        height=1450,
        template="plotly_dark",
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#111122",
        hovermode="x unified",
        barmode="overlay",
        title=dict(
            text=(
                f"<b>{SYMBOL} — Roll Dashboard</b>"
                f"  |  {status_label}"
                f"  |  Last {HISTORY_DAYS} days + today"
                f"  |  Updated: {updated_at}"
            ),
            font=dict(size=13, color="white"), x=0.01
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.005,
            xanchor="left", x=0, font=dict(size=10),
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=70, r=120, t=90, b=50),
    )

    for row in range(1, 8):
        fig.update_xaxes(
            tickvals=tick_vals, ticktext=tick_text,
            tickfont=dict(size=10, color="#888"),
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            showspikes=True, spikecolor="rgba(255,255,255,0.2)",
            spikethickness=1, row=row, col=1
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, row=row, col=1
        )

    fig.update_yaxes(title_text="Price (₹)",    title_font=dict(size=10), row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Spread (₹)",   title_font=dict(size=10), row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Raw Pressure", title_font=dict(size=10), row=2, col=1)
    fig.update_yaxes(title_text="NRP Short",    title_font=dict(size=10), row=3, col=1)
    fig.update_yaxes(title_text="NRP Long",     title_font=dict(size=10), row=4, col=1)
    fig.update_yaxes(title_text="Completion %", title_font=dict(size=10), row=5, col=1, range=[0, 80])
    fig.update_yaxes(title_text="OI",           title_font=dict(size=10), row=6, col=1)
    fig.update_yaxes(title_text="Volume",       title_font=dict(size=10), row=7, col=1)

    return fig


# ── 8. DASH APP ───────────────────────────────
app = Dash(__name__, title=f"{SYMBOL} Roll Dashboard")

app.layout = html.Div(
    style={"backgroundColor": "#0d0d1a", "minHeight": "100vh"},
    children=[
        html.Div(
            style={
                "background": "#111122", "padding": "8px 20px",
                "borderBottom": "1px solid #1e1e3a",
                "display": "flex", "justifyContent": "space-between",
                "fontSize": "12px", "fontFamily": "monospace", "color": "#666"
            },
            children=[
                html.Span(id="status-left", children="Initializing..."),
                html.Span(id="status-right", children=""),
            ]
        ),
        dcc.Graph(
            id="live-chart",
            config={"displayModeBar": True, "scrollZoom": True},
            style={"height": "calc(100vh - 40px)"}
        ),
        dcc.Interval(
            id="interval",
            interval=REFRESH_SECONDS * 1000,
            n_intervals=0
        )
    ]
)


@app.callback(
    Output("live-chart",   "figure"),
    Output("status-left",  "children"),
    Output("status-left",  "style"),
    Output("status-right", "children"),
    Input("interval",      "n_intervals")
)
def update_chart(n):
    now, date_str, is_open, is_holiday, is_weekend = get_market_status()

    # ── Status bar text ───────────────────────
    if is_open:
        left_text  = f"● LIVE  |  {SYMBOL}  |  Refresh #{n}  |  {now.strftime('%H:%M:%S IST')}"
        left_style = {"color": "#34D399"}
        right_text = f"Last {HISTORY_DAYS} days + today  |  Updates every {REFRESH_SECONDS}s"
    elif is_holiday:
        left_text  = f"◉ HOLIDAY  |  {now.strftime('%d %b %Y')}"
        left_style = {"color": "#F59E0B"}
        right_text = f"Showing last {HISTORY_DAYS} trading days — static"
    elif is_weekend:
        left_text  = f"◉ WEEKEND  |  {now.strftime('%d %b %Y')}"
        left_style = {"color": "#F59E0B"}
        right_text = f"Showing last {HISTORY_DAYS} trading days — static"
    elif now.time() < MARKET_OPEN:
        left_text  = f"◉ PRE-MARKET  |  Opens at 09:15 IST  |  {now.strftime('%H:%M IST')}"
        left_style = {"color": "#A78BFA"}
        right_text = f"Showing last {HISTORY_DAYS} trading days — live starts at 09:15"
    else:
        left_text  = f"◉ MARKET CLOSED  |  {now.strftime('%d %b  %H:%M IST')}"
        left_style = {"color": "#F87171"}
        right_text = f"Showing last {HISTORY_DAYS} trading days including today"

    # ── Build status label for chart title ───
    status_label = "● LIVE" if is_open else "◉ Market Closed — Historical"

    df  = fetch_and_build()
    fig = build_figure(df, status_label)

    return fig, left_text, left_style, right_text


# ── 9. THESE TWO LINES MUST BE AT THE BOTTOM ──
server = app.server

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)

