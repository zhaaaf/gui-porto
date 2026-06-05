"""
Portfolio Diversification – Strategic, Tactical, Downside Variance & Piecewise Linear
Gabungan empat model dalam satu aplikasi Streamlit

Jalankan dengan:
    python -m streamlit run main.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpStatus,
    lpSum, value, PULP_CBC_CMD
)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

import urllib.request, json as _json

# ══════════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📈",
    layout="wide",
)

COLORS = ['#4361EE', '#F72585', '#4CC9F0', '#7209B7', '#3A0CA3', '#560BAD']
C_LINE  = '#4361EE'
C_FILL  = '#4361EE'
C_DOT   = '#E63946'
C_GOLD  = '#FFB703'
C_DOWN  = '#F72585'

# ══════════════════════════════════════════════════════════════════════
# DAFTAR SAHAM POPULER (statis)
# ══════════════════════════════════════════════════════════════════════

IDX_POPULAR = {
    # LQ45 Indonesia
    "BBCA.JK": "Bank BCA",
    "BBRI.JK": "Bank BRI",
    "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank BNI",
    "TLKM.JK": "Telkom Indonesia",
    "ASII.JK": "Astra International",
    "GOTO.JK": "GoTo Gojek Tokopedia",
    "BREN.JK": "Barito Renewables",
    "BYAN.JK": "Bayan Resources",
    "MDKA.JK": "Merdeka Copper Gold",
    "AMMN.JK": "Amman Mineral",
    "TPIA.JK": "Chandra Asri",
    "UNVR.JK": "Unilever Indonesia",
    "KLBF.JK": "Kalbe Farma",
    "ICBP.JK": "Indofood CBP",
    "INDF.JK": "Indofood",
    "SMGR.JK": "Semen Indonesia",
    "ADRO.JK": "Adaro Energy",
    "PTBA.JK": "Bukit Asam",
    "ITMG.JK": "Indo Tambangraya",
    "PGAS.JK": "Perusahaan Gas Negara",
    "ANTM.JK": "Aneka Tambang",
    "INCO.JK": "Vale Indonesia",
    "MAPI.JK": "Mitra Adiperkasa",
    "EXCL.JK": "XL Axiata",
}

US_POPULAR = {
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "GOOGL": "Alphabet (Google)",
    "AMZN":  "Amazon",
    "NVDA":  "NVIDIA",
    "META":  "Meta",
    "TSLA":  "Tesla",
    "BRK-B": "Berkshire Hathaway",
    "JPM":   "JPMorgan Chase",
    "JNJ":   "Johnson & Johnson",
    "V":     "Visa",
    "UNH":   "UnitedHealth",
    "XOM":   "ExxonMobil",
    "WMT":   "Walmart",
    "PG":    "Procter & Gamble",
}

ALL_POPULAR = {**IDX_POPULAR, **US_POPULAR}

# ══════════════════════════════════════════════════════════════════════
# TICKER SEARCH (Yahoo Finance API)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=300)
def search_tickers(query: str):
    """Cari ticker via Yahoo Finance search API. Return list of (symbol, name, type)."""
    if not query or len(query.strip()) < 2:
        return []
    try:
        q = urllib.parse.quote(query.strip())
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}&quotesCount=10&newsCount=0&listsCount=0"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            data = _json.loads(r.read())
        results = []
        for item in data.get("quotes", []):
            sym  = item.get("symbol", "")
            name = item.get("longname") or item.get("shortname") or sym
            qtype = item.get("quoteType", "")
            if sym and qtype in ("EQUITY", "ETF", "MUTUALFUND"):
                results.append((sym, name, qtype))
        return results
    except Exception:
        return []

import urllib.parse

# ══════════════════════════════════════════════════════════════════════
# WIDGET PEMILIH TICKER (reusable di semua model)
# ══════════════════════════════════════════════════════════════════════

def ticker_selector(key_prefix: str, default_tickers: str) -> str:
    """
    Widget sidebar untuk memilih ticker:
    - Tab 1: pilih dari daftar populer (IDX + US)
    - Tab 2: cari via Yahoo Finance
    Mengembalikan string ticker dipisahkan koma.
    """
    # State untuk ticker yang sudah dipilih
    sel_key = f"{key_prefix}_selected_tickers"
    if sel_key not in st.session_state:
        st.session_state[sel_key] = [t.strip() for t in default_tickers.split(",") if t.strip()]

    selected = st.session_state[sel_key]

    # ── Tab pilih / cari ──
    tab_pop, tab_search = st.tabs(["Saham Populer", "Cari Saham"])

    with tab_pop:
        st.caption("Centang saham yang ingin dimasukkan:")
        col1, col2 = st.columns(2)
        # Indonesia
        col1.markdown("**Indonesia (IDX)**")
        for ticker, name in IDX_POPULAR.items():
            checked = ticker in selected
            if col1.checkbox(f"{ticker}", value=checked,
                             key=f"{key_prefix}_pop_{ticker}",
                             help=name):
                if ticker not in selected:
                    selected.append(ticker)
            else:
                if ticker in selected:
                    selected.remove(ticker)
        # US
        col2.markdown("**US Stocks**")
        for ticker, name in US_POPULAR.items():
            checked = ticker in selected
            if col2.checkbox(f"{ticker}", value=checked,
                             key=f"{key_prefix}_pop_{ticker}",
                             help=name):
                if ticker not in selected:
                    selected.append(ticker)
            else:
                if ticker in selected:
                    selected.remove(ticker)

    with tab_search:
        search_q = st.text_input("Nama perusahaan / ticker",
                                 key=f"{key_prefix}_search_q",
                                 placeholder="cth: Bank BCA, Tesla, BBCA...")
        if st.button("Cari", key=f"{key_prefix}_search_btn", use_container_width=True):
            st.session_state[f"{key_prefix}_search_results"] = search_tickers(search_q)

        results = st.session_state.get(f"{key_prefix}_search_results", [])
        if results:
            st.caption(f"{len(results)} hasil ditemukan:")
            for sym, name, qtype in results:
                already = sym in selected
                label = f"**{sym}** — {name} _{qtype}_"
                if st.checkbox(label, value=already, key=f"{key_prefix}_sr_{sym}"):
                    if sym not in selected:
                        selected.append(sym)
                else:
                    if sym in selected:
                        selected.remove(sym)
        elif f"{key_prefix}_search_results" in st.session_state:
            st.warning("Tidak ada hasil. Coba kata kunci lain.")

    st.session_state[sel_key] = selected

    # ── Ringkasan yang dipilih ──
    if selected:
        st.markdown(f"**Terpilih ({len(selected)}):** `{', '.join(selected)}`")
    else:
        st.warning("Belum ada saham dipilih.")

    return ", ".join(selected)

# ══════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════

def calculate_returns(price_matrix):
    return (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1] * 100

def calculate_expected_returns(return_matrix):
    return np.mean(return_matrix, axis=0)

def parse_price_inputs(assets_str, prices_str):
    assets = [x.strip() for x in assets_str.split(",")]
    rows   = prices_str.strip().split("\n")
    prices = np.array([[float(v) for v in row.split(",")] for row in rows])
    return assets, prices

DEFAULT_ASSETS = "RD,AKZ,KLM,PHI,UN"
DEFAULT_PRICES = """111.0,82.5,70.0,154.6,110.8
108.1,81.6,73.7,152.4,108.0
107.9,80.1,72.3,146.1,103.7
108.5,83.1,69.7,157.5,106.6
111.4,85.0,69.5,168.4,107.3
115.5,92.6,74.8,166.9,109.5
113.2,91.6,73.8,164.1,108.7
111.9,88.3,70.2,169.0,111.1
99.7,80.8,64.3,143.8,101.0
105.1,86.1,71.8,151.3,105.4
100.9,81.8,71.7,148.3,109.6
105.0,85.6,69.5,140.6,112.8
105.2,84.6,70.5,131.5,113.6
107.0,90.3,74.9,138.0,117.4
109.0,88.3,78.5,135.4,123.1
111.4,85.6,73.0,114.0,124.5
107.2,81.6,74.5,116.1,118.7
111.3,87.4,75.0,121.6,125.0
108.6,87.5,77.0,127.6,127.7
105.6,86.6,72.4,116.2,121.2
105.9,90.0,71.4,128.6,125.1
104.7,91.4,68.9,129.4,119.9
107.7,95.3,69.7,137.1,117.9
107.4,92.9,68.6,134.5,124.6
108.0,97.0,70.2,156.0,124.3
104.7,102.6,74.3,159.5,128.8
112.8,107.0,76.3,155.9,134.2
109.7,110.4,86.0,155.0,140.9
111.7,109.7,88.9,149.9,138.2
120.4,105.9,83.5,153.5,141.6
118.0,105.9,83.4,153.0,140.6
119.7,103.0,84.9,149.9,158.2
116.7,102.4,84.9,153.7,149.6
115.8,107.2,86.1,167.0,152.8
113.7,104.5,78.9,181.0,144.3
115.7,105.8,79.5,189.4,155.5
114.4,104.8,79.1,197.9,154.2
113.8,103.8,79.9,201.7,154.5
114.0,107.0,82.0,196.3,158.0
114.1,107.4,77.2,188.0,163.8
111.5,112.0,78.5,189.9,164.3
109.2,106.8,77.1,172.1,163.9
110.1,105.3,76.1,178.0,165.6
112.8,113.1,82.6,171.0,161.4
111.0,116.6,91.4,179.5,165.4
105.6,126.7,94.5,180.6,160.9
107.3,123.1,90.0,173.5,162.0
103.2,112.3,88.5,164.4,153.0
102.8,103.1,81.8,164.2,141.2
93.9,95.0,80.7,153.0,133.4
93.6,92.7,80.5,164.0,139.3"""

# ══════════════════════════════════════════════════════════════════════
# YAHOO FINANCE HELPER
# ══════════════════════════════════════════════════════════════════════

PERIOD_OPTIONS = {
    "6 bulan": "6mo",
    "1 tahun": "1y",
    "2 tahun": "2y",
    "3 tahun": "3y",
    "5 tahun": "5y",
}

FREQ_OPTIONS = {
    "Mingguan (sesuai AIMMS)": {"resample": "W",  "label": "minggu", "min_periods": 8},
    "Bulanan":                  {"resample": "ME", "label": "bulan",  "min_periods": 4},
    "Harian":                   {"resample": None, "label": "hari",   "min_periods": 20},
}

def _get_close(tickers, period):
    """Helper: download & normalisasi kolom Close dari yfinance."""
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
    if raw.empty:
        return None, "Data tidak ditemukan. Periksa kembali nama ticker."
    if len(tickers) == 1:
        close = raw[["Close"]].copy()
        close.columns = tickers
    else:
        close = raw["Close"].copy()
    return close, None

@st.cache_data(show_spinner=False)
def fetch_yfinance(tickers_str, period, freq_key="Mingguan (sesuai AIMMS)"):
    """Download harga, resample sesuai frekuensi, hitung return & kovarians."""
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return None, None, None, "Tidak ada ticker yang valid."
    freq_cfg = FREQ_OPTIONS[freq_key]
    try:
        close, err = _get_close(tickers, period)
        if err:
            return None, None, None, err
        if freq_cfg["resample"]:
            close = close.resample(freq_cfg["resample"]).last()
        close = close.dropna()
        min_p = freq_cfg["min_periods"]
        if len(close) < min_p:
            return None, None, None, f"Data terlalu sedikit (< {min_p} {freq_cfg['label']})."
        rets = close.pct_change().dropna() * 100
        rets = rets.dropna(axis=1, how="all")
        available_tickers = list(rets.columns)
        return available_tickers, rets.mean().values, rets.cov().values, None
    except Exception as e:
        return None, None, None, f"Error mengambil data: {e}"

@st.cache_data(show_spinner=False)
def fetch_prices(tickers_str, period, freq_key="Mingguan (sesuai AIMMS)"):
    """Download harga & resample sesuai frekuensi, kembalikan array harga."""
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    freq_cfg = FREQ_OPTIONS[freq_key]
    try:
        close, err = _get_close(tickers, period)
        if err:
            return None, None, err
        if freq_cfg["resample"]:
            close = close.resample(freq_cfg["resample"]).last()
        close = close.dropna()
        min_p = freq_cfg["min_periods"]
        if len(close) < min_p:
            return None, None, f"Data terlalu sedikit (< {min_p} {freq_cfg['label']})."
        available = list(close.columns)
        return available, close.values, None
    except Exception as e:
        return None, None, f"Error: {e}"

# ══════════════════════════════════════════════════════════════════════
# MODEL STRATEGIC
# ══════════════════════════════════════════════════════════════════════

def strategic_portfolio_return(x, mu):
    return np.sum(x * mu)

def solve_strategic_portfolio(M_target, mu, cov_matrix):
    n = len(mu)
    def objective(x):
        return x @ cov_matrix @ x
    x0 = np.ones(n) / n
    bounds = Bounds([0] * n, [1] * n)
    constraints = [
        LinearConstraint(np.ones(n), lb=[1.0], ub=[1.0]),
        LinearConstraint(-mu, lb=-np.inf, ub=[-M_target]),
    ]
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        x_res = result.x
        return x_res, result.fun, strategic_portfolio_return(x_res, mu), True
    return None, None, None, False

def parse_strategic_inputs(categories_str, returns_str, cov_str):
    cats = [c.strip() for c in categories_str.split(",")]
    rets = np.array([float(r.strip()) for r in returns_str.split(",")])
    cov_lines = cov_str.strip().split("\n")
    cov = np.array([[float(v.strip()) for v in line.split(",")] for line in cov_lines])
    return cats, rets, cov

@st.cache_data
def compute_strategic_curve(rets_tuple, cov_tuple, n_points=40):
    rets_arr = np.array(rets_tuple)
    cov_arr  = np.array(cov_tuple)
    M_levels = np.linspace(float(min(rets_arr)), float(max(rets_arr)), n_points)
    alloc_list, feasible_M, risks_std = [], [], []
    for M in M_levels:
        x_l, rv_l, _, ok = solve_strategic_portfolio(M, rets_arr, cov_arr)
        if ok:
            alloc_list.append(x_l.tolist())
            feasible_M.append(M)
            risks_std.append(float(np.sqrt(rv_l)))
    return feasible_M, risks_std, alloc_list

# ══════════════════════════════════════════════════════════════════════
# MODEL TACTICAL
# ══════════════════════════════════════════════════════════════════════

def solve_tactical_portfolio(M_target, return_matrix):
    expected_returns = calculate_expected_returns(return_matrix)
    D = return_matrix - expected_returns
    T = return_matrix.shape[0]
    probs = np.ones(T) / T
    def objective(x):
        y = D @ x
        return np.sum(probs * y ** 2)
    n = len(expected_returns)
    x0 = np.ones(n) / n
    bounds = Bounds(np.zeros(n), np.ones(n))
    constraints = [
        LinearConstraint(np.ones((1, n)), [1], [1]),
        LinearConstraint(expected_returns.reshape(1, -1), [M_target], [np.inf]),
    ]
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        return result.x, result.fun, np.sum(result.x * expected_returns), expected_returns, True
    return None, None, None, expected_returns, False

@st.cache_data
def compute_tactical_curve(prices_tuple, n_points=30):
    prices_arr  = np.array(prices_tuple)
    returns_arr = calculate_returns(prices_arr)
    mu          = calculate_expected_returns(returns_arr)
    M_levels    = np.linspace(0, float(max(mu)), n_points)
    alloc_dict  = {i: [] for i in range(prices_arr.shape[1])}
    risks, feasible_M = [], []
    for M in M_levels:
        x, r, _, _, ok = solve_tactical_portfolio(M, returns_arr)
        if ok:
            feasible_M.append(M)
            risks.append(float(np.sqrt(r)))
            for i in range(len(mu)):
                alloc_dict[i].append(float(x[i]))
    return feasible_M, risks, alloc_dict, mu.tolist()

# ══════════════════════════════════════════════════════════════════════
# MODEL DOWNSIDE VARIANCE
# ══════════════════════════════════════════════════════════════════════

def solve_downside_portfolio(M_target, return_matrix):
    mu = calculate_expected_returns(return_matrix)
    T = return_matrix.shape[0]
    n_assets = return_matrix.shape[1]
    probabilities = np.ones(T) / T
    def objective(z):
        q = z[n_assets:]
        return np.sum(probabilities * q ** 2)
    x0 = np.ones(n_assets) / n_assets
    q0 = np.ones(T) * 0.01
    z0 = np.concatenate([x0, q0])
    lb = np.concatenate([np.zeros(n_assets), np.zeros(T)])
    ub = np.concatenate([np.ones(n_assets), np.ones(T) * 1e6])
    bounds = Bounds(lb, ub)
    constraints = []
    Aeq = np.zeros((1, n_assets + T)); Aeq[0, :n_assets] = 1
    constraints.append(LinearConstraint(Aeq, [1], [1]))
    Aret = np.zeros((1, n_assets + T)); Aret[0, :n_assets] = mu
    constraints.append(LinearConstraint(Aret, [M_target], [np.inf]))
    for t in range(T):
        A = np.zeros(n_assets + T)
        A[:n_assets] = return_matrix[t]
        A[n_assets + t] = 1
        constraints.append(LinearConstraint(A, [M_target], [np.inf]))
    result = minimize(objective, z0, method="SLSQP", bounds=bounds, constraints=constraints)
    if result.success:
        x = result.x[:n_assets]
        q = result.x[n_assets:]
        risk = np.sum(probabilities * q ** 2)
        ret  = np.sum(x * mu)
        return x, risk, ret, mu, True
    return None, None, None, mu, False

@st.cache_data
def compute_downside_curve(prices_tuple, n_points=25):
    prices_arr  = np.array(prices_tuple)
    returns_arr = calculate_returns(prices_arr)
    mu          = calculate_expected_returns(returns_arr)
    M_levels    = np.linspace(0, float(max(mu)), n_points)
    alloc_dict  = {i: [] for i in range(prices_arr.shape[1])}
    risks, feasible_M = [], []
    for M in M_levels:
        x, r, _, _, ok = solve_downside_portfolio(M, returns_arr)
        if ok:
            feasible_M.append(M)
            risks.append(float(np.sqrt(r)))
            for i in range(prices_arr.shape[1]):
                alloc_dict[i].append(float(x[i]))
    return feasible_M, risks, alloc_dict, mu.tolist()

# ══════════════════════════════════════════════════════════════════════
# MODEL PIECEWISE LINEAR (Exercise 18.3)
# ══════════════════════════════════════════════════════════════════════

def build_piecewise(qmax, segments):
    breaks = np.linspace(0, qmax, segments + 1)
    slopes, widths, errors = [], [], []
    for i in range(segments):
        qb, qe = breaks[i], breaks[i+1]
        slopes.append(qb + qe)
        widths.append(qe - qb)
        errors.append(((qe - qb) ** 2) / 4)
    return breaks, slopes, widths, errors

def dynamic_segments(q_previous, epsilon):
    seg = int(np.ceil(q_previous / (2 * np.sqrt(epsilon))))
    return max(seg, 3)

def solve_piecewise_portfolio(
        return_matrix, asset_names,
        target_return=0.20, min_fraction=0.05,
        epsilon=0.10, segments=10, use_dynamic=True,
        logic_asset1=None, logic_asset2=None):

    mu = calculate_expected_returns(return_matrix)
    T, n = return_matrix.shape[0], len(asset_names)
    probability = 1 / T
    qmax = np.max(np.abs(return_matrix))

    if use_dynamic:
        segments = dynamic_segments(qmax, epsilon)

    breaks, slopes, widths, errors = build_piecewise(qmax, segments)

    model = LpProblem("Exercise18_3", LpMinimize)

    x = {a: LpVariable(f"x_{a}", lowBound=0, upBound=1) for a in asset_names}
    z = {a: LpVariable(f"z_{a}", cat="Binary") for a in asset_names}
    trigger_RD = LpVariable("trigger_RD", cat="Binary")
    q = {t: LpVariable(f"q_{t}", lowBound=0) for t in range(T)}
    u = {(t, l): LpVariable(f"u_{t}_{l}", lowBound=0, upBound=widths[l])
         for t in range(T) for l in range(segments)}

    # Objective
    model += lpSum(probability * slopes[l] * u[t, l]
                   for t in range(T) for l in range(segments))

    # Budget
    model += lpSum(x[a] for a in asset_names) == 1

    # Target return
    model += lpSum(mu[i] * x[asset_names[i]] for i in range(n)) >= target_return

    # Minimum investment (0 OR >= min_fraction)
    for a in asset_names:
        model += x[a] <= z[a]
        model += x[a] >= min_fraction * z[a]

    # Piecewise decomposition
    for t in range(T):
        model += q[t] == lpSum(u[t, l] for l in range(segments))

    # Downside scenario constraints
    for t in range(T):
        model += (lpSum(return_matrix[t, i] * x[asset_names[i]] for i in range(n))
                  + q[t] >= target_return)

    # Logical constraint: IF asset1 > 20% THEN asset2 < 30%
    a1 = logic_asset1 if logic_asset1 and logic_asset1 in asset_names else asset_names[0]
    a2 = logic_asset2 if logic_asset2 and logic_asset2 in asset_names else asset_names[1]
    BIG_M = 1
    model += x[a1] - 0.20 <= BIG_M * trigger_RD
    model += x[a2] <= 0.30 + BIG_M * (1 - trigger_RD)

    model.solve(PULP_CBC_CMD(msg=False))

    allocation = {a: value(x[a]) for a in asset_names}
    selected   = {a: int(round(value(z[a]))) for a in asset_names}
    expected_return = np.sum([allocation[asset_names[i]] * mu[i] for i in range(n)])

    return {
        "status":          LpStatus[model.status],
        "allocation":      allocation,
        "selected":        selected,
        "expected_return": expected_return,
        "risk":            value(model.objective),
        "segments":        segments,
        "epsilon":         epsilon,
        "max_error":       max(errors),
        "errors":          errors,
        "breaks":          breaks,
        "slopes":          slopes,
        "qmax":            qmax,
    }

# ══════════════════════════════════════════════════════════════════════
# PIECEWISE LINEAR ILLUSTRATION CHART (Figure 18.6 style)
# ══════════════════════════════════════════════════════════════════════

def _pw_value(q, breaks, slopes):
    """Evaluate piecewise linear function at a single point q."""
    v = 0.0
    for i in range(len(slopes)):
        used = min(max(q - breaks[i], 0.0), breaks[i + 1] - breaks[i])
        v += slopes[i] * used
    return v

def plot_piecewise_illustration(qmax, breaks_actual=None, slopes_actual=None):
    """
    Plot Figure 18.6-style: kuadratik q² vs aproksimasi piecewise linear.
    Jika breaks/slopes dari data tersedia, gunakan segmen aktual.
    """
    if breaks_actual is not None and slopes_actual is not None:
        breaks_ill = np.array(breaks_actual)
        slopes_ill = list(slopes_actual)
    else:
        breaks_ill, slopes_ill, _, _ = build_piecewise(qmax, 3)

    x_dense = np.linspace(0, qmax, 400)
    quad_vals = x_dense ** 2
    pw_vals   = np.array([_pw_value(xx, breaks_ill, slopes_ill) for xx in x_dense])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    # Garis kuadratik dan piecewise
    ax.plot(x_dense, quad_vals, color="black", linewidth=2.2, label="Kuadratik $q^2$")
    ax.plot(x_dense, pw_vals,   color="black", linewidth=1.4, linestyle="-",
            label="Aproksimasi Piecewise Linear")

    # I-bar error di setiap breakpoint interior (persis Figure 18.6)
    tick_w = qmax * 0.03
    for bp in breaks_ill[1:-1]:
        pw_bp   = _pw_value(bp, breaks_ill, slopes_ill)
        quad_bp = bp ** 2
        # Titik perpotongan piecewise berada di ATAS kuadratik di tengah segmen
        # → tampilkan I-bar antara keduanya
        y_lo = min(pw_bp, quad_bp)
        y_hi = max(pw_bp, quad_bp)
        if (y_hi - y_lo) < 1e-9:
            continue
        ax.plot([bp, bp],                    [y_lo, y_hi], color="black", linewidth=1.6)
        ax.plot([bp - tick_w, bp + tick_w],  [y_hi, y_hi], color="black", linewidth=1.6)
        ax.plot([bp - tick_w, bp + tick_w],  [y_lo, y_lo], color="black", linewidth=1.6)

    # Garis putus-putus vertikal di breakpoints (seperti Figure 18.6)
    for bp in breaks_ill[1:-1]:
        ax.axvline(bp, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(qmax, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    ax.set_xlabel("q", fontsize=11)
    ax.set_ylabel("Nilai Aproksimasi", fontsize=11)
    ax.set_title("Piecewise Linear vs Kuadratik\n(Ilustrasi 3 Segmen)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════
PAGES = [
    "Dashboard",
    "Strategic Asset Allocation",
    "Tactical Asset Allocation",
    "Downside Variance Optimization",
    "Piecewise Linear (MILP)",
]

PAGE_DESC = {
    "Dashboard":                    "Penjelasan aplikasi & panduan",
    "Strategic Asset Allocation":   "Markowitz klasik — kovarians eksplisit",
    "Tactical Asset Allocation":    "Variance dari data return historis",
    "Downside Variance Optimization": "Semi-variance — hanya risiko di bawah M",
    "Piecewise Linear (MILP)":      "Aproksimasi MILP + logical constraint",
}

for k, v in {
    "page": "Dashboard",
    "s_cats": None, "s_rets": None, "s_cov": None, "s_error": None,
    "t_assets": None, "t_prices": None, "t_error": None,
    "d_assets": None, "d_prices": None, "d_error": None,
    "p_assets": None, "p_prices": None, "p_error": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
# NAVIGASI SIDEBAR (selalu tampil di semua halaman)
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## Navigasi")
    for page_name in PAGES:
        is_active = st.session_state["page"] == page_name
        btn_type  = "primary" if is_active else "secondary"
        if st.button(page_name, key=f"nav_{page_name}",
                     use_container_width=True, type=btn_type):
            st.session_state["page"] = page_name
            st.rerun()
    st.divider()

mode = st.session_state["page"]

# freq_key dibaca dari session_state agar persisten lintas halaman
if "global_freq" not in st.session_state:
    st.session_state["global_freq"] = "Mingguan (sesuai AIMMS)"
freq_key   = st.session_state["global_freq"]
freq_label = FREQ_OPTIONS[freq_key]["label"]

# ══════════════════════════════════════════════════════════════════════
# HEADER UTAMA
# ══════════════════════════════════════════════════════════════════════
st.title("Portfolio Optimization Suite")
st.divider()


# ══════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════
if mode == "Dashboard":

    # ── Hero section ──
    st.markdown("""
## Selamat Datang di Portfolio Optimization Suite

Aplikasi ini mengimplementasikan **4 model optimasi portofolio** berbasis riset operasi
dari **AIMMS Chapter 18: Portfolio Selection**. Data saham diambil secara otomatis dari
**Yahoo Finance** — tidak perlu input manual.
    """)

    col_hero1, col_hero2, col_hero3 = st.columns(3)
    col_hero1.info("**Data real-time**\nHarga saham dari Yahoo Finance, diresample ke return bulanan")
    col_hero2.info("**4 Model Optimasi**\nStrategic, Tactical, Downside Variance, Piecewise Linear")
    col_hero3.info("**Solver**\nSLSQP (continuous) + CBC (MILP) via PuLP")

    st.divider()

    # ── Tujuan & landasan ──
    st.markdown("## Tujuan & Landasan Model")
    st.markdown("""
Optimasi portofolio bertujuan menjawab pertanyaan:
> *"Bagaimana mengalokasikan modal ke sejumlah aset sehingga **risiko minimal** untuk target return tertentu?"*

Semua model dalam aplikasi ini berlandaskan kerangka **Mean-Variance Optimization** yang
diperkenalkan oleh Harry Markowitz (1952) dan dikembangkan dalam buku ajar AIMMS:

> *AIMMS — Optimization Modeling, Chapter 18: Portfolio Selection*

Setiap model memiliki asumsi dan pendekatan yang berbeda terhadap definisi "risiko":
    """)

    st.divider()

    # ── 4 model cards ──
    st.markdown("## Empat Model Optimasi")

    m1, m2 = st.columns(2, gap="large")

    with m1:
        st.markdown("""
### 1. Strategic Asset Allocation
**Landasan:** Model Markowitz klasik (Mean-Variance)

Meminimalkan **varians portofolio** $\\sigma^2 = \\mathbf{x}^T \\Sigma \\mathbf{x}$ untuk target return $M$:

$$\\min_{x} \\ \\mathbf{x}^T \\Sigma \\mathbf{x} \\quad \\text{s.t.} \\quad \\boldsymbol{\\mu}^T \\mathbf{x} \\geq M, \\ \\mathbf{1}^T \\mathbf{x} = 1$$

Matriks kovarians $\\Sigma$ dan expected return $\\boldsymbol{\\mu}$ **dihitung otomatis**
dari data historis bulanan Yahoo Finance.

**Output utama:** Efficient frontier (kurva Risk-Reward) & alokasi optimal per saham.
        """)
        st.markdown("""
### 3. Downside Variance Optimization
**Landasan:** Semi-variance / downside risk

Hanya menghitung deviasi di **bawah** target return $M$ (kerugian), bukan total variance:

$$\\min_{x,q} \\ \\sum_t p_t q_t^2 \\quad \\text{s.t.} \\quad r_t \\cdot x + q_t \\geq M, \\ q_t \\geq 0$$

Cocok untuk investor yang lebih sensitif terhadap kerugian daripada gain.
        """)

    with m2:
        st.markdown("""
### 2. Tactical Asset Allocation
**Landasan:** Mean-Variance dengan return historis langsung

Meminimalkan **variance aktual** berdasarkan realisasi return historis $r_t$:

$$\\min_{x} \\ \\sum_t p_t \\left( r_t \\cdot x - \\boldsymbol{\\mu} \\cdot x \\right)^2$$

Tidak memerlukan estimasi matriks kovarians secara eksplisit — kovarians
**diimplisitkan** dari data return historis.
        """)
        st.markdown("""
### 4. Piecewise Linear / MILP
**Landasan:** Aproksimasi piecewise linear dari fungsi kuadratik

Fungsi kuadratik $q^2$ diaproksimasi dengan segmen-segmen linear lurus,
lalu diselesaikan sebagai **MILP** (Mixed Integer Linear Program):

$$q^2 \\approx \\sum_{l=1}^{K} s_l \\cdot u_l, \\quad u_l \\in [0, w_l]$$

Jumlah segmen $K$ dihitung dinamis: $K = \\lceil q_{\\max} / 2\\sqrt{\\varepsilon} \\rceil$
        """)

    st.divider()

    # ── Cara pakai ──
    st.markdown("## Cara Menggunakan")
    st.markdown("""
1. Pilih salah satu **model** dari panel navigasi di **sidebar kiri**
2. Di sidebar, masukkan **ticker saham** (contoh: `BBCA.JK, BBRI.JK, TLKM.JK`)
3. Pilih **periode data** (rekomendasi: 2–3 tahun untuk data yang cukup)
4. Klik **Ambil Data dari Yahoo Finance**
5. Geser **slider Target Return** untuk menjelajahi efficient frontier
    """)

    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.markdown("""
**Contoh ticker saham Indonesia:**
| Saham | Ticker |
|---|---|
| Bank BCA | `BBCA.JK` |
| Bank BRI | `BBRI.JK` |
| Bank Mandiri | `BMRI.JK` |
| Telkom | `TLKM.JK` |
| Astra | `ASII.JK` |
| BNI | `BBNI.JK` |
        """)
    with col_ex2:
        st.markdown("""
**Contoh ticker saham US:**
| Saham | Ticker |
|---|---|
| Apple | `AAPL` |
| Microsoft | `MSFT` |
| Google | `GOOGL` |
| Amazon | `AMZN` |
| Meta | `META` |
| Tesla | `TSLA` |
        """)

    st.divider()
    st.caption("Portfolio Optimization Suite | Berdasarkan AIMMS Chapter 18: Portfolio Selection | Powered by Yahoo Finance & PuLP")


# ══════════════════════════════════════════════════════════════════════
# STRATEGIC
# ══════════════════════════════════════════════════════════════════════
elif mode.startswith("Strategic"):

    with st.sidebar:
        st.header("Input — Strategic")

        if not YFINANCE_AVAILABLE:
            st.error("yfinance belum terinstall. Jalankan: `pip install yfinance`")
        else:
            tickers_input = ticker_selector("s", "BBCA.JK, BBRI.JK, TLKM.JK")
            st.divider()
            period_label = st.selectbox("Periode Data", options=list(PERIOD_OPTIONS.keys()), index=2)
            st.selectbox("Frekuensi Data", options=list(FREQ_OPTIONS.keys()), key="global_freq",
                         help="Mingguan = sesuai AIMMS Ch.18")
            fetch_btn = st.button("Ambil Data dari Yahoo Finance", use_container_width=True, type="primary")

    if not YFINANCE_AVAILABLE:
        st.error("yfinance tidak tersedia. Install dengan: `pip install yfinance`")
        st.stop()

    if fetch_btn or st.session_state.s_cats is None:
        if fetch_btn or st.session_state.s_cats is None:
            period_code = PERIOD_OPTIONS[period_label]
            with st.spinner(f"Mengambil data {freq_key} dari Yahoo Finance ({period_label})..."):
                tickers_result, mu_yf, cov_yf, err = fetch_yfinance(tickers_input, period_code, freq_key)
            if err:
                st.session_state.s_error = f"❌ {err}"
            else:
                st.session_state.s_cats  = tickers_result
                st.session_state.s_rets  = mu_yf
                st.session_state.s_cov   = cov_yf
                st.session_state.s_error = None
                if fetch_btn:
                    st.toast(f"Data berhasil diambil: {', '.join(tickers_result)}", icon="✅")

    if st.session_state.s_error:
        st.error(st.session_state.s_error)
        st.info("Pastikan nama ticker benar dan koneksi internet tersedia.")
        st.stop()

    if st.session_state.s_cats is None:
        st.info("Klik **Ambil Data dari Yahoo Finance** di sidebar untuk memulai.")
        st.stop()

    cats = st.session_state.s_cats
    rets = st.session_state.s_rets
    cov  = st.session_state.s_cov

    # Info data yang digunakan
    with st.expander("Data yang digunakan", expanded=False):
        col_ret, col_cov = st.columns(2)
        with col_ret:
            st.markdown("**Expected Return per periode (%):**")
            df_ret_info = pd.DataFrame({
                "Saham": cats,
                "Expected Return (% per periode)": [f"{r:.4f}%" for r in rets],
            })
            st.dataframe(df_ret_info, use_container_width=True, hide_index=True)
        with col_cov:
            st.markdown("**Matriks Kovarians:**")
            st.dataframe(
                pd.DataFrame(cov, columns=cats, index=cats).map(lambda x: f"{x:.4f}"),
                use_container_width=True,
            )

    M_min_val = float(min(rets))
    M_max_val = float(max(rets))
    M_target  = st.slider(
        "Target Return (M)",
        min_value=M_min_val, max_value=M_max_val,
        value=round((M_min_val + M_max_val) / 2, 4),
        step=round((M_max_val - M_min_val) / 100, 5),
        format="%.4f",
    )
    st.divider()

    x_opt, risk_opt_var, return_opt, success = solve_strategic_portfolio(M_target, rets, cov)
    if not success:
        st.warning(f"Tidak ditemukan solusi feasible untuk M = {M_target:.4f}%.")
        st.stop()
    risk_opt_std = np.sqrt(risk_opt_var)

    feasible_M_levels, risks_std_curve, alloc_curve = compute_strategic_curve(
        tuple(rets.tolist()), tuple(map(tuple, cov.tolist()))
    )

    M_specific_raw = np.linspace(M_min_val, M_max_val, 6)
    M_specific     = np.round(M_specific_raw, 4)
    specific_risks = []
    for M_s in M_specific:
        _, rv_s, _, ok_s = solve_strategic_portfolio(float(M_s), rets, cov)
        specific_risks.append(float(np.sqrt(rv_s)) if ok_s else None)

    st.subheader(f"Hasil Optimasi — Target M = {M_target:.2f}% / bulan")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Target Return / Periode",      f"{M_target:.2f}%")
    mc2.metric("Expected Return Aktual / Periode", f"{return_opt:.2f}%")
    mc3.metric("Risiko (Varians)",           f"{risk_opt_var:.4f}")
    mc4.metric("Standar Deviasi / Periode",    f"{risk_opt_std:.2f}%")
    st.divider()

    col_g1, col_g2 = st.columns(2, gap="medium")

    with col_g1:
        st.markdown("#### Asset Allocation vs Expected Return")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        alloc_arr = np.array(alloc_curve)
        for i, cat in enumerate(cats):
            ax1.plot(feasible_M_levels, alloc_arr[:, i],
                     marker="o", linewidth=2.5, markersize=4,
                     label=cat, color=COLORS[i % len(COLORS)])
        ax1.axvline(M_target, color="red", linestyle="--", alpha=0.75,
                    linewidth=2, label=f"Target Return: {M_target:.4f}%")
        ax1.set_xlabel("Minimal level of expected return (%)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Budget fractions",                     fontsize=11, fontweight="bold")
        ax1.set_title("Asset Allocation vs Expected Return", fontsize=13, fontweight="bold")
        ax1.legend(loc="center right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.set_xlim(M_min_val - abs(M_min_val)*0.05, M_max_val + abs(M_max_val)*0.05)
        ax1.set_ylim(-0.05, 1.05)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    with col_g2:
        st.markdown("#### Risk-Reward Characteristic")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(feasible_M_levels, risks_std_curve, linewidth=2.5, color=C_LINE,
                 label="Risk-Reward Characteristic")
        y_fill_min = max(0.0, min(risks_std_curve) * 0.8)
        ax2.fill_between(feasible_M_levels, risks_std_curve, y_fill_min, alpha=0.08, color=C_FILL)
        for idx, M_val in enumerate(M_specific):
            if specific_risks[idx] is not None:
                ax2.plot(M_val, specific_risks[idx], "o", markersize=8,
                         color=C_DOT, markeredgecolor="white", markeredgewidth=1.2, zorder=4)
                ax2.annotate(f"{specific_risks[idx]:.3f}%", xy=(M_val, specific_risks[idx]),
                             xytext=(8, 5), textcoords="offset points",
                             fontsize=8.5, fontweight="bold", color=C_DOT)
        ax2.plot(M_target, risk_opt_std, "o", markersize=14, color=C_GOLD,
                 markeredgecolor="white", markeredgewidth=1.8,
                 label=f"Selected Portfolio (M = {M_target:.2f}%)", zorder=5)
        ax2.set_xlabel("Minimal level of expected return (%)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Portfolio risk — Std Dev (%)",         fontsize=11, fontweight="bold")
        ax2.set_title("Risk-Reward Characteristic", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.3)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.divider()

    col_tbl, col_rec = st.columns([3, 2], gap="large")

    with col_tbl:
        st.markdown("#### Tabel Alokasi pada Berbagai Target Return")
        key_returns = sorted(set(np.round(
            [M_min_val, (M_min_val + M_max_val) / 2, M_max_val, M_target], 4
        ).tolist()))
        tbl_rows = []
        for M_k in key_returns:
            x_k, rv_k, ret_k, ok_k = solve_strategic_portfolio(float(M_k), rets, cov)
            if ok_k:
                row = {"Target Return (%)": f"{M_k:.4f}%"}
                for i, cat in enumerate(cats):
                    row[cat] = f"{x_k[i]*100:.1f}%"
                row["Risk (Std Dev %)"] = f"{np.sqrt(rv_k):.4f}%"
                row["Exp. Return (%)"]  = f"{ret_k:.4f}%"
                tbl_rows.append(row)
        st.dataframe(tbl_rows, use_container_width=True, hide_index=True)
        st.info(
            f"M = {M_target:.4f}% → "
            + " | ".join([f"{cats[i]}: **{x_opt[i]*100:.1f}%**" for i in range(len(cats))])
            + f" | Std Dev: **{risk_opt_std:.4f}%**"
        )

    with col_rec:
        st.markdown(f"#### Rekomendasi untuk M = {M_target:.4f}%")
        fig_bar, ax_bar = plt.subplots(figsize=(5, max(3, len(cats) * 1.1)))
        bars = ax_bar.barh(cats, x_opt * 100,
                           color=[COLORS[i % len(COLORS)] for i in range(len(cats))],
                           edgecolor="white", height=0.55)
        for bar_obj, val in zip(bars, x_opt * 100):
            ax_bar.text(bar_obj.get_width() + 0.5,
                        bar_obj.get_y() + bar_obj.get_height() / 2,
                        f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax_bar.set_xlim(0, 115)
        ax_bar.set_xlabel("Alokasi (%)", fontsize=10)
        ax_bar.set_title(f"Komposisi Portofolio (M = {M_target:.4f}%)", fontsize=11, fontweight="bold")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
        ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)
        fig_bar.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)

        st.markdown("**Analisis Risk-Reward:**")
        r_low, r_high = min(risks_std_curve), max(risks_std_curve)
        delta_pct = (r_high - r_low) / r_low * 100
        st.markdown(
            f"- Return terendah ({M_min_val:.4f}%) → risiko = **{r_low:.4f}%**\n"
            f"- Return tertinggi ({M_max_val:.4f}%) → risiko = **{r_high:.4f}%**\n"
            f"- Risk naik sebesar **{delta_pct:.1f}%** dari min ke max return"
        )

    st.divider()
    st.caption("Model: Strategic — Mean-Variance Optimization (SLSQP) | AIMMS Chapter 18")


# ══════════════════════════════════════════════════════════════════════
# TACTICAL
# ══════════════════════════════════════════════════════════════════════
elif mode.startswith("Tactical"):

    with st.sidebar:
        st.header("Input — Tactical")
        tickers_input_t = ticker_selector("t", "BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, BMRI.JK")
        st.divider()
        period_label_t = st.selectbox("Periode Data", options=list(PERIOD_OPTIONS.keys()), index=2, key="t_period")
        st.selectbox("Frekuensi Data", options=list(FREQ_OPTIONS.keys()), key="global_freq",
                     help="Mingguan = sesuai AIMMS Ch.18 (~52 titik/tahun)")
        fetch_btn_t = st.button("Ambil Data dari Yahoo Finance", use_container_width=True, type="primary", key="t_fetch")

    if fetch_btn_t or st.session_state.t_assets is None:
        with st.spinner(f"Mengambil data {freq_key} dari Yahoo Finance..."):
            tickers_t, prices_t_arr, err_t = fetch_prices(tickers_input_t, PERIOD_OPTIONS[period_label_t], freq_key)
        if err_t:
            st.session_state.t_error = f"❌ {err_t}"
        else:
            st.session_state.t_assets = tickers_t
            st.session_state.t_prices = prices_t_arr
            st.session_state.t_error  = None
            if fetch_btn_t:
                st.toast(f"Data berhasil diambil: {', '.join(tickers_t)}", icon="✅")

    if st.session_state.t_error:
        st.error(st.session_state.t_error)
        st.info("Pastikan nama ticker benar dan koneksi internet tersedia.")
        st.stop()

    if st.session_state.t_assets is None:
        st.info("Klik **Ambil Data dari Yahoo Finance** di sidebar untuk memulai.")
        st.stop()

    assets = st.session_state.t_assets
    prices = st.session_state.t_prices
    returns = calculate_returns(prices)
    mu      = calculate_expected_returns(returns)

    M_slider = st.slider(
        "Target Return per Periode (M)",
        min_value=0.0, max_value=float(max(mu)),
        value=round(float(max(mu)) * 0.4, 3),
        step=round(float(max(mu)) / 50, 3),
        format="%.2f%%",
    )
    st.divider()

    x_opt_t, risk_t, ret_t, mu_t, success_t = solve_tactical_portfolio(M_slider, returns)
    if not success_t:
        st.warning(f"Tidak ditemukan solusi feasible untuk M = {M_slider:.2f}%.")
        st.stop()
    risk_std_t = np.sqrt(risk_t)

    st.subheader(f"Hasil Optimasi — Target M = {M_slider:.2f}% / bulan")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Target Return / Periode",      f"{M_slider:.2f}%")
    mc2.metric("Expected Return Aktual / Periode", f"{ret_t:.2f}%")
    mc3.metric("Risiko (Varians)",           f"{risk_t:.4f}")
    mc4.metric("Standar Deviasi / Periode",    f"{risk_std_t:.2f}%")
    st.divider()

    with st.expander("Expected Return & Alokasi Optimal per Aset"):
        df_mu = pd.DataFrame({
            "Aset":            assets,
            "Expected Return": [f"{v:.4f}%" for v in mu_t],
            "Alokasi Optimal": [f"{x_opt_t[i]*100:.2f}%" for i in range(len(assets))],
        })
        st.dataframe(df_mu, use_container_width=True, hide_index=True)

    feasible_M_t, risks_std_t, alloc_dict_t, mu_list = compute_tactical_curve(
        tuple(map(tuple, prices.tolist()))
    )

    col_g1t, col_g2t = st.columns(2, gap="medium")

    with col_g1t:
        st.markdown("#### Risk-Reward Characteristic")
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.plot(feasible_M_t, risks_std_t, linewidth=2.5, color=C_LINE,
                 label="Risk-Reward Characteristic")
        y_fill_min_t = max(0.0, min(risks_std_t) * 0.8)
        ax4.fill_between(feasible_M_t, risks_std_t, y_fill_min_t, alpha=0.08, color=C_FILL)
        ax4.plot(M_slider, risk_std_t, "o", markersize=14, color=C_GOLD,
                 markeredgecolor="white", markeredgewidth=1.8,
                 zorder=5, label=f"Selected (M={M_slider:.2f}%)")
        ax4.set_xlabel("Minimal Expected Return (%)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Portfolio Risk — Std Dev (%)", fontsize=11, fontweight="bold")
        ax4.set_title("Risk-Reward Characteristic", fontsize=13, fontweight="bold")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
        fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    with col_g2t:
        st.markdown("#### Portfolio Diversification")
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        for i, a in enumerate(assets):
            ax5.plot(feasible_M_t, alloc_dict_t[i],
                     linewidth=2.5, marker="o", markersize=4,
                     label=a, color=COLORS[i % len(COLORS)])
        ax5.axvline(M_slider, color="red", linestyle="--", alpha=0.75,
                    linewidth=2, label=f"Target Return: {M_slider:.2f}%")
        ax5.set_xlabel("Minimal Expected Return (%)", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Budget Fraction",              fontsize=11, fontweight="bold")
        ax5.set_title("Portfolio Diversification", fontsize=13, fontweight="bold")
        ax5.legend(loc="center right", fontsize=9)
        ax5.grid(True, linestyle="--", alpha=0.3)
        ax5.set_ylim(-0.05, 1.05)
        ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
        fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)

    st.divider()

    col_tbl_t, col_rec_t = st.columns([3, 2], gap="large")

    with col_tbl_t:
        st.markdown("#### Tabel Alokasi Berbagai Target Return")
        sample_M_t = np.linspace(0, float(max(mu_t)), 5)
        tbl_rows_t = []
        for M_k in sample_M_t:
            x_k, r_k, rr_k, _, ok_k = solve_tactical_portfolio(float(M_k), returns)
            if ok_k:
                row = {"Target Return (%)": f"{M_k:.4f}%", "Risk (Var)": f"{r_k:.4f}"}
                for i, a in enumerate(assets):
                    row[a] = f"{x_k[i]*100:.1f}%"
                tbl_rows_t.append(row)
        st.dataframe(tbl_rows_t, use_container_width=True, hide_index=True)
        st.info(
            f"M = {M_slider:.2f}% → "
            + " | ".join([f"{assets[i]}: **{x_opt_t[i]*100:.1f}%**" for i in range(len(assets))])
            + f" | Std Dev: **{risk_std_t:.4f}%**"
        )
        st.markdown("#### Scenario Returns (10 Baris Pertama)")
        df_ret = pd.DataFrame(returns, columns=assets)
        st.dataframe(df_ret.head(10).map(lambda x: f"{x:.4f}%"), use_container_width=True)

    with col_rec_t:
        st.markdown(f"#### Rekomendasi untuk M = {M_slider:.2f}%")
        fig_bar_t, ax_bar_t = plt.subplots(figsize=(5, max(3, len(assets) * 1.0)))
        bars_t = ax_bar_t.barh(
            assets, x_opt_t * 100,
            color=[COLORS[i % len(COLORS)] for i in range(len(assets))],
            edgecolor="white", height=0.55,
        )
        for bar_obj, val in zip(bars_t, x_opt_t * 100):
            ax_bar_t.text(bar_obj.get_width() + 0.5,
                          bar_obj.get_y() + bar_obj.get_height() / 2,
                          f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax_bar_t.set_xlim(0, 115)
        ax_bar_t.set_xlabel("Alokasi (%)", fontsize=10)
        ax_bar_t.set_title(f"Komposisi Portofolio (M = {M_slider:.2f}%)", fontsize=11, fontweight="bold")
        ax_bar_t.grid(axis="x", linestyle="--", alpha=0.4)
        ax_bar_t.spines["top"].set_visible(False); ax_bar_t.spines["right"].set_visible(False)
        fig_bar_t.tight_layout(); st.pyplot(fig_bar_t); plt.close(fig_bar_t)

        st.markdown("**Analisis Risk-Reward:**")
        r_low_t, r_high_t = min(risks_std_t), max(risks_std_t)
        delta_pct_t = (r_high_t - r_low_t) / r_low_t * 100
        st.markdown(
            f"- Return terendah (0.00%) → risiko = **{r_low_t:.4f}%**\n"
            f"- Return tertinggi ({max(mu_t):.2f}%) → risiko = **{r_high_t:.4f}%**\n"
            f"- Risk naik sebesar **{delta_pct_t:.1f}%** dari min ke max return"
        )

    st.divider()
    st.caption("Model: Tactical — Mean-Variance Optimization (SLSQP) | AIMMS Chapter 18")


# ══════════════════════════════════════════════════════════════════════
# DOWNSIDE VARIANCE
# ══════════════════════════════════════════════════════════════════════
elif mode.startswith("Downside"):

    with st.sidebar:
        st.header("Input — Downside Variance")
        tickers_input_d = ticker_selector("d", "BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, BMRI.JK")
        st.divider()
        period_label_d = st.selectbox("Periode Data", options=list(PERIOD_OPTIONS.keys()), index=2, key="d_period")
        st.selectbox("Frekuensi Data", options=list(FREQ_OPTIONS.keys()), key="global_freq",
                     help="Mingguan = sesuai AIMMS Ch.18 (~52 titik/tahun)")
        fetch_btn_d = st.button("Ambil Data dari Yahoo Finance", use_container_width=True, type="primary", key="d_fetch")
        st.divider()
        st.info("**Downside Variance** hanya meminimalkan deviasi di bawah target M, bukan total variance.")

    if fetch_btn_d or st.session_state.d_assets is None:
        with st.spinner(f"Mengambil data {freq_key} dari Yahoo Finance..."):
            tickers_d, prices_d_arr, err_d = fetch_prices(tickers_input_d, PERIOD_OPTIONS[period_label_d], freq_key)
        if err_d:
            st.session_state.d_error = f"❌ {err_d}"
        else:
            st.session_state.d_assets = tickers_d
            st.session_state.d_prices = prices_d_arr
            st.session_state.d_error  = None
            if fetch_btn_d:
                st.toast(f"Data berhasil diambil: {', '.join(tickers_d)}", icon="✅")

    if st.session_state.d_error:
        st.error(st.session_state.d_error)
        st.info("Pastikan nama ticker benar dan koneksi internet tersedia.")
        st.stop()

    if st.session_state.d_assets is None:
        st.info("Klik **Ambil Data dari Yahoo Finance** di sidebar untuk memulai.")
        st.stop()

    assets_d  = st.session_state.d_assets
    prices_d  = st.session_state.d_prices
    returns_d = calculate_returns(prices_d)
    mu_d      = calculate_expected_returns(returns_d)

    M_slider_d = st.slider(
        "Target Return per Periode (M)",
        min_value=0.0, max_value=float(max(mu_d)),
        value=round(float(max(mu_d)) * 0.4, 3),
        step=round(float(max(mu_d)) / 50, 3),
        format="%.2f%%", key="d_slider",
    )
    st.divider()

    with st.spinner("Menghitung optimasi downside..."):
        x_opt_d, risk_d, ret_d, mu_d_arr, success_d = solve_downside_portfolio(M_slider_d, returns_d)

    if not success_d:
        st.warning(f"Tidak ditemukan solusi feasible untuk M = {M_slider_d:.2f}%.")
        st.stop()
    risk_std_d = np.sqrt(risk_d)

    st.subheader(f"Hasil Optimasi — Target M = {M_slider_d:.2f}% / bulan")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Target Return / Periode",        f"{M_slider_d:.2f}%")
    mc2.metric("Expected Return Aktual / Periode", f"{ret_d:.2f}%")
    mc3.metric("Downside Risk (Semi-Var)",     f"{risk_d:.4f}")
    mc4.metric("Downside Std Dev / Periode",     f"{risk_std_d:.2f}%")
    st.divider()

    with st.expander("Expected Return & Alokasi Optimal per Aset"):
        df_mu_d = pd.DataFrame({
            "Aset":            assets_d,
            "Expected Return": [f"{v:.4f}%" for v in mu_d_arr],
            "Alokasi Optimal": [f"{x_opt_d[i]*100:.2f}%" for i in range(len(assets_d))],
        })
        st.dataframe(df_mu_d, use_container_width=True, hide_index=True)

    with st.spinner("Menghitung kurva downside..."):
        feasible_M_d, risks_std_d, alloc_dict_d, mu_list_d = compute_downside_curve(
            tuple(map(tuple, prices_d.tolist()))
        )

    col_g1d, col_g2d = st.columns(2, gap="medium")

    with col_g1d:
        st.markdown("#### Downside Risk-Reward Characteristic")
        fig_d1, ax_d1 = plt.subplots(figsize=(7, 5))
        ax_d1.plot(feasible_M_d, risks_std_d, linewidth=2.5, color=C_DOWN,
                   label="Downside Risk-Reward")
        y_fill_min_d = max(0.0, min(risks_std_d) * 0.8)
        ax_d1.fill_between(feasible_M_d, risks_std_d, y_fill_min_d, alpha=0.08, color=C_DOWN)
        ax_d1.plot(M_slider_d, risk_std_d, "o", markersize=14, color=C_GOLD,
                   markeredgecolor="white", markeredgewidth=1.8,
                   zorder=5, label=f"Selected (M={M_slider_d:.2f}%)")
        ax_d1.set_xlabel("Target Return M (%)", fontsize=11, fontweight="bold")
        ax_d1.set_ylabel("Downside Risk — Std Dev (%)", fontsize=11, fontweight="bold")
        ax_d1.set_title("Downside Risk-Reward Characteristic", fontsize=13, fontweight="bold")
        ax_d1.legend(loc="upper left", fontsize=9)
        ax_d1.grid(True, linestyle="--", alpha=0.4)
        ax_d1.spines["top"].set_visible(False); ax_d1.spines["right"].set_visible(False)
        fig_d1.tight_layout(); st.pyplot(fig_d1); plt.close(fig_d1)

    with col_g2d:
        st.markdown("#### Portfolio Diversification (Downside)")
        fig_d2, ax_d2 = plt.subplots(figsize=(7, 5))
        for i, a in enumerate(assets_d):
            ax_d2.plot(feasible_M_d, alloc_dict_d[i],
                       linewidth=2.5, marker="o", markersize=4,
                       label=a, color=COLORS[i % len(COLORS)])
        ax_d2.axvline(M_slider_d, color="red", linestyle="--", alpha=0.75,
                      linewidth=2, label=f"Target Return: {M_slider_d:.2f}%")
        ax_d2.set_xlabel("Target Return M (%)", fontsize=11, fontweight="bold")
        ax_d2.set_ylabel("Budget Fraction",      fontsize=11, fontweight="bold")
        ax_d2.set_title("Portfolio Diversification (Downside)", fontsize=13, fontweight="bold")
        ax_d2.legend(loc="center right", fontsize=9)
        ax_d2.grid(True, linestyle="--", alpha=0.3)
        ax_d2.set_ylim(-0.05, 1.05)
        ax_d2.spines["top"].set_visible(False); ax_d2.spines["right"].set_visible(False)
        fig_d2.tight_layout(); st.pyplot(fig_d2); plt.close(fig_d2)

    st.divider()

    col_tbl_d, col_rec_d = st.columns([3, 2], gap="large")

    with col_tbl_d:
        st.markdown("#### Tabel Alokasi Berbagai Target Return")
        sample_M_d = np.linspace(0, float(max(mu_list_d)), 5)
        tbl_rows_d = []
        for M_k in sample_M_d:
            x_k, r_k, rr_k, _, ok_k = solve_downside_portfolio(float(M_k), returns_d)
            if ok_k:
                row = {"Target Return (%)": f"{M_k:.4f}%", "Downside Risk": f"{r_k:.4f}"}
                for i, a in enumerate(assets_d):
                    row[a] = f"{x_k[i]*100:.1f}%"
                tbl_rows_d.append(row)
        st.dataframe(tbl_rows_d, use_container_width=True, hide_index=True)
        st.info(
            f"M = {M_slider_d:.2f}% → "
            + " | ".join([f"{assets_d[i]}: **{x_opt_d[i]*100:.1f}%**" for i in range(len(assets_d))])
            + f" | Downside Std Dev: **{risk_std_d:.4f}%**"
        )
        st.markdown("#### Scenario Returns (10 Baris Pertama)")
        df_ret_d = pd.DataFrame(returns_d, columns=assets_d)
        st.dataframe(df_ret_d.head(10).map(lambda x: f"{x:.4f}%"), use_container_width=True)

    with col_rec_d:
        st.markdown(f"#### Rekomendasi untuk M = {M_slider_d:.2f}%")
        fig_bar_d, ax_bar_d = plt.subplots(figsize=(5, max(3, len(assets_d) * 1.0)))
        bars_d = ax_bar_d.barh(
            assets_d, x_opt_d * 100,
            color=[COLORS[i % len(COLORS)] for i in range(len(assets_d))],
            edgecolor="white", height=0.55,
        )
        for bar_obj, val in zip(bars_d, x_opt_d * 100):
            ax_bar_d.text(bar_obj.get_width() + 0.5,
                          bar_obj.get_y() + bar_obj.get_height() / 2,
                          f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
        ax_bar_d.set_xlim(0, 115)
        ax_bar_d.set_xlabel("Alokasi (%)", fontsize=10)
        ax_bar_d.set_title(f"Komposisi Portofolio (M = {M_slider_d:.2f}%)", fontsize=11, fontweight="bold")
        ax_bar_d.grid(axis="x", linestyle="--", alpha=0.4)
        ax_bar_d.spines["top"].set_visible(False); ax_bar_d.spines["right"].set_visible(False)
        fig_bar_d.tight_layout(); st.pyplot(fig_bar_d); plt.close(fig_bar_d)

        st.markdown("**Analisis Risk-Reward:**")
        r_low_d, r_high_d = min(risks_std_d), max(risks_std_d)
        delta_pct_d = (r_high_d - r_low_d) / r_low_d * 100
        st.markdown(
            f"- Return terendah (0.00%) → downside risk = **{r_low_d:.4f}%**\n"
            f"- Return tertinggi ({max(mu_list_d):.2f}%) → downside risk = **{r_high_d:.4f}%**\n"
            f"- Risk naik sebesar **{delta_pct_d:.1f}%** dari min ke max return"
        )

    st.divider()
    st.caption("Model: Downside Variance Optimization (SLSQP) | AIMMS Chapter 18")


# ══════════════════════════════════════════════════════════════════════
# PIECEWISE LINEAR (MILP)
# ══════════════════════════════════════════════════════════════════════
else:  # Piecewise

    with st.sidebar:
        st.header("Input — Piecewise Linear")
        tickers_input_p = ticker_selector("p", "BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, BMRI.JK")
        st.divider()
        period_label_p = st.selectbox("Periode Data", options=list(PERIOD_OPTIONS.keys()), index=2, key="p_period")
        st.selectbox("Frekuensi Data", options=list(FREQ_OPTIONS.keys()), key="global_freq",
                     help="Mingguan = sesuai AIMMS Ch.18 (~52 titik/tahun)")
        fetch_btn_p = st.button("Ambil Data dari Yahoo Finance", use_container_width=True, type="primary", key="p_fetch")
        st.divider()
        st.markdown("**Parameter Optimasi:**")
        p_target   = st.slider("Target Return (M)",     0.00, 0.55, 0.20, 0.01, key="p_target",
                                format="%.2f%%")
        p_minfrac  = st.slider("Min. Investment per Aset (%)", 1, 20, 5, 1, key="p_minfrac",
                                help="0 ATAU >= nilai ini (constraint binary)")
        p_epsilon  = st.slider("Epsilon (ε)",           0.01, 1.00, 0.10, 0.01, key="p_epsilon",
                                help="Toleransi error aproksimasi piecewise")
        p_segments = st.slider("Jumlah Interval (manual)", 2, 50, 10, 1, key="p_segments",
                                help="Hanya aktif jika Dynamic Interval dinonaktifkan")
        p_dynamic  = st.checkbox("Dynamic Interval", value=True, key="p_dynamic",
                                  help="Hitung K otomatis: ceil(q_max / 2√ε)")
        st.divider()
        st.divider()
        st.markdown(
            "**Catatan Piecewise:**\n\n"
            "Logical constraint pada model ini menggunakan **2 aset pertama** dari daftar ticker\n\n"
            "Jika aset ke-1 > 20%, maka aset ke-2 < 30%"
        )

    if fetch_btn_p or st.session_state.p_assets is None:
        with st.spinner(f"Mengambil data {freq_key} dari Yahoo Finance..."):
            tickers_p, prices_p_arr, err_p = fetch_prices(tickers_input_p, PERIOD_OPTIONS[period_label_p], freq_key)
        if err_p:
            st.session_state.p_error = f"❌ {err_p}"
        elif len(tickers_p) < 2:
            st.session_state.p_error = "❌ Minimal 2 ticker untuk model Piecewise."
        else:
            st.session_state.p_assets = tickers_p
            st.session_state.p_prices = prices_p_arr
            st.session_state.p_error  = None
            if fetch_btn_p:
                st.toast(f"Data berhasil diambil: {', '.join(tickers_p)}", icon="✅")

    if st.session_state.p_error:
        st.error(st.session_state.p_error)
        st.info("Pastikan nama ticker benar dan koneksi internet tersedia.")
        st.stop()

    if st.session_state.p_assets is None:
        st.info("Klik **Ambil Data dari Yahoo Finance** di sidebar untuk memulai.")
        st.stop()

    assets_p  = st.session_state.p_assets
    prices_p  = st.session_state.p_prices
    returns_p = calculate_returns(prices_p)
    p_minfrac_val = p_minfrac / 100.0
    st.divider()

    logic_a1 = assets_p[0]
    logic_a2 = assets_p[1]

    with st.spinner("Menyelesaikan MILP... (mungkin 5–15 detik)"):
        result_p = solve_piecewise_portfolio(
            return_matrix=returns_p,
            asset_names=assets_p,
            target_return=p_target,
            min_fraction=p_minfrac_val,
            epsilon=p_epsilon,
            segments=p_segments,
            use_dynamic=p_dynamic,
            logic_asset1=logic_a1,
            logic_asset2=logic_a2,
        )

    # STATUS
    status_color = "Optimal" if result_p["status"] == "Optimal" else "Tidak Optimal"
    st.subheader(f"Hasil Optimasi MILP — {status_color}: {result_p['status']}")

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Target Return (M)",     f"{p_target:.2f}%")
    mc2.metric("Expected Return",       f"{result_p['expected_return']:.4f}%")
    mc3.metric("Piecewise Risk",        f"{result_p['risk']:.4f}")
    mc4.metric("Interval Digunakan",    f"{result_p['segments']}")
    mc5.metric("Max Approx Error",      f"{result_p['max_error']:.6f}")
    st.divider()

    # TABEL ALOKASI
    col_tbl_p, col_pie_p = st.columns([3, 2], gap="large")

    with col_tbl_p:
        st.markdown("#### Tabel Alokasi & Aset Terpilih")
        alloc_df = pd.DataFrame({
            "Aset":           assets_p,
            "Dipilih (0/1)":  [result_p["selected"][a] for a in assets_p],
            "Alokasi (%)":    [f"{(result_p['allocation'][a] or 0)*100:.2f}%" for a in assets_p],
        })
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        # Logical constraint check
        a1_alloc = result_p["allocation"].get(logic_a1, 0) or 0
        a2_alloc = result_p["allocation"].get(logic_a2, 0) or 0
        st.markdown(f"#### Cek Logical Constraint ({logic_a1} > 20% → {logic_a2} < 30%)")
        if a1_alloc > 0.20:
            if a2_alloc < 0.30:
                st.success(f"{logic_a1} = {a1_alloc*100:.1f}% > 20% dan {logic_a2} = {a2_alloc*100:.1f}% < 30% — Constraint terpenuhi")
            else:
                st.error(f"{logic_a1} = {a1_alloc*100:.1f}% > 20% tetapi {logic_a2} = {a2_alloc*100:.1f}% >= 30% — Dilanggar!")
        else:
            st.info(f"{logic_a1} = {a1_alloc*100:.1f}% <= 20% — Logical constraint tidak aktif")

        st.markdown("#### Dynamic Interval Report")
        st.markdown(
            f"- ε = **{result_p['epsilon']}** | "
            f"q_max = **{result_p['qmax']:.4f}** | "
            f"K = **{result_p['segments']}** interval"
        )

    with col_pie_p:
        st.markdown("#### Komposisi Portofolio")
        alloc_vals = [result_p["allocation"][a] or 0 for a in assets_p]
        nonzero_labels = [assets_p[i] for i, v in enumerate(alloc_vals) if v > 0.001]
        nonzero_vals   = [v for v in alloc_vals if v > 0.001]
        if nonzero_vals:
            fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
            ax_pie.pie(nonzero_vals, labels=nonzero_labels, autopct="%1.1f%%",
                       colors=[COLORS[i % len(COLORS)] for i in range(len(nonzero_vals))],
                       startangle=90)
            ax_pie.set_title("Optimal Portfolio Allocation", fontsize=12, fontweight="bold")
            fig_pie.tight_layout(); st.pyplot(fig_pie); plt.close(fig_pie)

    st.divider()

    # 3 GRAFIK BAWAH
    col_sel, col_err, col_pw = st.columns(3, gap="medium")

    with col_sel:
        st.markdown("#### Selected Assets")
        fig_sel, ax_sel = plt.subplots(figsize=(5, 4))
        sel_vals = [result_p["selected"][a] for a in assets_p]
        ax_sel.bar(assets_p, sel_vals,
                   color=[COLORS[i % len(COLORS)] for i in range(len(assets_p))],
                   edgecolor="white")
        ax_sel.set_ylabel("Dipilih (0 / 1)", fontsize=10)
        ax_sel.set_title("Selected Assets (Binary)", fontsize=11, fontweight="bold")
        ax_sel.set_ylim(0, 1.4)
        ax_sel.grid(axis="y", linestyle="--", alpha=0.4)
        ax_sel.spines["top"].set_visible(False); ax_sel.spines["right"].set_visible(False)
        fig_sel.tight_layout(); st.pyplot(fig_sel); plt.close(fig_sel)

    with col_err:
        st.markdown("#### Approximation Error per Interval")
        fig_err, ax_err = plt.subplots(figsize=(5, 4))
        ax_err.bar(range(len(result_p["errors"])), result_p["errors"],
                   color="#9467bd", edgecolor="white")
        ax_err.set_xlabel("Interval ke-", fontsize=10)
        ax_err.set_ylabel("Max Error", fontsize=10)
        ax_err.set_title("Piecewise Approximation Error", fontsize=11, fontweight="bold")
        ax_err.grid(axis="y", linestyle="--", alpha=0.4)
        ax_err.spines["top"].set_visible(False); ax_err.spines["right"].set_visible(False)
        fig_err.tight_layout(); st.pyplot(fig_err); plt.close(fig_err)

    with col_pw:
        st.markdown("#### Piecewise Linear vs Kuadratik")
        fig_pw = plot_piecewise_illustration(
            result_p["qmax"],
            breaks_actual=result_p["breaks"],
            slopes_actual=result_p["slopes"],
        )
        st.pyplot(fig_pw); plt.close(fig_pw)

    st.divider()
    st.caption("Model: Piecewise Linear MILP (CBC Solver / PuLP) | AIMMS Chapter 18 Exercise 18.3")
