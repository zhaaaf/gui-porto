"""
Portfolio Diversification – Strategic, Tactical & Downside Variance
Gabungan tiga model dalam satu aplikasi Streamlit

Jalankan dengan:
    python -m streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

# ══════════════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📈",
    layout="wide",
)

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ══════════════════════════════════════════════════════════════════════
# ── SHARED HELPERS ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════

def calculate_returns(price_matrix):
    return (price_matrix[1:] - price_matrix[:-1]) / price_matrix[:-1] * 100

def calculate_expected_returns(return_matrix):
    return np.mean(return_matrix, axis=0)

# ══════════════════════════════════════════════════════════════════════
# ── MODEL STRATEGIC ──────────────────────────────────────────────────
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
# ── MODEL TACTICAL ───────────────────────────────────────────────────
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

def parse_price_inputs(assets_str, prices_str):
    assets = [x.strip() for x in assets_str.split(",")]
    rows   = prices_str.strip().split("\n")
    prices = np.array([[float(v) for v in row.split(",")] for row in rows])
    return assets, prices

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
# ── MODEL DOWNSIDE VARIANCE ──────────────────────────────────────────
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

    # Sum x = 1
    Aeq = np.zeros((1, n_assets + T))
    Aeq[0, :n_assets] = 1
    constraints.append(LinearConstraint(Aeq, [1], [1]))

    # Expected return >= M
    Aret = np.zeros((1, n_assets + T))
    Aret[0, :n_assets] = mu
    constraints.append(LinearConstraint(Aret, [M_target], [np.inf]))

    # Scenario constraints: r_t @ x + q_t >= M
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
# DEFAULT DATA (shared Tactical & Downside)
# ══════════════════════════════════════════════════════════════════════
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
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════
for k, v in {
    "s_cats": None, "s_rets": None, "s_cov": None, "s_error": None,
    "t_assets": None, "t_prices": None, "t_error": None,
    "d_assets": None, "d_prices": None, "d_error": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════
# HEADER UTAMA
# ══════════════════════════════════════════════════════════════════════
st.title("📈 Portfolio Optimization")
st.markdown("**Mean-Variance & Downside Risk Optimization (SLSQP)** — AIMMS Chapter 18")

mode = st.radio(
    "Pilih Model Optimasi:",
    options=[
        "🏛️ Strategic Asset Allocation",
        "⚡ Tactical Asset Allocation",
        "📉 Downside Variance Optimization",
    ],
    horizontal=True,
)
st.divider()

# ══════════════════════════════════════════════════════════════════════
# ════════════════  STRATEGIC  ════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
if mode.startswith("🏛️"):

    with st.sidebar:
        st.header("⚙️ Input — Strategic")
        st.caption("Ubah data lalu klik **Terapkan Data Baru**.")
        categories_input = st.text_input(
            "Kategori Aset (pisahkan koma)",
            value="Stocks, Bonds, Real Estate",
        )
        returns_input = st.text_input(
            "Ekspektasi Return (pisahkan koma)",
            value="10.800, 7.600, 9.500",
        )
        cov_input = st.text_area(
            "Matriks Kovarians (baris baru = baris matriks)",
            value="2.250, -0.120, 0.450\n-0.120, 0.640, 0.336\n0.450, 0.336, 1.440",
            height=130,
            help="Setiap baris dipisahkan Enter, nilai dipisahkan koma",
        )
        apply_btn = st.button("✅ Terapkan Data Baru", use_container_width=True, type="primary")
        st.divider()
        st.markdown(
            "**Format:**\n"
            "- Kategori: `Stocks, Bonds, Gold`\n"
            "- Return: `10.8, 7.6, 9.5`\n"
            "- Kovarians: satu baris per baris matriks"
        )

    if apply_btn or st.session_state.s_cats is None:
        try:
            cats, rets, cov = parse_strategic_inputs(categories_input, returns_input, cov_input)
            n = len(cats)
            if len(rets) != n:
                st.session_state.s_error = f"❌ Jumlah kategori ({n}) ≠ jumlah return ({len(rets)})."
            elif cov.shape != (n, n):
                st.session_state.s_error = f"❌ Matriks kovarians harus {n}×{n}."
            else:
                st.session_state.s_cats  = cats
                st.session_state.s_rets  = rets
                st.session_state.s_cov   = cov
                st.session_state.s_error = None
                if apply_btn:
                    st.toast("✅ Data berhasil diperbarui!", icon="✅")
        except Exception as e:
            st.session_state.s_error = f"❌ Kesalahan membaca data: {e}"

    if st.session_state.s_error:
        st.error(st.session_state.s_error)
        st.stop()

    cats = st.session_state.s_cats
    rets = st.session_state.s_rets
    cov  = st.session_state.s_cov

    M_min_val = float(min(rets))
    M_max_val = float(max(rets))
    M_target  = st.slider(
        "🎚️ Target Return (M)",
        min_value=M_min_val, max_value=M_max_val,
        value=round((M_min_val + M_max_val) / 2, 1),
        step=0.1, format="%.1f",
    )
    st.divider()

    x_opt, risk_opt_var, return_opt, success = solve_strategic_portfolio(M_target, rets, cov)
    if not success:
        st.warning(f"⚠️ Tidak ditemukan solusi feasible untuk M = {M_target:.1f}.")
        st.stop()
    risk_opt_std = np.sqrt(risk_opt_var)

    feasible_M_levels, risks_std_curve, alloc_curve = compute_strategic_curve(
        tuple(rets.tolist()), tuple(map(tuple, cov.tolist()))
    )

    M_specific_raw = np.arange(np.ceil(M_min_val * 10) / 10, M_max_val + 0.1, 1.0)
    M_specific     = np.round(M_specific_raw, 1)
    M_specific     = M_specific[(M_specific >= M_min_val) & (M_specific <= M_max_val)]
    specific_risks = []
    for M_s in M_specific:
        _, rv_s, _, ok_s = solve_strategic_portfolio(float(M_s), rets, cov)
        specific_risks.append(float(np.sqrt(rv_s)) if ok_s else None)

    st.subheader(f"📊 Hasil Optimasi — Target M = {M_target:.1f}")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 Target Return (M)",      f"{M_target:.1f}")
    mc2.metric("📈 Expected Return Aktual", f"{return_opt:.4f}")
    mc3.metric("📉 Risiko (Varians)",       f"{risk_opt_var:.4f}")
    mc4.metric("📊 Standar Deviasi",        f"{risk_opt_std:.4f}")
    st.divider()

    col_g1, col_g2 = st.columns(2, gap="medium")

    with col_g1:
        st.markdown("#### 📉 Asset Allocation vs Expected Return")
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        alloc_arr = np.array(alloc_curve)
        for i, cat in enumerate(cats):
            ax1.plot(feasible_M_levels, alloc_arr[:, i],
                     marker="o", linewidth=2.5, markersize=4,
                     label=cat, color=COLORS[i % len(COLORS)])
        ax1.axvline(M_target, color="red", linestyle="--", alpha=0.75,
                    linewidth=2, label=f"Target Return: {M_target}")
        ax1.set_xlabel("Minimal level of expected return", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Budget fractions",                 fontsize=11, fontweight="bold")
        ax1.set_title("Asset Allocation vs Expected Return", fontsize=13, fontweight="bold")
        ax1.legend(loc="center right", fontsize=9)
        ax1.grid(True, linestyle="--", alpha=0.3)
        ax1.set_xlim(M_min_val - 0.2, M_max_val + 0.2)
        ax1.set_ylim(-0.05, 1.05)
        ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
        fig1.tight_layout(); st.pyplot(fig1); plt.close(fig1)

    with col_g2:
        st.markdown("#### 📈 Figure 18.2: Risk-Reward Characteristic")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.plot(feasible_M_levels, risks_std_curve, "b-", linewidth=2.5,
                 label="Risk-Reward Characteristic")
        y_fill_min = max(0.0, min(risks_std_curve) * 0.8)
        ax2.fill_between(feasible_M_levels, risks_std_curve, y_fill_min, alpha=0.1, color="blue")
        for idx, M_val in enumerate(M_specific):
            if specific_risks[idx] is not None:
                ax2.plot(M_val, specific_risks[idx], "ro", markersize=10,
                         markeredgecolor="darkred", markeredgewidth=1.5)
                ax2.annotate(f"{specific_risks[idx]:.3f}", xy=(M_val, specific_risks[idx]),
                             xytext=(8, 5), textcoords="offset points",
                             fontsize=9, fontweight="bold", color="darkred")
        ax2.plot(M_target, risk_opt_std, "g*", markersize=18,
                 markeredgecolor="darkgreen", markeredgewidth=1.5,
                 label=f"Selected Portfolio (M = {M_target})", zorder=5)
        ax2.set_xlabel("Minimal level of expected return", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Portfolio risk (Std Dev)",          fontsize=11, fontweight="bold")
        ax2.set_title("Figure 18.2: Risk-Reward Characteristic", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.set_xlim(M_min_val - 0.3, M_max_val + 0.3)
        ax2.set_ylim(y_fill_min, max(risks_std_curve) * 1.1)
        ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
        fig2.tight_layout(); st.pyplot(fig2); plt.close(fig2)

    st.divider()

    col_tbl, col_rec = st.columns([3, 2], gap="large")

    with col_tbl:
        st.markdown("#### 📋 Tabel Alokasi pada Berbagai Target Return")
        key_returns = sorted(set(np.round(
            [M_min_val, (M_min_val + M_max_val) / 2, M_max_val, M_target], 1
        ).tolist()))
        tbl_rows = []
        for M_k in key_returns:
            x_k, rv_k, ret_k, ok_k = solve_strategic_portfolio(float(M_k), rets, cov)
            if ok_k:
                row = {"Target Return": f"{M_k:.1f}"}
                for i, cat in enumerate(cats):
                    row[cat] = f"{x_k[i]*100:.1f}%"
                row["Risk (Std)"]  = f"{np.sqrt(rv_k):.4f}"
                row["Exp. Return"] = f"{ret_k:.4f}"
                tbl_rows.append(row)
        st.dataframe(tbl_rows, use_container_width=True, hide_index=True)
        st.info(
            f"📌 **M = {M_target:.1f}** → "
            + " | ".join([f"{cats[i]}: **{x_opt[i]*100:.1f}%**" for i in range(len(cats))])
            + f" | Risk: **{risk_opt_std:.4f}**"
        )

    with col_rec:
        st.markdown(f"#### 💡 Rekomendasi untuk M = {M_target:.1f}")
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
        ax_bar.set_title(f"Komposisi Portofolio (M = {M_target:.1f})", fontsize=11, fontweight="bold")
        ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
        ax_bar.spines["top"].set_visible(False); ax_bar.spines["right"].set_visible(False)
        fig_bar.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)

        st.markdown("**📊 Analisis Risk-Reward:**")
        r_low, r_high = min(risks_std_curve), max(risks_std_curve)
        delta_pct = (r_high - r_low) / r_low * 100
        st.markdown(
            f"- Return terendah ({M_min_val:.1f}) → risiko = **{r_low:.4f}**\n"
            f"- Return tertinggi ({M_max_val:.1f}) → risiko = **{r_high:.4f}**\n"
            f"- Risk naik sebesar **{delta_pct:.1f}%** dari min ke max return"
        )

    st.divider()
    st.caption("Model: Strategic — Mean-Variance Optimization (SLSQP) | AIMMS Chapter 18")


# ══════════════════════════════════════════════════════════════════════
# ════════════════  TACTICAL  ═════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
elif mode.startswith("⚡"):

    with st.sidebar:
        st.header("⚙️ Input — Tactical")
        st.caption("Masukkan nama aset dan data harga historis.")
        assets_input = st.text_input("Nama Aset (pisahkan koma)", value=DEFAULT_ASSETS)
        prices_input = st.text_area(
            "Data Harga Historis (baris = periode, kolom = aset)",
            value=DEFAULT_PRICES, height=320,
            help="Setiap baris adalah satu periode; nilai dipisahkan koma",
        )
        apply_btn_t = st.button("✅ Terapkan Data Baru", use_container_width=True, type="primary")
        st.divider()
        st.markdown("**Format:**\n- Aset: `RD,AKZ,KLM`\n- Harga: satu baris per periode, nilai koma")

    if apply_btn_t or st.session_state.t_assets is None:
        try:
            assets, prices = parse_price_inputs(assets_input, prices_input)
            n_a = len(assets)
            if prices.shape[1] != n_a:
                st.session_state.t_error = f"❌ Kolom harga ({prices.shape[1]}) ≠ jumlah aset ({n_a})."
            elif prices.shape[0] < 2:
                st.session_state.t_error = "❌ Data harga minimal 2 baris."
            else:
                st.session_state.t_assets = assets
                st.session_state.t_prices = prices
                st.session_state.t_error  = None
                if apply_btn_t:
                    st.toast("✅ Data berhasil diperbarui!", icon="✅")
        except Exception as e:
            st.session_state.t_error = f"❌ Kesalahan membaca data: {e}"

    if st.session_state.t_error:
        st.error(st.session_state.t_error)
        st.stop()

    assets = st.session_state.t_assets
    prices = st.session_state.t_prices
    returns = calculate_returns(prices)
    mu      = calculate_expected_returns(returns)

    M_slider = st.slider(
        "🎚️ Target Return (M)",
        min_value=0.0, max_value=float(max(mu)),
        value=round(float(max(mu)) * 0.4, 3),
        step=0.01, format="%.2f",
    )
    st.divider()

    x_opt_t, risk_t, ret_t, mu_t, success_t = solve_tactical_portfolio(M_slider, returns)
    if not success_t:
        st.warning(f"⚠️ Tidak ditemukan solusi feasible untuk M = {M_slider:.2f}.")
        st.stop()
    risk_std_t = np.sqrt(risk_t)

    st.subheader(f"📊 Hasil Optimasi — Target M = {M_slider:.2f}")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 Target Return (M)",      f"{M_slider:.4f}")
    mc2.metric("📈 Expected Return Aktual", f"{ret_t:.4f}")
    mc3.metric("📉 Risiko (Varians)",       f"{risk_t:.4f}")
    mc4.metric("📊 Standar Deviasi",        f"{risk_std_t:.4f}")
    st.divider()

    with st.expander("📊 Expected Return & Alokasi Optimal per Aset"):
        df_mu = pd.DataFrame({
            "Aset":            assets,
            "Expected Return": [f"{v:.4f}" for v in mu_t],
            "Alokasi Optimal": [f"{x_opt_t[i]*100:.2f}%" for i in range(len(assets))],
        })
        st.dataframe(df_mu, use_container_width=True, hide_index=True)

    feasible_M_t, risks_std_t, alloc_dict_t, mu_list = compute_tactical_curve(
        tuple(map(tuple, prices.tolist()))
    )

    col_g1t, col_g2t = st.columns(2, gap="medium")

    with col_g1t:
        st.markdown("#### 📈 Figure 18.4: Risk-Reward Characteristic")
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.plot(feasible_M_t, risks_std_t, linewidth=2.5, marker="o",
                 color="#1f77b4", label="Risk-Reward Characteristic")
        y_fill_min_t = max(0.0, min(risks_std_t) * 0.8)
        ax4.fill_between(feasible_M_t, risks_std_t, y_fill_min_t, alpha=0.1, color="#1f77b4")
        ax4.scatter(M_slider, risk_std_t, s=200, marker="*", color="green",
                    edgecolors="darkgreen", zorder=5, label=f"Selected (M={M_slider:.2f})")
        ax4.set_xlabel("Minimal Expected Return", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Portfolio Risk (Std Dev)", fontsize=11, fontweight="bold")
        ax4.set_title("Figure 18.4: Risk-Reward Characteristic", fontsize=13, fontweight="bold")
        ax4.legend(loc="upper left", fontsize=9)
        ax4.grid(True, linestyle="--", alpha=0.4)
        ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)
        fig4.tight_layout(); st.pyplot(fig4); plt.close(fig4)

    with col_g2t:
        st.markdown("#### 📉 Figure 18.5: Portfolio Diversification")
        fig5, ax5 = plt.subplots(figsize=(7, 5))
        for i, a in enumerate(assets):
            ax5.plot(feasible_M_t, alloc_dict_t[i],
                     linewidth=2.5, marker="o", markersize=4,
                     label=a, color=COLORS[i % len(COLORS)])
        ax5.axvline(M_slider, color="red", linestyle="--", alpha=0.75,
                    linewidth=2, label=f"Target Return: {M_slider:.2f}")
        ax5.set_xlabel("Minimal Expected Return", fontsize=11, fontweight="bold")
        ax5.set_ylabel("Budget Fraction",          fontsize=11, fontweight="bold")
        ax5.set_title("Figure 18.5: Portfolio Diversification", fontsize=13, fontweight="bold")
        ax5.legend(loc="center right", fontsize=9)
        ax5.grid(True, linestyle="--", alpha=0.3)
        ax5.set_ylim(-0.05, 1.05)
        ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)
        fig5.tight_layout(); st.pyplot(fig5); plt.close(fig5)

    st.divider()

    col_tbl_t, col_rec_t = st.columns([3, 2], gap="large")

    with col_tbl_t:
        st.markdown("#### 📋 Tabel Alokasi Berbagai Target Return")
        sample_M_t = np.linspace(0, float(max(mu_t)), 5)
        tbl_rows_t = []
        for M_k in sample_M_t:
            x_k, r_k, rr_k, _, ok_k = solve_tactical_portfolio(float(M_k), returns)
            if ok_k:
                row = {"Target Return": f"{M_k:.4f}", "Risk (Var)": f"{r_k:.4f}"}
                for i, a in enumerate(assets):
                    row[a] = f"{x_k[i]*100:.1f}%"
                tbl_rows_t.append(row)
        st.dataframe(tbl_rows_t, use_container_width=True, hide_index=True)
        st.info(
            f"📌 **M = {M_slider:.2f}** → "
            + " | ".join([f"{assets[i]}: **{x_opt_t[i]*100:.1f}%**" for i in range(len(assets))])
            + f" | Risk: **{risk_std_t:.4f}**"
        )
        st.markdown("#### 📋 Scenario Returns (10 Baris Pertama)")
        df_ret = pd.DataFrame(returns, columns=assets)
        st.dataframe(df_ret.head(10), use_container_width=True)

    with col_rec_t:
        st.markdown(f"#### 💡 Rekomendasi untuk M = {M_slider:.2f}")
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
        ax_bar_t.set_title(f"Komposisi Portofolio (M = {M_slider:.2f})", fontsize=11, fontweight="bold")
        ax_bar_t.grid(axis="x", linestyle="--", alpha=0.4)
        ax_bar_t.spines["top"].set_visible(False); ax_bar_t.spines["right"].set_visible(False)
        fig_bar_t.tight_layout(); st.pyplot(fig_bar_t); plt.close(fig_bar_t)

        st.markdown("**📊 Analisis Risk-Reward:**")
        r_low_t, r_high_t = min(risks_std_t), max(risks_std_t)
        delta_pct_t = (r_high_t - r_low_t) / r_low_t * 100
        st.markdown(
            f"- Return terendah (0.00) → risiko = **{r_low_t:.4f}**\n"
            f"- Return tertinggi ({max(mu_t):.2f}) → risiko = **{r_high_t:.4f}**\n"
            f"- Risk naik sebesar **{delta_pct_t:.1f}%** dari min ke max return"
        )

    st.divider()
    st.caption("Model: Tactical — Mean-Variance Optimization (SLSQP) | AIMMS Chapter 18")


# ══════════════════════════════════════════════════════════════════════
# ════════════════  DOWNSIDE VARIANCE  ════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
else:

    with st.sidebar:
        st.header("⚙️ Input — Downside Variance")
        st.caption("Masukkan nama aset dan data harga historis.")
        assets_input_d = st.text_input("Nama Aset (pisahkan koma)", value=DEFAULT_ASSETS, key="d_assets_input")
        prices_input_d = st.text_area(
            "Data Harga Historis (baris = periode, kolom = aset)",
            value=DEFAULT_PRICES, height=320, key="d_prices_input",
            help="Setiap baris adalah satu periode; nilai dipisahkan koma",
        )
        apply_btn_d = st.button("✅ Terapkan Data Baru", use_container_width=True, type="primary", key="d_apply")
        st.divider()
        st.markdown("**Format:**\n- Aset: `RD,AKZ,KLM`\n- Harga: satu baris per periode, nilai koma")
        st.divider()
        st.markdown(
            "**ℹ️ Downside Variance:**\n\n"
            "Meminimalkan **semi-variance** (hanya deviasi di bawah target), "
            "bukan total variance. Lebih konservatif terhadap kerugian."
        )

    if apply_btn_d or st.session_state.d_assets is None:
        try:
            assets_d, prices_d = parse_price_inputs(assets_input_d, prices_input_d)
            n_d = len(assets_d)
            if prices_d.shape[1] != n_d:
                st.session_state.d_error = f"❌ Kolom harga ({prices_d.shape[1]}) ≠ jumlah aset ({n_d})."
            elif prices_d.shape[0] < 2:
                st.session_state.d_error = "❌ Data harga minimal 2 baris."
            else:
                st.session_state.d_assets = assets_d
                st.session_state.d_prices = prices_d
                st.session_state.d_error  = None
                if apply_btn_d:
                    st.toast("✅ Data berhasil diperbarui!", icon="✅")
        except Exception as e:
            st.session_state.d_error = f"❌ Kesalahan membaca data: {e}"

    if st.session_state.d_error:
        st.error(st.session_state.d_error)
        st.stop()

    assets_d = st.session_state.d_assets
    prices_d = st.session_state.d_prices
    returns_d = calculate_returns(prices_d)
    mu_d      = calculate_expected_returns(returns_d)

    M_slider_d = st.slider(
        "🎚️ Target Return (M)",
        min_value=0.0, max_value=float(max(mu_d)),
        value=round(float(max(mu_d)) * 0.4, 3),
        step=0.01, format="%.2f",
        key="d_slider",
    )
    st.divider()

    with st.spinner("Menghitung optimasi downside..."):
        x_opt_d, risk_d, ret_d, mu_d_arr, success_d = solve_downside_portfolio(M_slider_d, returns_d)

    if not success_d:
        st.warning(f"⚠️ Tidak ditemukan solusi feasible untuk M = {M_slider_d:.2f}.")
        st.stop()
    risk_std_d = np.sqrt(risk_d)

    st.subheader(f"📊 Hasil Optimasi — Target M = {M_slider_d:.2f}")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 Target Return (M)",         f"{M_slider_d:.4f}")
    mc2.metric("📈 Expected Return Aktual",    f"{ret_d:.4f}")
    mc3.metric("📉 Downside Risk (Semi-Var)",  f"{risk_d:.4f}")
    mc4.metric("📊 Downside Std Dev",          f"{risk_std_d:.4f}")
    st.divider()

    with st.expander("📊 Expected Return & Alokasi Optimal per Aset"):
        df_mu_d = pd.DataFrame({
            "Aset":            assets_d,
            "Expected Return": [f"{v:.4f}" for v in mu_d_arr],
            "Alokasi Optimal": [f"{x_opt_d[i]*100:.2f}%" for i in range(len(assets_d))],
        })
        st.dataframe(df_mu_d, use_container_width=True, hide_index=True)

    with st.spinner("Menghitung kurva downside..."):
        feasible_M_d, risks_std_d, alloc_dict_d, mu_list_d = compute_downside_curve(
            tuple(map(tuple, prices_d.tolist()))
        )

    col_g1d, col_g2d = st.columns(2, gap="medium")

    with col_g1d:
        st.markdown("#### 📈 Downside Risk-Reward Characteristic")
        fig_d1, ax_d1 = plt.subplots(figsize=(7, 5))
        ax_d1.plot(feasible_M_d, risks_std_d, linewidth=2.5, marker="o",
                   color="#d62728", label="Downside Risk-Reward")
        y_fill_min_d = max(0.0, min(risks_std_d) * 0.8)
        ax_d1.fill_between(feasible_M_d, risks_std_d, y_fill_min_d, alpha=0.1, color="#d62728")
        ax_d1.scatter(M_slider_d, risk_std_d, s=200, marker="*", color="green",
                      edgecolors="darkgreen", zorder=5, label=f"Selected (M={M_slider_d:.2f})")
        ax_d1.set_xlabel("Target Return (M)", fontsize=11, fontweight="bold")
        ax_d1.set_ylabel("Downside Risk (Std Dev)", fontsize=11, fontweight="bold")
        ax_d1.set_title("Downside Risk-Reward Characteristic", fontsize=13, fontweight="bold")
        ax_d1.legend(loc="upper left", fontsize=9)
        ax_d1.grid(True, linestyle="--", alpha=0.4)
        ax_d1.spines["top"].set_visible(False); ax_d1.spines["right"].set_visible(False)
        fig_d1.tight_layout(); st.pyplot(fig_d1); plt.close(fig_d1)

    with col_g2d:
        st.markdown("#### 📉 Portfolio Diversification (Downside)")
        fig_d2, ax_d2 = plt.subplots(figsize=(7, 5))
        for i, a in enumerate(assets_d):
            ax_d2.plot(feasible_M_d, alloc_dict_d[i],
                       linewidth=2.5, marker="o", markersize=4,
                       label=a, color=COLORS[i % len(COLORS)])
        ax_d2.axvline(M_slider_d, color="red", linestyle="--", alpha=0.75,
                      linewidth=2, label=f"Target Return: {M_slider_d:.2f}")
        ax_d2.set_xlabel("Target Return (M)", fontsize=11, fontweight="bold")
        ax_d2.set_ylabel("Budget Fraction",    fontsize=11, fontweight="bold")
        ax_d2.set_title("Portfolio Diversification (Downside)", fontsize=13, fontweight="bold")
        ax_d2.legend(loc="center right", fontsize=9)
        ax_d2.grid(True, linestyle="--", alpha=0.3)
        ax_d2.set_ylim(-0.05, 1.05)
        ax_d2.spines["top"].set_visible(False); ax_d2.spines["right"].set_visible(False)
        fig_d2.tight_layout(); st.pyplot(fig_d2); plt.close(fig_d2)

    st.divider()

    col_tbl_d, col_rec_d = st.columns([3, 2], gap="large")

    with col_tbl_d:
        st.markdown("#### 📋 Tabel Alokasi Berbagai Target Return")
        sample_M_d = np.linspace(0, float(max(mu_list_d)), 5)
        tbl_rows_d = []
        for M_k in sample_M_d:
            x_k, r_k, rr_k, _, ok_k = solve_downside_portfolio(float(M_k), returns_d)
            if ok_k:
                row = {"Target Return": f"{M_k:.4f}", "Downside Risk": f"{r_k:.4f}"}
                for i, a in enumerate(assets_d):
                    row[a] = f"{x_k[i]*100:.1f}%"
                tbl_rows_d.append(row)
        st.dataframe(tbl_rows_d, use_container_width=True, hide_index=True)
        st.info(
            f"📌 **M = {M_slider_d:.2f}** → "
            + " | ".join([f"{assets_d[i]}: **{x_opt_d[i]*100:.1f}%**" for i in range(len(assets_d))])
            + f" | Downside Risk: **{risk_std_d:.4f}**"
        )
        st.markdown("#### 📋 Scenario Returns (10 Baris Pertama)")
        df_ret_d = pd.DataFrame(returns_d, columns=assets_d)
        st.dataframe(df_ret_d.head(10), use_container_width=True)

    with col_rec_d:
        st.markdown(f"#### 💡 Rekomendasi untuk M = {M_slider_d:.2f}")
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
        ax_bar_d.set_title(f"Komposisi Portofolio (M = {M_slider_d:.2f})", fontsize=11, fontweight="bold")
        ax_bar_d.grid(axis="x", linestyle="--", alpha=0.4)
        ax_bar_d.spines["top"].set_visible(False); ax_bar_d.spines["right"].set_visible(False)
        fig_bar_d.tight_layout(); st.pyplot(fig_bar_d); plt.close(fig_bar_d)

        st.markdown("**📊 Analisis Risk-Reward:**")
        r_low_d, r_high_d = min(risks_std_d), max(risks_std_d)
        delta_pct_d = (r_high_d - r_low_d) / r_low_d * 100
        st.markdown(
            f"- Return terendah (0.00) → downside risk = **{r_low_d:.4f}**\n"
            f"- Return tertinggi ({max(mu_list_d):.2f}) → downside risk = **{r_high_d:.4f}**\n"
            f"- Risk naik sebesar **{delta_pct_d:.1f}%** dari min ke max return"
        )

    st.divider()
    st.caption("Model: Downside Variance Optimization (SLSQP) | AIMMS Chapter 18")
