"""
Portfolio Diversification – Strategic Asset Allocation
Konversi penuh dari Jupyter Notebook (ipywidgets) ke Streamlit

Jalankan dengan:
    python -m streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, LinearConstraint

# ──────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📈",
    layout="wide",
)

# ──────────────────────────────────────────────
# FUNGSI MODEL
# ──────────────────────────────────────────────
def portfolio_return(x, expected_returns_array):
    return np.sum(x * expected_returns_array)


def solve_strategic_portfolio(M_target, expected_returns_array, cov_matrix):
    n = len(expected_returns_array)

    def objective(x):
        return x @ cov_matrix @ x

    x0 = np.ones(n) / n
    bounds = Bounds([0] * n, [1] * n)

    A_eq = [np.ones(n)]
    b_eq = [1.0]
    A_ub = [-expected_returns_array]
    b_ub = [-M_target]

    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=-np.inf, ub=b_ub),
    ]

    result = minimize(
        objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
    )

    if result.success:
        x_res = result.x
        return x_res, result.fun, portfolio_return(x_res, expected_returns_array), True
    else:
        return None, None, None, False


def parse_inputs(categories_str, returns_str, cov_str):
    cats = [c.strip() for c in categories_str.split(",")]
    rets = np.array([float(r.strip()) for r in returns_str.split(",")])
    cov_lines = cov_str.strip().split("\n")
    cov = np.array(
        [[float(v.strip()) for v in line.split(",")] for line in cov_lines]
    )
    return cats, rets, cov


# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
DEFAULTS = {
    "cats": None,
    "rets": None,
    "cov": None,
    "error_msg": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("📈 Portfolio Diversification")
st.markdown("**Strategic Asset Allocation** — Mean-Variance Optimization (Markowitz / SLSQP)")
st.divider()

# ──────────────────────────────────────────────
# SIDEBAR — INPUT DATA
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Input Data")
    st.caption("Ubah data lalu klik **Terapkan Data Baru**.")

    categories_input = st.text_input(
        "Kategori Aset (pisahkan dengan koma)",
        value="Stocks, Bonds, Real Estate",
    )
    returns_input = st.text_input(
        "Ekspektasi Return (pisahkan dengan koma)",
        value="10.800, 7.600, 9.500",
    )
    cov_input = st.text_area(
        "Matriks Kovarians (baris baru = baris matriks)",
        value="2.250, -0.120, 0.450\n-0.120, 0.640, 0.336\n0.450, 0.336, 1.440",
        height=120,
        help="Setiap baris dipisahkan Enter, nilai dipisahkan koma",
    )

    apply_btn = st.button("✅ Terapkan Data Baru", use_container_width=True, type="primary")
    st.divider()
    st.markdown(
        "**Panduan format:**\n"
        "- Kategori: `Stocks, Bonds, Gold`\n"
        "- Return: `10.8, 7.6, 9.5`\n"
        "- Kovarians: satu baris per baris matriks"
    )

# ──────────────────────────────────────────────
# PARSE & VALIDASI INPUT
# ──────────────────────────────────────────────
if apply_btn or st.session_state.cats is None:
    try:
        cats, rets, cov = parse_inputs(categories_input, returns_input, cov_input)
        n = len(cats)

        if len(rets) != n:
            st.session_state.error_msg = (
                f"❌ Jumlah kategori ({n}) ≠ jumlah return ({len(rets)})."
            )
        elif cov.shape != (n, n):
            st.session_state.error_msg = (
                f"❌ Matriks kovarians harus {n}×{n}, bukan {cov.shape[0]}×{cov.shape[1]}."
            )
        else:
            st.session_state.cats = cats
            st.session_state.rets = rets
            st.session_state.cov = cov
            st.session_state.error_msg = None
            if apply_btn:
                st.toast("✅ Data berhasil diperbarui!", icon="✅")
    except Exception as e:
        st.session_state.error_msg = (
            f"❌ Kesalahan membaca data: {e}\n"
            "Pastikan format angka benar (pisahkan dengan koma)."
        )

if st.session_state.error_msg:
    st.error(st.session_state.error_msg)
    st.stop()

cats = st.session_state.cats
rets = st.session_state.rets
cov  = st.session_state.cov

# ──────────────────────────────────────────────
# SLIDER TARGET RETURN M
# ──────────────────────────────────────────────
M_min_val = float(min(rets))
M_max_val = float(max(rets))

M_target = st.slider(
    "🎚️ Sesuaikan Target Return (M)",
    min_value=M_min_val,
    max_value=M_max_val,
    value=round((M_min_val + M_max_val) / 2, 1),
    step=0.1,
    format="%.1f",
)

st.divider()

# ──────────────────────────────────────────────
# HITUNG TITIK OPTIMAL UNTUK M SAAT INI
# ──────────────────────────────────────────────
x_opt, risk_opt_var, return_opt, success = solve_strategic_portfolio(M_target, rets, cov)

if not success:
    st.warning(
        f"⚠️ Tidak dapat menemukan solusi feasible untuk M = {M_target:.1f}. "
        "Coba geser slider ke nilai lain."
    )
    st.stop()

risk_opt_std = np.sqrt(risk_opt_var)

# ──────────────────────────────────────────────
# HITUNG DATA KURVA (cached per data + M range)
# ──────────────────────────────────────────────
@st.cache_data
def compute_curve(rets_tuple, cov_tuple, n_points=40):
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

feasible_M_levels, risks_std_curve, alloc_curve = compute_curve(
    tuple(rets.tolist()), tuple(map(tuple, cov.tolist()))
)

# Titik spesifik (tiap 1.0 unit, sama persis seperti notebook)
M_specific_raw = np.arange(
    np.ceil(M_min_val * 10) / 10,
    M_max_val + 0.1,
    1.0,
)
M_specific = np.round(M_specific_raw, 1)
M_specific = M_specific[(M_specific >= M_min_val) & (M_specific <= M_max_val)]

specific_risks = []
for M_s in M_specific:
    _, rv_s, _, ok_s = solve_strategic_portfolio(float(M_s), rets, cov)
    specific_risks.append(float(np.sqrt(rv_s)) if ok_s else None)

# ──────────────────────────────────────────────
# BARIS 1 — METRIK RINGKASAN
# ──────────────────────────────────────────────
st.subheader(f"📊 Hasil Optimasi — Target M = {M_target:.1f}")

mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("🎯 Target Return (M)", f"{M_target:.1f}")
mc2.metric("📈 Expected Return Aktual", f"{return_opt:.4f}")
mc3.metric("📉 Risiko (Varians)", f"{risk_opt_var:.4f}")
mc4.metric("📊 Standar Deviasi", f"{risk_opt_std:.4f}")

st.divider()

# ──────────────────────────────────────────────
# BARIS 2 — GRAFIK 1 & GRAFIK 2
# ──────────────────────────────────────────────
col_g1, col_g2 = st.columns(2, gap="medium")

COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ── GRAFIK 1: Line Chart Alokasi Aset ──
with col_g1:
    st.markdown("#### 📉 Asset Allocation vs Expected Return")
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    alloc_arr = np.array(alloc_curve)  # shape (n_feasible, n_assets)
    for i, cat in enumerate(cats):
        ax1.plot(
            feasible_M_levels,
            alloc_arr[:, i],
            marker="o",
            linewidth=2.5,
            markersize=4,
            label=cat,
            color=COLORS[i % len(COLORS)],
        )

    ax1.axvline(
        x=M_target,
        color="red",
        linestyle="--",
        alpha=0.75,
        linewidth=2,
        label=f"Target Return: {M_target}",
    )

    ax1.set_xlabel("Minimal level of expected return", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Budget fractions", fontsize=11, fontweight="bold")
    ax1.set_title("Asset Allocation vs Expected Return", fontsize=13, fontweight="bold")
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.set_xlim(M_min_val - 0.2, M_max_val + 0.2)
    ax1.set_ylim(-0.05, 1.05)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# ── GRAFIK 2: Risk-Reward Characteristic (Figure 18.2) ──
with col_g2:
    st.markdown("#### 📈 Figure 18.2: Risk-Reward Characteristic")
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    ax2.plot(
        feasible_M_levels,
        risks_std_curve,
        "b-",
        linewidth=2.5,
        label="Risk-Reward Characteristic",
    )

    # Isi area di bawah kurva
    y_fill_min = max(0.0, min(risks_std_curve) * 0.8)
    ax2.fill_between(feasible_M_levels, risks_std_curve, y_fill_min, alpha=0.1, color="blue")

    # Titik spesifik (tiap 1.0 unit)
    for idx, M_val in enumerate(M_specific):
        if specific_risks[idx] is not None:
            ax2.plot(
                M_val,
                specific_risks[idx],
                "ro",
                markersize=10,
                markeredgecolor="darkred",
                markeredgewidth=1.5,
            )
            ax2.annotate(
                f"{specific_risks[idx]:.3f}",
                xy=(M_val, specific_risks[idx]),
                xytext=(8, 5),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="darkred",
            )

    # Bintang hijau = portofolio terpilih dari slider
    ax2.plot(
        M_target,
        risk_opt_std,
        "g*",
        markersize=18,
        markeredgecolor="darkgreen",
        markeredgewidth=1.5,
        label=f"Selected Portfolio (M = {M_target})",
        zorder=5,
    )

    ax2.set_xlabel("Minimal level of expected return", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Portfolio risk (Std Dev)", fontsize=11, fontweight="bold")
    ax2.set_title("Figure 18.2: Risk-Reward Characteristic", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.set_xlim(M_min_val - 0.3, M_max_val + 0.3)
    ax2.set_ylim(y_fill_min, max(risks_std_curve) * 1.1)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

# ──────────────────────────────────────────────
# BARIS 3 — TABEL ALOKASI + REKOMENDASI
# ──────────────────────────────────────────────
col_tbl, col_rec = st.columns([3, 2], gap="large")

# ── TABEL ALOKASI MULTI-TARGET ──
with col_tbl:
    st.markdown("#### 📋 Tabel Alokasi pada Berbagai Target Return")

    key_returns = sorted(
        set(
            np.round(
                [M_min_val, (M_min_val + M_max_val) / 2, M_max_val, M_target], 1
            ).tolist()
        )
    )

    tbl_rows = []
    for M_k in key_returns:
        x_k, rv_k, ret_k, ok_k = solve_strategic_portfolio(float(M_k), rets, cov)
        if ok_k:
            row = {"Target Return": f"{M_k:.1f}"}
            for i, cat in enumerate(cats):
                row[cat] = f"{x_k[i] * 100:.1f}%"
            row["Risk (Std)"] = f"{np.sqrt(rv_k):.4f}"
            row["Exp. Return"] = f"{ret_k:.4f}"
            tbl_rows.append(row)

    st.dataframe(tbl_rows, use_container_width=True, hide_index=True)

    # Highlight baris aktif dengan info box
    st.info(
        f"📌 **M = {M_target:.1f}** → "
        + " | ".join(
            [f"{cats[i]}: **{x_opt[i]*100:.1f}%**" for i in range(len(cats))]
        )
        + f" | Risk: **{risk_opt_std:.4f}**"
    )

# ── REKOMENDASI BAR ──
with col_rec:
    st.markdown(f"#### 💡 Rekomendasi untuk M = {M_target:.1f}")

    # Bar chart horizontal (seperti bar ASCII di notebook)
    fig_bar, ax_bar = plt.subplots(figsize=(5, max(3, len(cats) * 1.1)))
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(cats))]
    bars = ax_bar.barh(cats, x_opt * 100, color=bar_colors, edgecolor="white", height=0.55)

    for bar_obj, val in zip(bars, x_opt * 100):
        ax_bar.text(
            bar_obj.get_width() + 0.5,
            bar_obj.get_y() + bar_obj.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax_bar.set_xlim(0, 115)
    ax_bar.set_xlabel("Alokasi (%)", fontsize=10)
    ax_bar.set_title(f"Komposisi Portofolio (M = {M_target:.1f})", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    fig_bar.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Analisis risk-reward ringkas
    st.markdown("**📊 Analisis Risk-Reward:**")
    r_low  = min(risks_std_curve)
    r_high = max(risks_std_curve)
    delta_pct = (r_high - r_low) / r_low * 100

    st.markdown(
        f"- Return terendah ({M_min_val:.1f}) → risiko = **{r_low:.4f}**\n"
        f"- Return tertinggi ({M_max_val:.1f}) → risiko = **{r_high:.4f}**\n"
        f"- Risk naik sebesar **{delta_pct:.1f}%** dari min ke max return"
    )

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Mean-Variance Optimization (SLSQP) | "
    "Konversi dari Jupyter Notebook ipywidgets → Streamlit"
)
