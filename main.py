import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, LinearConstraint, minimize


# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Dasbor Investasi Strategis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================== CUSTOM CSS ALA TAILWIND ====================
def load_tailwind_like_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .stApp {
                background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 45%, #f8fafc 100%);
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1280px;
            }

            [data-testid="stSidebar"] {
                background: #ffffff;
                border-right: 1px solid #e5e7eb;
            }

            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
                color: #0f172a;
            }

            .hero-card {
                background: rgba(255, 255, 255, 0.92);
                border: 1px solid #e5e7eb;
                border-radius: 24px;
                padding: 28px 30px;
                box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
                margin-bottom: 22px;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 7px 12px;
                border-radius: 999px;
                background: #eef2ff;
                color: #4338ca;
                font-size: 13px;
                font-weight: 700;
                margin-bottom: 14px;
            }

            .hero-title {
                color: #0f172a;
                font-size: 38px;
                line-height: 1.08;
                font-weight: 800;
                letter-spacing: -0.04em;
                margin: 0 0 10px 0;
            }

            .hero-subtitle {
                color: #475569;
                font-size: 16px;
                line-height: 1.7;
                margin: 0;
                max-width: 780px;
            }

            .section-card {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #e5e7eb;
                border-radius: 22px;
                padding: 22px;
                box-shadow: 0 14px 35px rgba(15, 23, 42, 0.06);
                margin-bottom: 18px;
            }

            .small-label {
                color: #64748b;
                font-size: 13px;
                font-weight: 700;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                margin-bottom: 6px;
            }

            .card-title {
                color: #0f172a;
                font-size: 21px;
                font-weight: 800;
                margin: 0 0 14px 0;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 12px;
                margin: 14px 0;
            }

            .metric-card {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 16px;
            }

            .metric-label {
                color: #64748b;
                font-size: 13px;
                font-weight: 700;
                margin-bottom: 8px;
            }

            .metric-value {
                color: #0f172a;
                font-size: 24px;
                line-height: 1;
                font-weight: 800;
            }

            .info-box {
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 18px;
                padding: 15px 16px;
                color: #1e3a8a;
                font-size: 14px;
                line-height: 1.6;
                margin-top: 12px;
            }

            div.stButton > button:first-child {
                border-radius: 14px;
                border: 0;
                background: linear-gradient(135deg, #4f46e5, #2563eb);
                color: white;
                font-weight: 800;
                padding: 0.75rem 1rem;
                box-shadow: 0 10px 22px rgba(37, 99, 235, 0.22);
            }

            div.stButton > button:first-child:hover {
                background: linear-gradient(135deg, #4338ca, #1d4ed8);
                color: white;
                border: 0;
            }

            [data-testid="stMetric"] {
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 18px;
                padding: 14px 16px;
            }

            [data-testid="stDataFrame"] {
                border-radius: 16px;
                overflow: hidden;
            }

            .footer-text {
                color: #64748b;
                text-align: center;
                font-size: 13px;
                padding: 18px 0 8px 0;
            }

            @media (max-width: 768px) {
                .hero-title {
                    font-size: 30px;
                }

                .metric-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


load_tailwind_like_css()


# ==================== FUNGSI MODEL ====================
def portfolio_return(weights, expected_returns_array):
    return float(np.sum(weights * expected_returns_array))


def portfolio_risk(weights, cov_matrix):
    return float(weights @ cov_matrix @ weights)


def solve_strategic_portfolio(target_return, expected_returns_array, cov_matrix):
    n_assets = len(expected_returns_array)

    def objective(weights):
        return portfolio_risk(weights, cov_matrix)

    x0 = np.ones(n_assets) / n_assets
    bounds = Bounds(np.zeros(n_assets), np.ones(n_assets))

    constraints = [
        LinearConstraint(np.ones((1, n_assets)), lb=[1.0], ub=[1.0]),
        LinearConstraint(expected_returns_array.reshape(1, -1), lb=[target_return], ub=[np.inf]),
    ]

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        return None, None, None, False

    weights = np.clip(result.x, 0, 1)
    total_weight = weights.sum()

    if total_weight <= 0:
        return None, None, None, False

    weights = weights / total_weight

    actual_return = portfolio_return(weights, expected_returns_array)
    actual_risk = portfolio_risk(weights, cov_matrix)

    if actual_return + 1e-7 < target_return:
        return None, None, None, False

    return weights, actual_risk, actual_return, True


def parse_number_list(raw_text):
    values = [item.strip() for item in raw_text.split(",") if item.strip()]
    return np.array([float(item) for item in values], dtype=float)


def parse_covariance_matrix(raw_text):
    rows = [row.strip() for row in raw_text.strip().splitlines() if row.strip()]
    matrix = []

    for row in rows:
        matrix.append([float(value.strip()) for value in row.split(",") if value.strip()])

    return np.array(matrix, dtype=float)


def parse_inputs(categories_str, returns_str, cov_str):
    categories = [item.strip() for item in categories_str.split(",") if item.strip()]
    returns = parse_number_list(returns_str)
    cov_matrix = parse_covariance_matrix(cov_str)

    return categories, returns, cov_matrix


def validate_inputs(categories, returns, cov_matrix):
    errors = []
    n_assets = len(categories)

    if n_assets == 0:
        errors.append("Kategori aset tidak boleh kosong.")

    if len(set(categories)) != n_assets:
        errors.append("Nama kategori aset tidak boleh duplikat.")

    if len(returns) != n_assets:
        errors.append(
            f"Jumlah return harus sama dengan jumlah kategori. Kategori: {n_assets}, return: {len(returns)}."
        )

    if cov_matrix.ndim != 2:
        errors.append("Matriks kovarians harus berbentuk dua dimensi.")
    elif cov_matrix.shape != (n_assets, n_assets):
        errors.append(
            f"Ukuran matriks kovarians harus {n_assets} × {n_assets}, bukan {cov_matrix.shape[0]} × {cov_matrix.shape[1]}."
        )

    if errors:
        return errors

    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(cov_matrix)):
        errors.append("Return dan matriks kovarians hanya boleh berisi angka valid.")

    if not np.allclose(cov_matrix, cov_matrix.T, atol=1e-8):
        errors.append("Matriks kovarians harus simetris. Nilai baris dan kolom yang berpasangan harus sama.")

    eigen_values = np.linalg.eigvalsh((cov_matrix + cov_matrix.T) / 2)
    if np.min(eigen_values) < -1e-8:
        errors.append("Matriks kovarians tidak valid karena bukan positive semi-definite.")

    return errors


def format_percent(value):
    return f"{value:.2f}%"


# ==================== HEADER ====================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-badge">📊 Mean-Variance Optimization</div>
        <h1 class="hero-title">Dasbor Investasi Strategis</h1>
        <p class="hero-subtitle">
            Optimasi alokasi aset berbasis model Markowitz. Masukkan kategori aset,
            ekspektasi return, dan matriks kovarians untuk mencari portofolio dengan risiko minimum
            pada target return tertentu.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================== SIDEBAR INPUT ====================
with st.sidebar:
    st.markdown("### ⚙️ Input Data")

    categories_input = st.text_input(
        "Kategori aset",
        value="Stocks, Bonds, Real Estate",
        help="Pisahkan setiap kategori dengan koma.",
    )

    returns_input = st.text_input(
        "Ekspektasi return",
        value="10.800, 7.600, 9.500",
        help="Gunakan titik sebagai desimal. Contoh: 10.8, 7.6, 9.5.",
    )

    cov_input = st.text_area(
        "Matriks kovarians",
        value="2.250, -0.120, 0.450\n-0.120, 0.640, 0.336\n0.450, 0.336, 1.440",
        height=140,
        help="Baris baru untuk baris matriks. Koma untuk memisahkan nilai.",
    )

    apply_btn = st.button("Terapkan Data", use_container_width=True, type="primary")

    st.markdown(
        """
        <div class="info-box">
            Format angka memakai titik desimal. Contoh benar: 10.8, 7.6, 9.5.
            Jangan memakai koma desimal karena koma dipakai sebagai pemisah data.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================== SESSION STATE ====================
def initialize_state():
    if "cats" not in st.session_state:
        st.session_state.cats = None
        st.session_state.rets = None
        st.session_state.cov = None
        st.session_state.error_msg = None


initialize_state()

if apply_btn or st.session_state.cats is None:
    try:
        cats, rets, cov = parse_inputs(categories_input, returns_input, cov_input)
        validation_errors = validate_inputs(cats, rets, cov)

        if validation_errors:
            st.session_state.error_msg = "\n".join([f"• {error}" for error in validation_errors])
        else:
            st.session_state.cats = cats
            st.session_state.rets = rets
            st.session_state.cov = (cov + cov.T) / 2
            st.session_state.error_msg = None

            if apply_btn:
                st.success("Data berhasil diperbarui.")

    except ValueError:
        st.session_state.error_msg = (
            "Input angka tidak valid. Gunakan angka dengan titik desimal dan pisahkan nilai memakai koma."
        )
    except Exception as error:
        st.session_state.error_msg = f"Terjadi kesalahan saat membaca input: {error}"


if st.session_state.error_msg:
    st.error(st.session_state.error_msg)
    st.stop()

cats = st.session_state.cats
rets = st.session_state.rets
cov = st.session_state.cov


# ==================== RINGKASAN DATA ====================
summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
summary_col_1.metric("Jumlah Aset", f"{len(cats)}")
summary_col_2.metric("Return Minimum", f"{float(np.min(rets)):.2f}")
summary_col_3.metric("Return Maksimum", f"{float(np.max(rets)):.2f}")

st.markdown("<br>", unsafe_allow_html=True)


# ==================== TARGET RETURN ====================
M_min_val = float(np.min(rets))
M_max_val = float(np.max(rets))

if np.isclose(M_min_val, M_max_val):
    M_target = M_min_val
    st.info(f"Semua aset memiliki return yang sama. Target return otomatis diatur ke {M_target:.2f}.")
else:
    M_default = float(np.round((M_min_val + M_max_val) / 2, 2))
    M_target = st.slider(
        "🎯 Target Return (M)",
        min_value=M_min_val,
        max_value=M_max_val,
        value=M_default,
        step=0.1,
        format="%.2f",
    )


# ==================== OPTIMASI ====================
x_opt, risk_opt, return_opt, success = solve_strategic_portfolio(M_target, rets, cov)

if not success:
    st.warning(
        f"Solusi feasible tidak ditemukan untuk target return {M_target:.2f}. Turunkan target return atau periksa ulang matriks kovarians."
    )
    st.stop()


# ==================== LAYOUT HASIL ====================
col_result, col_chart = st.columns([1, 1.65], gap="large")

with col_result:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Hasil Optimasi</div>
            <h2 class="card-title">Alokasi Portofolio</h2>
        """,
        unsafe_allow_html=True,
    )

    alloc_df = pd.DataFrame(
        {
            "Kategori": cats,
            "Bobot": np.round(x_opt, 6),
            "Persentase": [format_percent(value * 100) for value in x_opt],
        }
    )

    st.dataframe(alloc_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Target Return</div>
                    <div class="metric-value">{M_target:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Return Aktual</div>
                    <div class="metric-value">{return_opt:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Risiko Varians</div>
                    <div class="metric-value">{risk_opt:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Aset Dominan</div>
                    <div class="metric-value">{cats[int(np.argmax(x_opt))]}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Komposisi</div>
            <h2 class="card-title">Pie Chart Alokasi</h2>
        """,
        unsafe_allow_html=True,
    )

    fig_pie, ax_pie = plt.subplots(figsize=(4.8, 4.2))
    colors = plt.cm.Set3(np.linspace(0, 1, len(cats)))
    ax_pie.pie(
        x_opt,
        labels=cats,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    ax_pie.set_title("Komposisi Aset", fontsize=12, fontweight="bold")
    fig_pie.tight_layout()
    st.pyplot(fig_pie, use_container_width=True)
    plt.close(fig_pie)

    st.markdown("</div>", unsafe_allow_html=True)

with col_chart:
    st.markdown(
        """
        <div class="section-card">
            <div class="small-label">Efficient Frontier</div>
            <h2 class="card-title">Kurva Risiko dan Imbalan</h2>
        """,
        unsafe_allow_html=True,
    )

    M_vals = np.linspace(M_min_val, M_max_val, 50)
    risks_curve = []
    feasible_returns = []

    for current_target in M_vals:
        weights_loop, risk_loop, return_loop, success_loop = solve_strategic_portfolio(
            current_target, rets, cov
        )
        if success_loop:
            risks_curve.append(risk_loop)
            feasible_returns.append(return_loop)

    fig, ax = plt.subplots(figsize=(9, 5.6))

    if feasible_returns and risks_curve:
        ax.plot(
            feasible_returns,
            risks_curve,
            color="#2563eb",
            linewidth=2.5,
            label="Efficient Frontier",
        )

    ax.scatter(
        return_opt,
        risk_opt,
        color="#dc2626",
        s=120,
        zorder=5,
        label=f"Portofolio Terpilih M={M_target:.2f}",
    )

    ax.annotate(
        f"Return: {return_opt:.3f}\nRisiko: {risk_opt:.3f}",
        xy=(return_opt, risk_opt),
        xytext=(return_opt + 0.10, risk_opt + 0.05),
        fontsize=9,
        color="#991b1b",
        bbox={"boxstyle": "round,pad=0.4", "fc": "#fee2e2", "ec": "#fecaca"},
    )

    ax.set_xlabel("Ekspektasi Return", fontsize=11, fontweight="bold")
    ax.set_ylabel("Risiko Portofolio (Varians)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"""
        <div class="info-box">
            Titik merah menunjukkan portofolio yang dipilih berdasarkan target return.
            Model mencari bobot aset dengan risiko paling rendah selama return aktual tetap memenuhi target.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================== FOOTER ====================
st.markdown(
    """
    <div class="footer-text">
        Model: Mean-Variance Optimization dengan SLSQP | Tampilan: CSS custom bergaya Tailwind
    </div>
    """,
    unsafe_allow_html=True,
)
