"""
================================================================================
SMART PRODUCT CATEGORY ASSISTANT
Aplikasi Segmentasi Produk Retail dengan K-Means Clustering
================================================================================
NIM         : 38250016
Mata Kuliah : Pemrograman Kecerdasan Buatan (AIB02)
Universitas : Universitas Bunda Mulia
================================================================================
Alur Notebook → App:
  [1] Load Dataset  →  Tab Dataset (preview + stats)
  [2] Preprocessing →  Tab Metodologi (scaler info)
  [3] Elbow / Sil   →  Tab Evaluasi  (live charts)
  [4] Training      →  Tab Evaluasi  (model metrics)
  [5] Profiling     →  Tab Visualisasi (profil cluster)
  [6] Visualisasi   →  Tab Visualisasi (scatter + charts)
  [7] Save Model    →  (sudah dimuat via .pkl)
  [8] Prediksi Baru →  Tab Prediksi
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Product Category Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0f14;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1e2330;
    min-width: 350px !important; /* Tambahkan baris ini, sesuaikan angkanya jika kurang lebar */
    max-width: 450px !important; /* Opsional: agar user tidak bisa menariknya terlalu lebar */
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* ── Main canvas ── */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Hero header ── */
.hero-wrap {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0d1117 100%);
    border: 1px solid #1e2330;
    border-radius: 20px;
    padding: 2.2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #6366f130 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f1f5f9;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-title span { color: #818cf8; }
.hero-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0;
    font-weight: 300;
}
.badge-row {
    display: flex;
    gap: 0.6rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    background: #1e2330;
    border: 1px solid #2d3748;
    color: #94a3b8;
    letter-spacing: 0.5px;
}
.badge.accent { background: #1e1b4b; border-color: #4338ca; color: #818cf8; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #111318;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #1e2330;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 0.5rem 1.1rem;
    border-radius: 8px;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1e2330 !important;
    color: #e2e8f0 !important;
}

/* ── Section header ── */
.sec-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.3rem 0;
}
.sec-sub {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0 0 1.5rem 0;
}

/* ── Stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.8rem; margin-bottom: 1.5rem; }
.stat-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.75rem;
    font-weight: 800;
    color: #f1f5f9;
    display: block;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.stat-label { font-size: 0.72rem; color: #475569; letter-spacing: 0.8px; text-transform: uppercase; }

/* ── Cluster cards ── */
.cluster-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    margin-bottom: 0.8rem;
    border-left-width: 4px;
}
.cluster-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
}
.cluster-desc { font-size: 0.85rem; color: #94a3b8; margin: 0; line-height: 1.5; }

/* ── Strategy list ── */
.strat-item {
    display: flex;
    gap: 0.8rem;
    align-items: flex-start;
    padding: 0.6rem 0.8rem;
    background: #0d0f14;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border: 1px solid #1e2330;
}
.strat-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #4f46e5;
    background: #1e1b4b;
    border-radius: 50%;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.strat-text { font-size: 0.85rem; color: #94a3b8; line-height: 1.5; }

/* ── Info / result box ── */
.result-box {
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    border: 1px solid;
    margin-bottom: 1.2rem;
}

/* ── Metric override ── */

/* ── 1. KOTAK METRIK DI HALAMAN UTAMA (MAIN CONTENT) ── */
[data-testid="stAppViewContainer"] section:not([data-testid="stSidebar"]) [data-testid="stMetric"] {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 0.7rem 0.3rem !important; 
    text-align: center !important; 
    
    /* KUNCI BARU: Gunakan min-height untuk memaksa tinggi seragam */
    min-height: 110px !important; 
    
    display: flex !important; 
    flex-direction: column !important;
    justify-content: center !important; 
}

/* Memastikan konten di dalam kotak benar-benar di tengah */
[data-testid="stAppViewContainer"] section:not([data-testid="stSidebar"]) [data-testid="stMetric"] > div {
    display: flex;
    flex-direction: column;
    justify-content: center !important;
    align-items: center !important;
    height: 100%;
}

[data-testid="stAppViewContainer"] section:not([data-testid="stSidebar"]) [data-testid="stMetricLabel"] { 
    font-size: 0.65rem !important; 
    color: #475569 !important; 
    text-transform: uppercase; 
    letter-spacing: 0.5px; 
    line-height: 1.2 !important;
    margin-bottom: 0.2rem !important;
}

[data-testid="stAppViewContainer"] section:not([data-testid="stSidebar"]) [data-testid="stMetricLabel"] div {
    white-space: pre-line !important; /* Membaca \n sebagai enter */
    word-break: keep-all !important; 
    text-align: center !important;
}

[data-testid="stAppViewContainer"] section:not([data-testid="stSidebar"]) [data-testid="stMetricValue"] { 
    font-family: 'Syne', sans-serif !important; 
    font-size: 0.9rem !important; 
    color: #f1f5f9 !important; 
}

[data-testid="stAppViewContainer"] section:not([data-testid="stSidebar"]) [data-testid="stMetricValue"] div {
    white-space: nowrap !important; 
    overflow: visible !important;   
    text-overflow: clip !important; 
}


/* ── 2. KOTAK METRIK DI SIDEBAR (RINGKASAN DATASET) ── */
/* CSS ini KHUSUS mengunci metrik di Sidebar agar kembali ke posisi rata kiri asli yang rapi */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 12px;
    padding: 0.8rem 1.0rem !important; /* Jarak padding lebih longgar karena area sidebar luas */
    text-align: left !important; /* Dikunci agar TETAP RATA KIRI sesuai keinginanmu */
}

[data-testid="stSidebar"] [data-testid="stMetric"] > div {
    justify-content: flex-start !important;
    align-items: flex-start !important;
}

[data-testid="stSidebar"] [data-testid="stMetricLabel"] { 
    font-size: 0.7rem !important; 
    color: #475569 !important; 
    text-transform: uppercase; 
    letter-spacing: 0.5px; 
    margin-bottom: 0.3rem !important;
}

[data-testid="stSidebar"] [data-testid="stMetricValue"] { 
    font-family: 'Syne', sans-serif !important; 
    font-size: 1.2rem !important; /* Ukuran font lebih besar agar proporsional di sidebar */
    color: #f1f5f9 !important; 
}
            
/* ── Dataframe dark style ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; border: 1px solid #1e2330; }

/* ── Selectbox / input dark ── */
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
    background: #111318 !important;
    border-color: #2d3748 !important;
    color: #e2e8f0 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
hr { border-color: #1e2330 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d0f14; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 10px; }

/* ── Notebook step pill ── */
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #1a1f2e;
    border: 1px solid #2d3748;
    border-radius: 8px;
    padding: 0.3rem 0.9rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #818cf8;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB DARK THEME (global)
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#111318",
    "axes.facecolor":    "#0d0f14",
    "axes.edgecolor":    "#2d3748",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#e2e8f0",
    "grid.color":        "#1e2330",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "legend.facecolor":  "#111318",
    "legend.edgecolor":  "#2d3748",
    "legend.labelcolor": "#94a3b8",
    "figure.dpi":        120,
    "font.family":       "sans-serif",
})

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
FEATURES   = ["Units Sold", "Unit Price", "Total Revenue"]
PALETTE    = {0: "#818cf8", 1: "#f59e0b", 2: "#10b981"}   # selaras notebook [6]
MARKERS    = {0: "o", 1: "s", 2: "^"}
CLUSTER_BG = {0: "#1e1b4b", 1: "#451a03", 2: "#064e3b"}
CLUSTER_BD = {0: "#4f46e5", 1: "#d97706", 2: "#059669"}


# ─────────────────────────────────────────────
#  HELPER: GET CLUSTER META  (sama dengan notebook [5])
# ─────────────────────────────────────────────
def get_cluster_meta(cluster_id, profil):
    """
    Pemetaan label sesuai logika notebook [5]:
      price > median & units < median → PREMIUM STARS
      units > median                  → VOLUME DRIVERS
      else                            → STANDARD ITEMS
    """
    default = {
        "label": f"Cluster {cluster_id}",
        "icon": "⚪",
        "desc": "Informasi tidak tersedia.",
        "strategi": [],
        "color": CLUSTER_BD[cluster_id % 3],
        "bg":    CLUSTER_BG[cluster_id % 3],
    }
    if profil is None:
        return default
    try:
        price = profil.loc[cluster_id, "Unit Price"]
        units = profil.loc[cluster_id, "Units Sold"]
        med_price = profil["Unit Price"].median()
        med_units = profil["Units Sold"].median()

        if price > med_price and units < med_units:
            return {
                "label": "PREMIUM STARS",
                "icon": "💎",
                "desc": "Produk harga tinggi dengan volume terbatas. Revenue per-unit tertinggi — segmen pelanggan premium.",
                "strategi": [
                    "Pertahankan eksklusivitas; hindari diskon besar-besaran.",
                    "Targetkan kampanye ke segmen pelanggan premium secara spesifik.",
                    "Tawarkan layanan purna jual (after-sales) & garansi extended.",
                    "Pantau kompetitor di segmen harga serupa secara rutin.",
                    "Buat paket bundle eksklusif dengan produk premium lain.",
                ],
                "color": CLUSTER_BD[0],
                "bg":    CLUSTER_BG[0],
            }
        elif units > med_units:
            return {
                "label": "VOLUME DRIVERS",
                "icon": "🔥",
                "desc": "Motor penggerak revenue dengan volume penjualan tertinggi. Produk laris yang menarik traffic utama.",
                "strategi": [
                    "Pastikan ketersediaan stok selalu terjaga — hindari stockout.",
                    "Gunakan sebagai anchor product untuk menarik traffic ke toko.",
                    "Terapkan bulk-discount untuk mendorong volume lebih tinggi.",
                    "Bundling dengan produk Standard Items untuk cross-sell.",
                    "Monitor tren harian dan musiman secara ketat.",
                ],
                "color": CLUSTER_BD[1],
                "bg":    CLUSTER_BG[1],
            }
        else:
            return {
                "label": "STANDARD ITEMS",
                "icon": "🛒",
                "desc": "Produk performa rata-rata. Memerlukan optimasi strategi untuk mendorong pertumbuhan.",
                "strategi": [
                    "Tingkatkan visibilitas lewat placement lebih strategis.",
                    "Buat bundling dengan produk Volume Drivers.",
                    "Terapkan diskon atau promosi musiman.",
                    "Optimalkan deskripsi, foto, dan ulasan produk.",
                    "Analisis kompetitor untuk produk serupa di pasar.",
                ],
                "color": CLUSTER_BD[2],
                "bg":    CLUSTER_BG[2],
            }
    except Exception:
        return default


# ─────────────────────────────────────────────
#  LOAD FUNCTIONS  (notebook [1] & [7])
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    try:
        with open("kmeans_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_data():
    try:
        return pd.read_csv("dataset_clustered.csv")
    except FileNotFoundError:
        try:
            return pd.read_csv("Online Sales Data.csv")
        except FileNotFoundError:
            return None


@st.cache_data
def load_profil():
    try:
        return pd.read_csv("profil_cluster.csv", index_col="Cluster")
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────
#  EVALUATION HELPER  (notebook [3] & [4])
# ─────────────────────────────────────────────
def compute_silhouette(df):
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        X = df[FEATURES].dropna().values
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        labels = df.loc[df[FEATURES].dropna().index, "Cluster"].values
        return round(silhouette_score(Xs, labels), 4)
    except Exception:
        return None


@st.cache_data
def compute_elbow_sil(df):
    from sklearn.cluster import KMeans as KM
    from sklearn.preprocessing import StandardScaler as SS
    from sklearn.metrics import silhouette_score as ss_

    # Sesuai notebook [3]: hanya 2 fitur — Units Sold & Unit Price
    X = df[["Units Sold", "Unit Price"]].dropna().values
    sc = SS(); Xs = sc.fit_transform(X)

    inertias, sil_scores = [], []
    # K=1 inertia (only for elbow chart, tidak bisa hitung silhouette)
    km1 = KM(n_clusters=1, init="k-means++", random_state=42, n_init=10).fit(Xs)
    inertias.append(km1.inertia_)

    for k in range(2, 11):
        km = KM(n_clusters=k, init="k-means++", random_state=42, n_init=5)
        lbl = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sil_scores.append(ss_(Xs, lbl))

    return inertias, sil_scores   # inertias[0]=k1, sil_scores[0]=k2


# ─────────────────────────────────────────────
#  INIT
# ─────────────────────────────────────────────
model, scaler = load_model_and_scaler()
df    = load_data()
profil = load_profil()

if df is not None and "Cluster" not in df.columns and model is not None and scaler is not None:
    X = df[FEATURES].dropna()
    df = df.loc[X.index].copy()
    df["Cluster"] = model.predict(scaler.transform(X))

cluster_counts = df["Cluster"].value_counts().sort_index() if df is not None else None
sil_score = compute_silhouette(df) if df is not None else None


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1rem 0;">
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;color:#f1f5f9;">
            Smart Product<br>Category Assistant
        </div>
        <div style="font-size:0.72rem;color:#475569;margin-top:0.2rem;font-family:'DM Mono',monospace;">
            K-MEANS CLUSTERING
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if df is not None:
        st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.7rem;">RINGKASAN DATASET</div>', unsafe_allow_html=True)

        cols_sb = st.columns(2)
        cols_sb[0].metric("Total Produk", f"{len(df):,}")
        cols_sb[1].metric("Cluster", "3")

        if sil_score:
            st.metric("Silhouette Score", f"{sil_score:.4f}",
                      delta="Baik ✓" if sil_score > 0.5 else "Cukup",
                      help=">0.5 = cluster terpisah dengan baik")

        if model is not None:
            st.metric("Inertia (WCSS)", f"{model.inertia_:,.1f}")
            st.metric("Iterasi Konvergen", model.n_iter_)

        st.divider()
        st.markdown('<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.7rem;">DISTRIBUSI CLUSTER</div>', unsafe_allow_html=True)

        for cid, count in cluster_counts.items():
            meta = get_cluster_meta(cid, profil)
            pct = count / len(df) * 100
            st.markdown(f"""
            <div style="background:{meta['bg']};border:1px solid {meta['color']}40;
                border-left:3px solid {meta['color']};border-radius:10px;
                padding:0.6rem 0.8rem;margin-bottom:0.5rem;">
                <div style="font-weight:600;font-size:0.82rem;color:{meta['color']};">
                    {meta['icon']} {meta['label']}
                </div>
                <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">
                    {count} produk · {pct:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:0.72rem;color:#334155;text-align:center;line-height:1.6;">
        NIM: 38250016<br>
        AIB02 · Universitas Bunda Mulia
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
if df is None:
    st.error("❌ Dataset tidak ditemukan. Pastikan `dataset_clustered.csv` ada di direktori yang sama.")
    st.stop()

st.markdown("""
<div class="hero-wrap">
    <p class="hero-title">🛒 Smart Product <span>Category</span> Assistant</p>
    <p class="hero-sub">Sistem segmentasi produk retail berbasis K-Means Clustering · AIB02 Pemrograman Kecerdasan Buatan</p>
    <div class="badge-row">
        <span class="badge accent">NIM: 38250016</span>
        <span class="badge">K-Means · K=3</span>
        <span class="badge">StandardScaler</span>
        <span class="badge">Scikit-Learn</span>
        <span class="badge">300 Transaksi</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Analisis Produk",
    "🎯 Prediksi Baru",
    "📊 Visualisasi",
    "📈 Evaluasi Model",
    "📋 Dataset",
])


# ═══════════════════════════════════════════════
#  TAB 1 — ANALISIS PRODUK EXISTING
# ═══════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-pill">📓 Notebook [5][6] → Profiling & Analisis Produk</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">Analisis Produk Existing</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Pilih produk dari dataset untuk melihat cluster dan rekomendasi strategi bisnis.</p>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
        all_categories = ["Semua Kategori"] + sorted(df["Product Category"].unique().tolist())
        kat = st.selectbox("Filter Kategori", all_categories)

        df_kat = df if kat == "Semua Kategori" else df[df["Product Category"] == kat]
        produk_pilihan = st.selectbox("🛍️ Pilih Produk", sorted(df_kat["Product Name"].unique()))

        if produk_pilihan:
            row = df[df["Product Name"] == produk_pilihan].iloc[0]
            cid = int(row["Cluster"])

            st.markdown("---")
            c1, c2, c3 = st.columns(3)

            # Tambahkan \n setelah emoji agar teks terdorong ke bawah
            c1.metric("📦\nUnits Sold",      f"{row['Units Sold']:,.0f}")
            c2.metric("💰\nUnit Price",      f"${row['Unit Price']:,.2f}")
            c3.metric("💵\nTotal Revenue",   f"${row['Total Revenue']:,.2f}")

            st.markdown("""<div style="background:#111318;border:1px solid #1e2330;border-radius:12px;padding:1rem;margin-top:1rem;">""", unsafe_allow_html=True)
            d1, d2 = st.columns(2)
            d1.markdown(f"**Kategori** \n{row['Product Category']}")
            d1.markdown(f"**Region** \n{row['Region']}")
            d2.markdown(f"**Metode Bayar** \n{row.get('Payment Method','–')}")
            d2.markdown(f"**Tanggal** \n{row.get('Date','–')}")
            st.markdown("</div>", unsafe_allow_html=True)

            if profil is not None:
                # ── 1. Perbandingan terhadap Rata-rata Dataset ──
                st.markdown("<div style='margin-top:1.5rem;font-size:0.85rem;color:#818cf8;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;text-align:center;'>Perbandingan Performa terhadap Rata-rata</div>", unsafe_allow_html=True)
                for feat in FEATURES:
                    avg_val = df[feat].mean()
                    val     = row[feat]
                    delta   = (val - avg_val) / avg_val * 100
                    
                    if delta >= 0:
                        if feat == "Units Sold": ket = "lebih laris"
                        elif feat == "Unit Price": ket = "lebih mahal"
                        else: ket = "omset naik"
                        delta_text = f"+{delta:.1f}% ({ket})"
                    else:
                        if feat == "Units Sold": ket = "lebih sepi"
                        elif feat == "Unit Price": ket = "lebih murah"
                        else: ket = "omset turun"
                        delta_text = f"-{abs(delta):.1f}% ({ket})"
                        
                    st.metric(feat, f"{val:,.1f}", delta_text)

                # ── 2. Perbandingan terhadap Centroid Cluster ──
                st.markdown(f"<div style='margin-top:1.5rem;font-size:0.85rem;color:#818cf8;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;text-align:center;'>Jarak terhadap Centroid Cluster {cid}</div>", unsafe_allow_html=True)
                for feat in FEATURES:
                    centroid_val = profil.loc[cid, feat]
                    val          = row[feat]
                    delta_c      = (val - centroid_val) / centroid_val * 100 if centroid_val != 0 else 0
                    
                    # Logika label dan kata-kata informatif berbasis teori clustering
                    if feat == "Units Sold":
                        display_label = "Units Sold (Pusat Cluster)"
                        ket_c = "Melampaui Target Kelompok" if delta_c >= 0 else "Di Bawah Target Kelompok"
                    elif feat == "Unit Price":
                        display_label = "Unit Price (Karakteristik Cluster)"
                        ket_c = "Premium di Kelasnya" if delta_c >= 0 else "Ekonomis di Kelasnya"
                    else:
                        display_label = "Total Revenue (Acuan Cluster)"
                        ket_c = "Kontributor Utama Cluster" if delta_c >= 0 else "Kontributor Lemah Cluster"
                    
                    # Format teks persenan
                    if delta_c >= 0:
                        delta_c_text = f"+{delta_c:.1f}% ({ket_c})"
                    else:
                        delta_c_text = f"-{abs(delta_c):.1f}% ({ket_c})"
                        
                    st.metric(display_label, f"{val:,.1f}", delta_c_text)

    with col_r:
        if produk_pilihan:
            meta = get_cluster_meta(cid, profil)
            st.markdown(f"""
            <div class="cluster-card" style="border-left-color:{meta['color']};background:{meta['bg']};">
                <p class="cluster-title" style="color:{meta['color']};">
                    {meta['icon']} Cluster {cid} — {meta['label']}
                </p>
                <p class="cluster-desc">{meta['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="font-size:0.8rem;color:#475569;text-transform:uppercase;letter-spacing:0.6px;margin:1rem 0 0.5rem;">💡 Rekomendasi Strategi</div>', unsafe_allow_html=True)
            for i, s in enumerate(meta["strategi"], 1):
                st.markdown(f"""
                <div class="strat-item">
                    <div class="strat-num">{i}</div>
                    <div class="strat-text">{s}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── [GRAFIK 1 - ATAS] Comparison bar chart (Terhadap Seluruh Cluster) ──
            if profil is not None:
                st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
                fig1, axes1 = plt.subplots(1, 3, figsize=(10, 3))
                input_vals = [row[f] for f in FEATURES]
                short_lbl  = ["Units Sold", "Unit Price", "Revenue"]

                for i, (feat, val, short) in enumerate(zip(FEATURES, input_vals, short_lbl)):
                    ax = axes1[i]
                    cvals = [profil.loc[c, feat] for c in sorted(profil.index)]
                    clrs  = [PALETTE[c] for c in sorted(profil.index)]
                    bars  = ax.bar([f"C{c}" for c in sorted(profil.index)],
                                   cvals, color=clrs, alpha=0.7, edgecolor="#2d3748", linewidth=0.7)
                    ax.axhline(val, color=meta["color"], linewidth=2, linestyle="--", label="Produk ini")
                    ax.set_title(short, fontsize=9, fontweight="bold")
                    ax.tick_params(labelsize=8)
                    ax.legend(fontsize=7)
                    ax.grid(axis="y")

                fig1.suptitle(f"{produk_pilihan[:35]} — Perbandingan Performa terhadap Rata-rata Kelompok", fontsize=10, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig1)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)

                # ── [GRAFIK 2 - BAWAH] Centroid Proximity Chart (Posisi Spesifik vs Centroid Clusternya) ──
                st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
                fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.2))
                
                for i, (feat, val, short) in enumerate(zip(FEATURES, input_vals, short_lbl)):
                    ax = axes2[i]
                    centroid_val = profil.loc[cid, feat]
                    
                    # Membuat visualisasi komparasi 1-on-1: Nilai Produk vs Nilai Centroid Kelompoknya
                    bars = ax.bar(["Nilai Produk", "Pusat Cluster"], [val, centroid_val], 
                                  color=[meta["color"], "#475569"], alpha=0.8, edgecolor="#2d3748", linewidth=0.7)
                    
                    # Menambahkan label angka di atas setiap bar agar informatif saat dosen melihat nilai eksaknya
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                                f"{height:,.1f}" if short == "Units Sold" else f"${height:,.1f}",
                                ha='center', va='bottom', fontsize=7, color="#94a3b8")
                                
                    ax.set_title(short, fontsize=9, fontweight="bold")
                    ax.tick_params(labelsize=8)
                    ax.grid(axis="y")
                
                fig2.suptitle(f"Posisi Detil '{produk_pilihan[:25]}' vs Centroid Akurasi Kelompok", fontsize=10, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
                st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  TAB 2 — PREDIKSI PRODUK BARU  (notebook [8])
# ═══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="step-pill">📓 Notebook [8] → Prediksi Produk Baru</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">Prediksi Cluster Produk Baru</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Masukkan atribut produk baru — sistem akan memprediksi segmen dan strategi yang tepat.</p>', unsafe_allow_html=True)

    if model is None or scaler is None:
        st.warning("⚠️ Model belum dimuat. Pastikan `kmeans_model.pkl` dan `scaler.pkl` ada.")
    else:
        # Referensi dari notebook [8]: test_samples
        col_inp, col_res = st.columns([1, 1.3], gap="large")

        with col_inp:
            nama_baru = st.text_input("Nama Produk (opsional)", placeholder="Contoh: Smart Watch Pro X")

            mn_u, mx_u = int(df["Units Sold"].min()), int(df["Units Sold"].max())
            mn_p, mx_p = float(df["Unit Price"].min()), float(df["Unit Price"].max())
            mn_r, mx_r = float(df["Total Revenue"].min()), float(df["Total Revenue"].max())

            units_in = st.number_input("📦 Units Sold", min_value=0, max_value=mx_u*3,
                                       value=int((mn_u+mx_u)/2), step=1,
                                       help=f"Rentang data: {mn_u} – {mx_u}")
            price_in = st.number_input("💰 Unit Price (USD)", min_value=0.0, max_value=mx_p*3,
                                       value=round((mn_p+mx_p)/2, 2), step=0.01, format="%.2f",
                                       help=f"Rentang data: ${mn_p:.2f} – ${mx_p:.2f}")
            rev_auto = units_in * price_in
            rev_in   = st.number_input("💵 Total Revenue (USD)", min_value=0.0, max_value=mx_r*3,
                                       value=float(rev_auto), step=0.01, format="%.2f",
                                       help=f"Otomatis: Units × Price = ${rev_auto:,.2f}")

            if abs(rev_in - rev_auto) > 0.01:
                st.caption(f"💡 Units × Price = ${rev_auto:,.2f} (Revenue bisa disesuaikan manual)")

            predict_btn = st.button("🔍 Prediksi Sekarang", use_container_width=True)

            # Quick reference table (sesuai notebook [8] test_samples)
            if profil is not None:
                st.markdown("<div style='margin-top:1.5rem;font-size:0.8rem;color:#475569;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.6rem;'>Referensi Centroid</div>", unsafe_allow_html=True)
                for cid in sorted(profil.index):
                    m2 = get_cluster_meta(cid, profil)
                    st.markdown(f"""
                    <div style="background:{m2['bg']};border-left:3px solid {m2['color']};
                         border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;">
                        <b style="color:{m2['color']};font-size:0.82rem;">{m2['icon']} C{cid}: {m2['label']}</b>
                        <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">
                        Units: {profil.loc[cid,'Units Sold']:.0f} &nbsp;|&nbsp;
                        Price: ${profil.loc[cid,'Unit Price']:.0f} &nbsp;|&nbsp;
                        Rev: ${profil.loc[cid,'Total Revenue']:,.0f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# --- SISI KANAN: OUTPUT PREDIKSI ---
        with col_res:
            if predict_btn:
                X_new    = np.array([[units_in, price_in, rev_in]])
                X_scaled = scaler.transform(X_new)
                pred_cid = int(model.predict(X_scaled)[0])
                meta_p   = get_cluster_meta(pred_cid, profil)
                label_p  = nama_baru if nama_baru else "Produk Baru"

                st.markdown(f"""
                <div class="result-box" style="background:{meta_p['bg']};border-color:{meta_p['color']};">
                    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:700;color:{meta_p['color']};">
                        ✅ {meta_p['icon']} {label_p} → {meta_p['label']}
                    </div>
                    <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.4rem;">{meta_p['desc']}</div>
                </div>
                """, unsafe_allow_html=True)

                input_vals = [units_in, price_in, rev_in]
                # Membuat mapping dictionary agar pengambilan nilai di dalam loop akurat
                input_dict = dict(zip(FEATURES, input_vals))

                # ── A. PERBANDINGAN TERHADAP RATA-RATA DATASET (Struktur Linear) ──
                st.markdown("<div style='margin-top:1.5rem;font-size:0.85rem;color:#818cf8;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;text-align:center;'>Perbandingan Performa terhadap Rata-rata</div>", unsafe_allow_html=True)
                
                for feat in FEATURES:
                    avg_val = df[feat].mean()
                    val     = input_dict[feat]
                    delta   = (val - avg_val) / avg_val * 100
                    
                    if delta >= 0:
                        if feat == "Units Sold": ket = "lebih laris"
                        elif feat == "Unit Price": ket = "lebih mahal"
                        else: ket = "omset naik"
                        delta_text = f"+{delta:.1f}% ({ket})"
                    else:
                        if feat == "Units Sold": ket = "lebih sepi"
                        elif feat == "Unit Price": ket = "lebih murah"
                        else: ket = "omset turun"
                        delta_text = f"-{abs(delta):.1f}% ({ket})"
                        
                    st.metric(feat, f"{val:,.1f}", delta_text)

                # ── B. JARAK TERHADAP CENTROID CLUSTER (Struktur Linear) ──
                st.markdown(f"<div style='margin-top:1.5rem;font-size:0.85rem;color:#818cf8;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;text-align:center;'>Jarak terhadap Centroid Cluster {pred_cid}</div>", unsafe_allow_html=True)
                
                for feat in FEATURES:
                    centroid_val = profil.loc[pred_cid, feat]
                    val          = input_dict[feat]
                    delta_c      = (val - centroid_val) / centroid_val * 100 if centroid_val != 0 else 0
                    
                    # Menggunakan label analitik informatif pilihanmu
                    if feat == "Units Sold":
                        display_label = "Units Sold (Pusat Cluster)"
                        ket_c = "Melampaui Target Kelompok" if delta_c >= 0 else "Di Bawah Target Kelompok"
                    elif feat == "Unit Price":
                        display_label = "Unit Price (Karakteristik Cluster)"
                        ket_c = "Premium di Kelasnya" if delta_c >= 0 else "Ekonomis di Kelasnya"
                    else:
                        display_label = "Total Revenue (Acuan Cluster)"
                        ket_c = "Kontributor Utama Cluster" if delta_c >= 0 else "Kontributor Lemah Cluster"
                    
                    delta_c_text = f"{'+' if delta_c >= 0 else '-'}{abs(delta_c):.1f}% ({ket_c})"
                    st.metric(display_label, f"{val:,.1f}", delta_c_text)

                # ── C. STRATEGI REKOMENDASI BISNIS ──
                st.markdown('<div style="margin-top:1.5rem;font-size:0.8rem;color:#475569;text-transform:uppercase;letter-spacing:0.6px;margin-bottom:0.5rem;">💡 Rekomendasi Strategi</div>', unsafe_allow_html=True)
                for i, s in enumerate(meta_p["strategi"], 1):
                    st.markdown(f"""
                    <div class="strat-item">
                        <div class="strat-num">{i}</div>
                        <div class="strat-text">{s}</div>
                    </div>
                    """, unsafe_allow_html=True)

               # ── D. VISUALISASI DIAGRAM BATANG (DUA BARIS: GLOBAL & SPESIFIK) ──
                if profil is not None:
                    # 📊 GRAFIK 1 (ATAS): Posisi Input vs Seluruh Cluster Dataset
                    st.markdown("<div style='margin-top:1.5rem;'>", unsafe_allow_html=True)
                    fig1, axes1 = plt.subplots(1, 3, figsize=(10, 3.2))
                    for i, (feat, val) in enumerate(zip(FEATURES, input_vals)):
                        ax = axes1[i]
                        cvals = [profil.loc[c, feat] for c in sorted(profil.index)]
                        ax.bar([f"C{c}" for c in sorted(profil.index)], cvals,
                               color=[PALETTE[c] for c in sorted(profil.index)],
                               alpha=0.65, edgecolor="#2d3748", linewidth=0.7)
                        ax.axhline(val, color=meta_p["color"], linewidth=2.2,
                                   linestyle="--", label=f"{label_p[:15]}")
                        ax.set_title(feat.replace(" ", "\n"), fontsize=8.5, fontweight="bold")
                        ax.tick_params(labelsize=8)
                        ax.legend(fontsize=7)
                        ax.grid(axis="y")
                    fig1.suptitle(f"{label_p[:30]} - Perbandingan Performa terhadap Rata-rata Kelompok", fontsize=10, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig1)
                    plt.close()
                    st.markdown("</div>", unsafe_allow_html=True)

                    # 🎯 GRAFIK 2 (BAWAH): Komparasi Langsung Jarak Detil vs Pusat Clusternya
                    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
                    fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.4))
                    short_lbl = ["Units Sold", "Unit Price", "Revenue"]
                    
                    for i, (feat, val, short) in enumerate(zip(FEATURES, input_vals, short_lbl)):
                        ax = axes2[i]
                        centroid_val = profil.loc[pred_cid, feat]
                        
                        # Membuat diagram batang komparasi langsung produk vs centroid
                        bars = ax.bar(["Nilai Produk", "Pusat Cluster"], [val, centroid_val], 
                                      color=[meta_p["color"], "#475569"], alpha=0.8, edgecolor="#2d3748", linewidth=0.7)
                        
                        # Menambahkan teks angka nilai presisi tepat di atas balok grafik
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + (height * 0.02),
                                    f"{height:,.1f}" if short == "Units Sold" else f"${height:,.1f}",
                                    ha='center', va='bottom', fontsize=7, color="#94a3b8")
                                    
                        ax.set_title(short, fontsize=8.5, fontweight="bold")
                        ax.tick_params(labelsize=8)
                        ax.grid(axis="y")
                    
                    fig2.suptitle(f"Posisi Detil '{label_p[:25]}' vs Centroid Akurasi Kelompok", fontsize=10, fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close()
                    st.markdown("</div>", unsafe_allow_html=True)

            else:
                st.markdown("""
                <div style="background:#111318;border:1px solid #1e2330;border-radius:14px;
                     padding:3rem;text-align:center;color:#475569;">
                    <div style="font-size:2.5rem;margin-bottom:0.8rem;">🤖</div>
                    <div style="font-size:0.9rem;">Masukkan data produk di sebelah kiri<br>dan klik <b>Prediksi Sekarang</b></div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  TAB 3 — VISUALISASI  (notebook [5][6])
# ═══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="step-pill">📓 Notebook [5][6] → Profiling & Visualisasi Cluster</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">Visualisasi Hasil Clustering</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Scatter plot, distribusi, dan profil rata-rata setiap cluster.</p>', unsafe_allow_html=True)

    if df is not None and profil is not None:
        # ── KPI strip ──
        cols_kpi = st.columns(3)
        for i, (cid, count) in enumerate(cluster_counts.items()):
            meta = get_cluster_meta(cid, profil)
            pct  = count / len(df) * 100
            with cols_kpi[i]:
                st.markdown(f"""
                <div style="background:{meta['bg']};border:1px solid {meta['color']}40;
                     border-radius:14px;padding:1.1rem;text-align:center;">
                    <div style="font-size:1.8rem;">{meta['icon']}</div>
                    <div style="font-family:'Syne',sans-serif;font-weight:700;
                         color:{meta['color']};font-size:0.9rem;margin:0.3rem 0;">{meta['label']}</div>
                    <div style="font-size:1.7rem;font-weight:800;color:#f1f5f9;">{count}</div>
                    <div style="font-size:0.72rem;color:#64748b;">{pct:.1f}% dari total</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'/>", unsafe_allow_html=True)

        # ── Row 1: Scatter (sesuai notebook [6]: Units Sold vs Unit Price) ──
        col_s1, col_s2 = st.columns(2)

        centroids_orig = scaler.inverse_transform(model.cluster_centers_) if model and scaler else None

        with col_s1:
            st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:0.4rem;">SCATTER — Units Sold vs Unit Price</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for cid in sorted(df["Cluster"].unique()):
                meta = get_cluster_meta(cid, profil)
                mask = df["Cluster"] == cid
                ax.scatter(df.loc[mask, "Units Sold"], df.loc[mask, "Unit Price"],
                           c=PALETTE[cid], label=f"C{cid}: {meta['label']}",
                           alpha=0.65, s=55, marker=MARKERS[cid], edgecolors="#0d0f14", linewidths=0.5)
            if centroids_orig is not None:
                ax.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
                           c="white", marker="X", s=220, zorder=5, label="Centroid", edgecolors="#2d3748")
            ax.set_xlabel("Units Sold"); ax.set_ylabel("Unit Price ($)")
            ax.set_title("Units Sold vs Unit Price", fontweight="bold", fontsize=11)
            ax.legend(fontsize=8); ax.grid(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_s2:
            st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:0.4rem;">SCATTER — Unit Price vs Total Revenue</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            for cid in sorted(df["Cluster"].unique()):
                meta = get_cluster_meta(cid, profil)
                mask = df["Cluster"] == cid
                ax.scatter(df.loc[mask, "Unit Price"], df.loc[mask, "Total Revenue"],
                           c=PALETTE[cid], label=f"C{cid}: {meta['label']}",
                           alpha=0.65, s=55, marker=MARKERS[cid], edgecolors="#0d0f14", linewidths=0.5)
            if centroids_orig is not None:
                ax.scatter(centroids_orig[:, 1], centroids_orig[:, 2],
                           c="white", marker="X", s=220, zorder=5, label="Centroid", edgecolors="#2d3748")
            ax.set_xlabel("Unit Price ($)"); ax.set_ylabel("Total Revenue ($)")
            ax.set_title("Unit Price vs Total Revenue", fontweight="bold", fontsize=11)
            ax.legend(fontsize=8); ax.grid(True)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Row 2: Pie + Profil bar ──
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:0.4rem;">DISTRIBUSI PER CLUSTER</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5.5, 5.5))
            labels_pie = []
            for cid in cluster_counts.index:
                meta = get_cluster_meta(cid, profil)
                labels_pie.append(f"C{cid}: {meta['label']}\n({meta['icon']})")
            wedge_props = {"linewidth": 1.5, "edgecolor": "#0d0f14"}
            ax.pie(cluster_counts.values, labels=labels_pie, autopct="%1.1f%%",
                   colors=[PALETTE[c] for c in cluster_counts.index],
                   explode=[0.04]*len(cluster_counts), shadow=False, startangle=90,
                   textprops={"fontsize": 9, "color": "#e2e8f0"},
                   wedgeprops=wedge_props)
            ax.set_title("Distribusi Produk per Cluster", fontweight="bold", fontsize=11)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_p2:
            st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:0.4rem;">PROFIL RATA-RATA CLUSTER</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 4.8))
            x = np.arange(len(profil)); w = 0.25
            b1 = ax.bar(x - w, profil["Units Sold"],    w, label="Units Sold",    color="#818cf8", alpha=0.85, edgecolor="#2d3748")
            b2 = ax.bar(x,     profil["Unit Price"],    w, label="Unit Price",    color="#f59e0b", alpha=0.85, edgecolor="#2d3748")
            b3 = ax.bar(x + w, profil["Total Revenue"], w, label="Total Revenue", color="#10b981", alpha=0.85, edgecolor="#2d3748")
            for bars in [b1, b2, b3]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                            f"{h:.0f}", ha="center", va="bottom", fontsize=7, color="#94a3b8")
            ax.set_xticks(x)
            ax.set_xticklabels([f"Cluster {i}" for i in profil.index])
            ax.set_title("Perbandingan Profil Cluster", fontweight="bold", fontsize=11)
            ax.legend(fontsize=9); ax.grid(axis="y")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Row 3: Heatmap + Category bar ──
        col_h1, col_h2 = st.columns(2)

        with col_h1:
            st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:0.4rem;">HEATMAP PROFIL (NORMALIZED)</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            profil_norm = (profil - profil.min()) / (profil.max() - profil.min())
            sns.heatmap(profil_norm, annot=profil.round(1), fmt="g",
                        cmap="Blues", linewidths=0.5, ax=ax,
                        cbar_kws={"label": "Normalized"},
                        annot_kws={"size": 9})
            ax.set_title("Heatmap Profil Cluster (Nilai Asli di Sel)", fontweight="bold", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_h2:
            st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin-bottom:0.4rem;">DISTRIBUSI CLUSTER PER KATEGORI</div>', unsafe_allow_html=True)
            if "Product Category" in df.columns:
                cat_cluster = df.groupby(["Product Category", "Cluster"]).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(7, 3.5))
                cat_cluster.plot(kind="bar", ax=ax,
                                 color=[PALETTE[c] for c in cat_cluster.columns],
                                 edgecolor="#2d3748", alpha=0.85, linewidth=0.7)
                ax.set_title("Jumlah Produk per Kategori & Cluster", fontweight="bold", fontsize=10)
                ax.set_xlabel(""); ax.set_ylabel("Jumlah Produk")
                ax.legend(title="Cluster", fontsize=8)
                ax.tick_params(axis="x", rotation=25)
                ax.grid(axis="y")
                plt.tight_layout(); st.pyplot(fig); plt.close()

        # ── Boxplot ──
        st.markdown('<div style="font-size:0.8rem;color:#818cf8;font-weight:600;margin:0.5rem 0 0.4rem;">BOXPLOT DISTRIBUSI FITUR PER CLUSTER</div>', unsafe_allow_html=True)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for i, feat in enumerate(FEATURES):
            data_box = [df[df["Cluster"] == c][feat].values for c in sorted(df["Cluster"].unique())]
            bp = axes[i].boxplot(data_box, patch_artist=True, notch=False,
                                 medianprops=dict(color="#f1f5f9", linewidth=2))
            for j, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(list(PALETTE.values())[j])
                patch.set_alpha(0.65)
            for whisker in bp["whiskers"]:
                whisker.set(color="#475569", linewidth=1.2)
            for cap in bp["caps"]:
                cap.set(color="#475569", linewidth=1.2)
            axes[i].set_xticklabels([f"C{c}" for c in sorted(df["Cluster"].unique())])
            axes[i].set_title(feat, fontweight="bold", fontsize=10)
            axes[i].set_xlabel("Cluster"); axes[i].grid(axis="y")
        fig.suptitle("Distribusi Fitur per Cluster", fontsize=12, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════
#  TAB 4 — EVALUASI MODEL  (notebook [2][3][4])
# ═══════════════════════════════════════════════
with tab4:
    st.markdown('<div class="step-pill">📓 Notebook [2][3][4] → Preprocessing, Evaluasi K & Training</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">Metodologi & Evaluasi Model</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">Pipeline lengkap: StandardScaler → Elbow → Silhouette → K-Means Training.</p>', unsafe_allow_html=True)

    # ── Metric strip ──
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("🎯 Silhouette Score", f"{sil_score:.4f}" if sil_score else "N/A",
              delta="Baik ✓" if (sil_score and sil_score > 0.5) else "Cukup ⚠" if sil_score else None)
    e2.metric("📉 Inertia (WCSS)", f"{model.inertia_:,.2f}" if model else "N/A")
    e3.metric("🔄 Iterasi", f"{model.n_iter_}" if model else "N/A")
    e4.metric("📦 Total Data", f"{len(df):,}")

    st.markdown("---")

    # ── Theory columns ──
    col_t1, col_t2 = st.columns([1.6, 1], gap="large")

    with col_t1:
        st.markdown("""
### 🧠 Algoritma K-Means Clustering

**K-Means** adalah algoritma *unsupervised learning* yang mengelompokkan data ke K cluster berdasarkan kemiripan karakteristik tanpa memerlukan label.

---
**① Inisialisasi Centroid (K-Means++)** Memilih centroid awal yang saling berjauhan untuk menghindari solusi sub-optimal.

**② Assignment Step** Setiap titik data ditetapkan ke centroid terdekat berdasarkan jarak Euclidean:

$$d(x, c) = \\sqrt{\\sum_{i=1}^{n}(x_i - c_i)^2}$$

**③ Update Step** Centroid diperbarui menjadi rata-rata dari semua titik dalam clusternya:

$$c_k = \\frac{1}{|S_k|}\\sum_{x \\in S_k} x$$

**④ Konvergensi** Ulangi langkah ②–③ hingga posisi centroid tidak berubah signifikan.

---
**🔧 StandardScaler (Preprocessing — Notebook [2])**

K-Means menggunakan jarak Euclidean — fitur berskala besar mendominasi.
StandardScaler menstandarisasi setiap fitur: **mean = 0, std = 1**

$$z = \\frac{x - \\mu}{\\sigma}$$

Fitur yang digunakan (sesuai notebook):
- `Units Sold` — jumlah unit terjual
- `Unit Price` — harga per unit
- `Total Revenue` — total pendapatan
        """)

    with col_t2:
        st.markdown("""
### ⚙️ Parameter Model

| Parameter | Nilai |
|-----------|-------|
| n_clusters | **3** |
| init | **k-means++** |
| random_state | **42** |
| n_init | **10** |
| max_iter | **300** |

---
### 📊 Metrik Evaluasi

**Inertia (WCSS)** Jumlah kuadrat jarak dalam cluster — semakin kecil semakin baik.

**Silhouette Score**

$$s = \\frac{b - a}{\\max(a, b)}$$

| Nilai | Interpretasi |
|-------|-------------|
| > 0.5 | Baik ✅ |
| 0.25–0.5 | Cukup ⚠️ |
| < 0.25 | Kurang ❌ |
        """)

        if sil_score:
            if sil_score > 0.5:
                st.success(f"✅ Silhouette = {sil_score:.4f} — Cluster **terpisah baik**")
            elif sil_score > 0.25:
                st.warning(f"⚠️ Silhouette = {sil_score:.4f} — Cluster **cukup terpisah**")
            else:
                st.error(f"❌ Silhouette = {sil_score:.4f} — Cluster **kurang terpisah**")

    st.markdown("---")

    # ── Centroid table ──
    if model is not None and scaler is not None:
        st.markdown("### 📍 Koordinat Centroid (Skala Asli)")
        centroids_s = model.cluster_centers_
        centroids_o = scaler.inverse_transform(centroids_s)
        centroid_df = pd.DataFrame(centroids_o, columns=FEATURES)
        centroid_df.index.name = "Cluster"
        st.dataframe(
            centroid_df.style.format("{:.2f}").background_gradient(cmap="Blues", axis=0),
            use_container_width=True
        )

    st.markdown("---")

    # ── Elbow + Silhouette charts (notebook [3]) ──
    st.markdown("### 📉 Elbow Method & Silhouette Score per K  _(Notebook [3])_")
    with st.spinner("Menghitung Elbow Method & Silhouette..."):
        inertias, sil_scores_k = compute_elbow_sil(df)

    col_elbow, col_sil = st.columns(2)

    with col_elbow:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        k_all = list(range(1, len(inertias) + 1))
        ax.plot(k_all, inertias, marker="o", color="#818cf8", linewidth=2.5,
                markersize=8, markerfacecolor="#0d0f14", markeredgewidth=2)
        ax.axvline(x=3, color="#10b981", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.scatter([3], [inertias[2]], color="#10b981", s=180, zorder=5,
                   marker="X", label="Elbow: K=3")
        ax.set_title("Elbow Method (Inertia)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Jumlah Cluster (k)"); ax.set_ylabel("Inertia / WCSS")
        ax.legend(fontsize=9); ax.grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_sil:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        k_sil = list(range(2, 2 + len(sil_scores_k)))
        ax.plot(k_sil, sil_scores_k, marker="s", color="#f59e0b", linewidth=2.5,
                markersize=8, markerfacecolor="#0d0f14", markeredgewidth=2)
        best_idx = int(np.argmax(sil_scores_k))
        best_k   = k_sil[best_idx]
        ax.scatter([best_k], [sil_scores_k[best_idx]], color="#10b981", s=180,
                   zorder=5, marker="*", label=f"Best K={best_k} ({sil_scores_k[best_idx]:.3f})")
        ax.axvline(x=3, color="#f43f5e", linestyle="--", alpha=0.6, linewidth=1.5, label="K=3 dipilih")
        ax.set_title("Silhouette Score per K", fontsize=12, fontweight="bold")
        ax.set_xlabel("Jumlah Cluster (k)"); ax.set_ylabel("Silhouette Score")
        ax.legend(fontsize=9); ax.grid(True)
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════
#  TAB 5 — DATASET  (notebook [1])
# ═══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="step-pill">📓 Notebook [1] → Load Dataset</div>', unsafe_allow_html=True)
    st.markdown('<p class="sec-header">Dataset Lengkap</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-sub">300 transaksi retail dengan hasil clustering K-Means.</p>', unsafe_allow_html=True)

    # ── Stats row ──
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Total Baris", f"{len(df):,}")
    s2.metric("Kolom", f"{len(df.columns)}")
    s3.metric("Kategori Produk", f"{df['Product Category'].nunique()}")
    s4.metric("Region", f"{df['Region'].nunique()}")

    st.markdown("---")

    # ── Filters ──
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        cl_filter = st.multiselect("Filter Cluster",
                                    options=sorted(df["Cluster"].unique()),
                                    default=sorted(df["Cluster"].unique()),
                                    format_func=lambda x: f"C{x} – {get_cluster_meta(x, profil)['label']}")
    with cf2:
        cat_filter = st.multiselect("Filter Kategori",
                                     options=sorted(df["Product Category"].unique()),
                                     default=sorted(df["Product Category"].unique()))
    with cf3:
        sort_by = st.selectbox("Urutkan", ["Total Revenue ↓", "Units Sold ↓", "Unit Price ↓", "Cluster"])

    df_f = df.copy()
    if cl_filter:  df_f = df_f[df_f["Cluster"].isin(cl_filter)]
    if cat_filter: df_f = df_f[df_f["Product Category"].isin(cat_filter)]

    sort_map = {
        "Total Revenue ↓": ("Total Revenue", False),
        "Units Sold ↓":    ("Units Sold", False),
        "Unit Price ↓":    ("Unit Price", False),
        "Cluster":         ("Cluster", True),
    }
    sc_col, sc_asc = sort_map[sort_by]
    df_f = df_f.sort_values(sc_col, ascending=sc_asc)

    st.markdown(f"**Menampilkan {len(df_f):,} dari {len(df):,} produk**")

    cols_show = [c for c in ["Product Name","Product Category","Units Sold","Unit Price","Total Revenue","Region","Payment Method","Cluster"] if c in df_f.columns]
    st.dataframe(
        df_f[cols_show].style.format({"Unit Price": "${:.2f}", "Total Revenue": "${:.2f}", "Units Sold": "{:,.0f}"}),
        use_container_width=True, hide_index=True, height=400
    )

    st.markdown("---")
    st.markdown("### 📊 Statistik Deskriptif  _(Notebook [2])_")
    st.dataframe(df_f[FEATURES].describe().round(2), use_container_width=True)

    csv_dl = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV Terfilter", csv_dl, "hasil_cluster.csv", "text/csv", use_container_width=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1.5rem;background:#111318;border:1px solid #1e2330;border-radius:16px;">
    <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;color:#f1f5f9;margin-bottom:0.3rem;">
        🛒 Smart Product Category Assistant
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#475569;">
        NIM: 38250016 &nbsp;·&nbsp; AIB02 Pemrograman Kecerdasan Buatan &nbsp;·&nbsp; Universitas Bunda Mulia<br>
        K-Means Clustering &nbsp;·&nbsp; Scikit-Learn &nbsp;·&nbsp; Streamlit
    </div>
</div>
""", unsafe_allow_html=True)