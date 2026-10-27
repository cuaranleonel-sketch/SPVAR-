# =============================================================================
# LuzEspacial – SpVAR Colombia (Streamlit Cloud)
# ITAED (DANE) + VIIRS desde CSV + mapas de impacto
# =============================================================================
import os
import tempfile
import warnings
import traceback

warnings.filterwarnings("ignore")
try:
    cmap = plt.colormaps[cmap_name]          # matplotlib >= 3.7
except Exception:
    cmap = plt.cm.get_cmap(cmap_name)        # versiones antiguas
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm
from libpysal.weights import KNN, Queen, lag_spatial
import folium
from folium.features import GeoJsonTooltip
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import gdown
except ImportError:
    gdown = None

# ---------- Config página ----------
st.set_page_config(
    page_title="LuzEspacial",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ITAED_FILE_ID = "1SYu1r-3MbUvmQ5WzoH_uKs5pCAzcEr0P"
ITAED_PATH = "Resultados_ITAED.xlsx"
VIIRS_CSV = "viirs_departamental.csv"
LAGS = 2

# ---------- Estilos ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #070b14; }
h1, h2, h3, p, label, span { color: #e2e8f0 !important; }
div[data-testid="stSidebar"] {
  background: #0f1623;
  border-right: 1px solid #1e293b;
}
.lz-header {
  text-align: center; padding: 1.2rem 1rem 1rem;
  background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(34,211,238,0.12), transparent);
  border-bottom: 1px solid #1e293b; margin-bottom: 1rem; border-radius: 0 0 16px 16px;
}
.lz-title {
  font-size: 2.1rem; font-weight: 700;
  background: linear-gradient(100deg, #22d3ee, #34d399, #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.lz-sub { color: #94a3b8; font-size: 0.95rem; margin-top: 0.3rem; }
.lz-pill {
  display: inline-block; margin-top: 0.5rem; padding: 0.2rem 0.7rem;
  border-radius: 999px; font-size: 0.72rem; font-weight: 600;
  color: #67e8f9; background: rgba(34,211,238,0.1);
  border: 1px solid rgba(34,211,238,0.25); text-transform: uppercase;
}
.lz-badge {
  display: inline-block; background: rgba(16,185,129,0.15); color: #6ee7b7;
  border: 1px solid rgba(16,185,129,0.3); padding: 2px 9px;
  border-radius: 999px; font-size: 0.75rem; font-weight: 600;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Utilidades de nombres ----------
def norm_name(s):
    return (
        str(s)
        .strip()
        .title()
        .replace("Bogotá, D.C.", "Bogotá D.C.")
        .replace("Bogota D.C.", "Bogotá D.C.")
        .replace("Archipielago De San Andres Providencia Y Santa Catalina", "San Andrés")
        .replace("Archipiélago De San Andrés", "San Andrés")
    )


# ---------- Geometrías ----------
@st.cache_data(show_spinner="Cargando geometrías…")
def load_gdf():
    urls = [
        "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
    ]
    gdf = None
    for url in urls:
        try:
            gdf = gpd.read_file(url)
            if gdf is not None and not gdf.empty:
                break
        except Exception:
            continue
    if gdf is None or gdf.empty:
        raise RuntimeError("No se pudieron cargar geometrías de Colombia")
    col = next(
        c
        for c in ["NOMBRE_DPT", "DPTO_CNMBR", "NOM_DEP", "NAME_1", "departamento", "DPTO"]
        if c in gdf.columns
    )
    gdf = gdf.rename(columns={col: "departamento"})
    gdf["departamento"] = gdf["departamento"].str.strip().str.title()
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return (
        gdf.to_crs("EPSG:4326")[["departamento", "geometry"]]
        .drop_duplicates("departamento")
        .reset_index(drop=True)
    )


# ---------- ITAED (DANE) ----------
@st.cache_data(show_spinner="Descargando ITAED (DANE)…")
def load_itaed(path=ITAED_PATH):
    if not os.path.exists(path):
        if gdown is None:
            raise RuntimeError("Falta el paquete gdown")
        url = f"https://docs.google.com/spreadsheets/d/{ITAED_FILE_ID}/export?format=xlsx"
        gdown.download(url, path, quiet=True)
    raw = pd.read_excel(path, sheet_name="Cuadro 1", header=None)
    years_row, q_row = raw.iloc[10, 2:].tolist(), raw.iloc[11, 2:].tolist()
    years, cur = [], None
    for y in years_row:
        if pd.notna(y):
            cur = int(str(y).replace("p", "").replace("r", "").strip())
        years.append(cur)
    qmap = {"I": 1, "II": 2, "III": 3, "IV": 4}
    quarters = [qmap.get(str(q).strip(), None) if pd.notna(q) else None for q in q_row]
    data = raw.iloc[13:].copy()
    data = data[data[1].notna()]
    data = data[
        ~data[1]
        .astype(str)
        .str.contains("Fuente|provisional|Resto|Colombia", case=False, na=False)
    ]
    data = data.iloc[:, : 2 + len(years)]
    data.columns = ["codigo", "departamento"] + [f"c{i}" for i in range(len(years))]
    recs = []
    for _, r in data.iterrows():
        dept = str(r["departamento"]).strip().title()
        for i, (yr, q) in enumerate(zip(years, quarters)):
            if yr is None or q is None:
                continue
            try:
                val = float(r[f"c{i}"])
            except Exception:
                continue
            if pd.isna(val):
                continue
            recs.append(
                {"year": yr, "quarter": q, "departamento": dept, "itaed_index": val}
            )
    df = pd.DataFrame(recs)
    df["fecha"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + ((df["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2)
        + "-01"
    )
    df = df.sort_values(["departamento", "fecha"])
    df["eco_ind"] = df.groupby("departamento")["itaed_index"].pct_change() * 100
    return (
        df.dropna(subset=["eco_ind"])
        .reset_index(drop=True)[["fecha", "departamento", "eco_ind", "itaed_index"]]
    )


# ---------- VIIRS desde CSV (sin Earth Engine) ----------
@st.cache_data(show_spinner="Cargando VIIRS…")
def load_viirs_csv(path=VIIRS_CSV):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # normalizar nombres de columnas por si acaso
    df.columns = [c.strip() for c in df.columns]
    if "nl_growth" not in df.columns and "nl_radiance" in df.columns:
        df = df.sort_values(["departamento", "year"])
        df["nl_growth"] = df.groupby("departamento")["nl_radiance"].pct_change() * 100
        df = df.dropna(subset=["nl_growth"])
    return df.reset_index(drop=True)


def to_quarterly(df_nl):
    recs = []
    for _, r in df_nl.iterrows():
        for q in [1, 2, 3, 4]:
            recs.append(
                {
                    "year": int(r["year"]),
                    "quarter": q,
                    "departamento": r["departamento"],
                    "nl_growth": r["nl_growth"],
                    "nl_radiance": r["nl_radiance"],
                }
            )
    df = pd.DataFrame(recs)
    df["fecha"] = pd.to_datetime(
        df["year"].astype(str)
        + "-"
        + ((df["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2)
        + "-01"
    )
    return df


# ---------- SpVAR ----------
class SpVAR:
    def __init__(self, df, w, lags=2, endog=None):
        self.df, self.w, self.lags = df, w, lags
        self.endog = endog or ["nl_growth", "eco_ind"]
        self.N = w.n
        self.depts = sorted(df["departamento"].unique())
        self.coefs_ = {}
        self._prep()

    def _prep(self):
        self.Y, self.Ylag, self.WYlag = {}, {}, {}
        for v in self.endog:
            Y = self.df.pivot(index="fecha", columns="departamento", values=v)[
                self.depts
            ].values
            self.Y[v] = Y[self.lags :]
            for ell in range(1, self.lags + 1):
                self.Ylag[f"{v}_L{ell}"] = Y[self.lags - ell : (-ell if ell else None)]
                self.WYlag[f"W_{v}_L{ell}"] = np.array(
                    [
                        lag_spatial(self.w, Y[t])
                        for t in range(self.lags - ell, Y.shape[0] - ell)
                    ]
                )
        self.Teff = self.Y[self.endog[0]].shape[0]

    def fit(self):
        for tgt in self.endog:
            y = self.Y[tgt].ravel(order="F")
            X, names = [np.ones((self.Teff * self.N, 1))], ["const"]
            for v in self.endog:
                for ell in range(1, self.lags + 1):
                    X += [
                        self.Ylag[f"{v}_L{ell}"].ravel(order="F")[:, None],
                        self.WYlag[f"W_{v}_L{ell}"].ravel(order="F")[:, None],
                    ]
                    names += [f"{v}_L{ell}", f"W_{v}_L{ell}"]
            m = sm.OLS(y, np.hstack(X)).fit(cov_type="HC1")
            self.coefs_[tgt] = pd.Series(m.params, index=names)
        K, p = len(self.endog), self.lags
        As = []
        for ell in range(1, p + 1):
            A = np.zeros((K, K))
            for i, t in enumerate(self.endog):
                for j, s in enumerate(self.endog):
                    A[i, j] = self.coefs_[t].get(f"{s}_L{ell}", 0) + self.coefs_[
                        t
                    ].get(f"W_{s}_L{ell}", 0)
            As.append(A)
        C = np.zeros((K * p, K * p))
        for i, A in enumerate(As):
            C[:K, i * K : (i + 1) * K] = A
        if p > 1:
            C[K:, :-K] = np.eye(K * (p - 1))
        self.maxeig = float(np.max(np.abs(np.linalg.eigvals(C))))
        return self


def spirf(sp, shock="nl_growth", idx=0, H=12):
    K, N, p = len(sp.endog), sp.N, sp.lags
    hist = np.zeros((p, K, N))
    sh = np.zeros((K, N))
    sh[sp.endog.index(shock), idx] = 1.0
    irf = {v: np.zeros((H + 1, N)) for v in sp.endog}
    for h in range(H + 1):
        cur = sh.copy() if h == 0 else np.zeros((K, N))
        if h > 0:
            for ell in range(1, min(p, h) + 1):
                for i, t in enumerate(sp.endog):
                    for j, s in enumerate(sp.endog):
                        cur[i] += (
                            sp.coefs_[t].get(f"{s}_L{ell}", 0) * hist[ell - 1, j]
                        )
                        cur[i] += sp.coefs_[t].get(f"W_{s}_L{ell}", 0) * lag_spatial(
                            sp.w, hist[ell - 1, j]
                        )
        for i, v in enumerate(sp.endog):
            irf[v][h] = cur[i]
        hist = np.roll(hist, 1, axis=0)
        hist[0] = cur
    return irf


# ---------- Mapas ----------
def folium_html(m, height=560):
    path = tempfile.mktemp(suffix=".html")
    m.save(path)
    with open(path, encoding="utf-8") as f:
        return f.read(), height


def build_choropleth(gdf, values_dict, col_name, cmap_name, alias, center_dept=None):
    import matplotlib as mpl

    g = gdf.copy()
    g[col_name] = g["departamento"].map(values_dict)
    m = folium.Map(location=[4.5, -74.0], zoom_start=5.7, tiles="CartoDB dark_matter")
    vmin, vmax = g[col_name].min(), g[col_name].max()

    try:
        cmap = mpl.colormaps[cmap_name]
    except Exception:
        cmap = mpl.cm.get_cmap(cmap_name)

    def style(feat):
        val = feat["properties"].get(col_name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            c = "#333333"
        else:
            norm = 0.5 if vmax == vmin else (float(val) - vmin) / (vmax - vmin + 1e-12)
            c = mcolors.to_hex(cmap(np.clip(norm, 0, 1)))
        return {
            "fillColor": c,
            "color": "#22d3ee",
            "weight": 1.4,
            "fillOpacity": 0.88,
        }

    folium.GeoJson(
        g.__geo_interface__,
        style_function=style,
        tooltip=GeoJsonTooltip(
            fields=["departamento", col_name], aliases=["Departamento", alias]
        ),
    ).add_to(m)

    if center_dept and center_dept in list(g["departamento"]):
        cent = g.loc[g["departamento"] == center_dept, "geometry"].centroid.iloc[0]
        folium.Marker(
            [cent.y, cent.x],
            popup=f"Shock: {center_dept}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    return folium_html(m)

    def style(feat):
        val = feat["properties"].get(col_name)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            c = "#333333"
        else:
            norm = 0.5 if vmax == vmin else (float(val) - vmin) / (vmax - vmin + 1e-12)
            c = mcolors.to_hex(cmap(np.clip(norm, 0, 1)))
        return {
            "fillColor": c,
            "color": "#22d3ee",
            "weight": 1.4,
            "fillOpacity": 0.88,
        }

    folium.GeoJson(
        g.__geo_interface__,
        style_function=style,
        tooltip=GeoJsonTooltip(
            fields=["departamento", col_name], aliases=["Departamento", alias]
        ),
    ).add_to(m)
    if center_dept and center_dept in list(g["departamento"]):
        cent = g.loc[g["departamento"] == center_dept, "geometry"].centroid.iloc[0]
        folium.Marker(
            [cent.y, cent.x],
            popup=f"Shock: {center_dept}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    return folium_html(m)


def make_spirf_fig(model, shock_dept, H):
    depts = model.depts
    if shock_dept not in depts:
        shock_dept = depts[0]
    idx = depts.index(shock_dept)
    irf = spirf(model, "nl_growth", idx, H)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Luces nocturnas (nl_growth)", "Actividad económica (eco_ind)"],
        vertical_spacing=0.12,
    )
    colors = ["#22d3ee", "#34d399", "#a78bfa", "#f472b6", "#fbbf24", "#fb923c"]
    for r, (var, mat) in enumerate(irf.items(), 1):
        top = np.argsort(np.abs(mat).sum(0))[-6:][::-1]
        for j, i in enumerate(top):
            fig.add_trace(
                go.Scatter(
                    x=list(range(H + 1)),
                    y=mat[:, i],
                    name=depts[i],
                    line=dict(
                        width=4 if depts[i] == shock_dept else 2,
                        color=colors[j % 6],
                    ),
                ),
                row=r,
                col=1,
            )
    fig.update_layout(
        height=560,
        template="plotly_dark",
        paper_bgcolor="#070b14",
        plot_bgcolor="#0f1623",
        title=f"SpIRF · Shock de luces en {shock_dept}",
        legend=dict(orientation="h", y=-0.08),
        font=dict(color="#e2e8f0"),
    )
    return fig, irf


# ---------- Pipeline ----------
@st.cache_resource(show_spinner="Estimando SpVAR…")
def run_pipeline():
    gdf_all = load_gdf()
    eco = load_itaed()

    # VIIRS desde CSV (no usa Earth Engine)
    nl = load_viirs_csv(VIIRS_CSV)
    if nl.empty:
        raise RuntimeError(
            f"No se encontró {VIIRS_CSV} en el repositorio. "
            "Genera el archivo en Colab y súbelo a la raíz del repo."
        )

    nlq = to_quarterly(nl)

    for d in (eco, nlq, gdf_all):
        d["departamento"] = d["departamento"].apply(norm_name)

    df = pd.merge(
        eco,
        nlq[["fecha", "departamento", "nl_growth", "nl_radiance"]],
        on=["fecha", "departamento"],
        how="inner",
    )
    df = df.drop_duplicates(subset=["fecha", "departamento"])
    df = df[df["departamento"].isin(set(gdf_all["departamento"]))]
    df = df.sort_values(["departamento", "fecha"]).reset_index(drop=True)

    depts = sorted(df["departamento"].unique())
    gdf = gdf_all[gdf_all["departamento"].isin(depts)].copy().reset_index(drop=True)

    gdf_p = gdf.to_crs(epsg=3116)
    try:
        w = KNN.from_dataframe(gdf_p, k=min(4, max(1, len(gdf_p) - 1)))
    except Exception:
        w = Queen.from_dataframe(gdf_p, silence_warnings=True)
    w.transform = "r"

    model = SpVAR(df, w, lags=LAGS).fit()
    return model, gdf, df


# ---------- UI ----------
st.markdown(
    """
<div class="lz-header">
  <div class="lz-title">🛰️ LuzEspacial</div>
  <p class="lz-sub">VAR Espacial · Luces Nocturnas VIIRS + Actividad Económica (Colombia)</p>
  <div class="lz-pill">Datos reales · DANE ITAED + NOAA VIIRS</div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    model, gdf, df = run_pipeline()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.code(traceback.format_exc())
    st.stop()

with st.sidebar:
    st.markdown("### Parámetros del shock")
    shock = st.selectbox("Departamento del shock", model.depts, index=0)
    horizon = st.slider("Horizonte (trimestres)", 6, 16, 12)
    st.markdown("### Resumen")
    st.markdown(
        f'<span class="lz-badge">✓ SpVAR</span> max|λ| = **{model.maxeig:.3f}** · n = **{model.N}**',
        unsafe_allow_html=True,
    )
    st.caption(f"Shock: **{shock}**")
    st.markdown("---")
    st.caption("Spatial VAR (no Panel VAR)")
    st.caption("Fuentes: DANE ITAED + NOAA VIIRS")

fig, irf = make_spirf_fig(model, shock, horizon)
impact = {model.depts[i]: float(irf["eco_ind"][:, i].sum()) for i in range(model.N)}
growth = df.groupby("departamento")["nl_growth"].mean().to_dict()
rad = df.groupby("departamento")["nl_radiance"].mean().to_dict()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1. SpIRF",
        "2. Impacto económico",
        "3. Crecimiento luces",
        "4. Radiancia (calor)",
    ]
)

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        html_i, h = build_choropleth(
            gdf, impact, "impacto", "RdYlGn", "Impacto acum. eco", shock
        )
        components.html(html_i, height=h, scrolling=False)
    with c2:
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    html_i, h = build_choropleth(
        gdf, impact, "impacto", "RdYlGn", "Impacto acum. eco", shock
    )
    components.html(html_i, height=h, scrolling=False)

with tab3:
    html_g, h = build_choropleth(gdf, growth, "nl_growth", "YlOrRd", "Crec. luces %")
    components.html(html_g, height=h, scrolling=False)

with tab4:
    html_r, h = build_choropleth(gdf, rad, "nl_radiance", "inferno", "Radiancia")
    components.html(html_r, height=h, scrolling=False)

st.markdown("---")
st.caption("LuzEspacial · Spatial VAR · DANE (ITAED) + NOAA VIIRS")
