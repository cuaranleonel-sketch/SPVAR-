# =============================================================================
# LuzEspacial – SpVAR Colombia (Streamlit) v5
# - Mapas 100% gratuitos (OpenStreetMap)
# - Barra de calor vertical (Colorbar)
# - Layout vertical: un mapa por pestaña a lo ancho
# - Asistente con nueva librería google-genai (compatible con claves AQ.)
# - Traceback de errores desactivado para una UI limpia
# =============================================================================
import os
import io
import warnings
import traceback

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from libpysal.weights import KNN, Queen, lag_spatial
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import gdown
except ImportError:
    gdown = None

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ---------- Configuración Global ----------
st.set_page_config(
    page_title="LuzEspacial",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 🧹 Desactivar el texto técnico de error al final de la app
st.set_option('client.showErrorDetails', False)

ITAED_FILE_ID = "1SYu1r-3MbUvmQ5WzoH_uKs5pCAzcEr0P"
ITAED_PATH = "Resultados_ITAED.xlsx"
VIIRS_CSV = "viirs_departamental.csv"
LUCES_PNG = "luces_colombia_2021.png"
LAGS = 2

# ---------- Estilos ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #070b14; }
h1, h2, h3, h4, p, label, span { color: #e2e8f0 !important; }
div[data-testid="stSidebar"] { background: #0f1623; border-right: 1px solid #1e293b; }

.lz-header {
  text-align: center; padding: 1.4rem 1rem 1.1rem;
  background: radial-gradient(ellipse 80% 60% at 50% -20%, rgba(34,211,238,0.15), transparent);
  border-bottom: 1px solid #1e293b; margin-bottom: 1.2rem; border-radius: 0 0 20px 20px;
}
.lz-title {
  font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em;
  background: linear-gradient(100deg, #22d3ee, #34d399, #a78bfa);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.lz-sub { color: #94a3b8; font-size: 0.95rem; margin-top: 0.35rem; }
.lz-pill {
  display: inline-block; margin-top: 0.6rem; padding: 0.25rem 0.8rem;
  border-radius: 999px; font-size: 0.72rem; font-weight: 600;
  color: #67e8f9; background: rgba(34,211,238,0.1);
  border: 1px solid rgba(34,211,238,0.25); text-transform: uppercase; letter-spacing: 0.04em;
}
.lz-badge {
  display: inline-block; background: rgba(16,185,129,0.15); color: #6ee7b7;
  border: 1px solid rgba(16,185,129,0.3); padding: 3px 10px;
  border-radius: 999px; font-size: 0.75rem; font-weight: 600;
}
.lz-card {
  background: #0f1623; border: 1px solid #1e293b; border-radius: 14px;
  padding: 1rem 1.1rem; margin-bottom: 1rem;
}
.lz-card h4 { margin: 0 0 0.4rem 0; font-size: 0.95rem; color: #67e8f9 !important; }
.lz-card p { margin: 0; font-size: 0.85rem; color: #94a3b8 !important; }
</style>
""",
    unsafe_allow_html=True,
)


def norm_name(s):
    return (
        str(s).strip().title()
        .replace("Bogotá, D.C.", "Bogotá D.C.")
        .replace("Bogota D.C.", "Bogotá D.C.")
        .replace("Archipielago De San Andres Providencia Y Santa Catalina", "San Andrés")
        .replace("Archipiélago De San Andrés", "San Andrés")
    )


# ---------- Datos ----------
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
        c for c in ["NOMBRE_DPT", "DPTO_CNMBR", "NOM_DEP", "NAME_1", "departamento", "DPTO"]
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
        ~data[1].astype(str).str.contains("Fuente|provisional|Resto|Colombia", case=False, na=False)
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
            recs.append({"year": yr, "quarter": q, "departamento": dept, "itaed_index": val})
    df = pd.DataFrame(recs)
    df["fecha"] = pd.to_datetime(
        df["year"].astype(str) + "-" + ((df["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
    )
    df = df.sort_values(["departamento", "fecha"])
    df["eco_ind"] = df.groupby("departamento")["itaed_index"].pct_change() * 100
    return df.dropna(subset=["eco_ind"]).reset_index(drop=True)[
        ["fecha", "departamento", "eco_ind", "itaed_index"]
    ]


@st.cache_data(show_spinner="Cargando VIIRS…")
def load_viirs_csv(path=VIIRS_CSV):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
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
            recs.append({
                "year": int(r["year"]),
                "quarter": q,
                "departamento": r["departamento"],
                "nl_growth": r["nl_growth"],
                "nl_radiance": r["nl_radiance"],
            })
    df = pd.DataFrame(recs)
    df["fecha"] = pd.to_datetime(
        df["year"].astype(str) + "-" + ((df["quarter"] - 1) * 3 + 1).astype(str).str.zfill(2) + "-01"
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
            Y = self.df.pivot(index="fecha", columns="departamento", values=v)[self.depts].values
            self.Y[v] = Y[self.lags:]
            for ell in range(1, self.lags + 1):
                self.Ylag[f"{v}_L{ell}"] = Y[self.lags - ell: (-ell if ell else None)]
                self.WYlag[f"W_{v}_L{ell}"] = np.array(
                    [lag_spatial(self.w, Y[t]) for t in range(self.lags - ell, Y.shape[0] - ell)]
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
                    A[i, j] = self.coefs_[t].get(f"{s}_L{ell}", 0) + self.coefs_[t].get(f"W_{s}_L{ell}", 0)
            As.append(A)
        C = np.zeros((K * p, K * p))
        for i, A in enumerate(As):
            C[:K, i * K:(i + 1) * K] = A
        if p > 1:
            C[K:, :-K] = np.eye(K * (p - 1))
        self.maxeig = float(np.max(np.abs(np.linalg.eigvals(C))))
        return self


def spirf(sp, shock="nl_growth", idx=0, H=12):
    K, N, p = len(sp.endog), sp.N, sp.lags
    A = []
    for ell in range(1, p + 1):
        M = np.zeros((K, K))
        for i, t in enumerate(sp.endog):
            for j, s in enumerate(sp.endog):
                M[i, j] = (
                    sp.coefs_[t].get(f"{s}_L{ell}", 0.0)
                    + sp.coefs_[t].get(f"W_{s}_L{ell}", 0.0)
                )
        A.append(M)

    hist = [np.zeros((K, N)) for _ in range(p)]
    sh = np.zeros((K, N))
    sh[sp.endog.index(shock), idx] = 1.0
    irf = {v: np.zeros((H + 1, N)) for v in sp.endog}

    response = sh.copy()
    for h in range(H + 1):
        if h > 0:
            response = np.zeros((K, N))
            for ell in range(1, min(p, h) + 1):
                past = hist[ell - 1]
                for i in range(K):
                    for j in range(K):
                        response[i] += A[ell - 1][i, j] * past[j]
        for i, v in enumerate(sp.endog):
            irf[v][h] = response[i]
        hist = [response] + hist[:-1]
    return irf


# ---------- Mapas ----------
def folium_html(m, height=600):
    return m.get_root().render(), height


def build_choropleth(gdf, values_dict, col_name, cmap_name, alias, center_dept=None):
    g = gdf.copy()
    g[col_name] = g["departamento"].map(values_dict)

    # OpenStreetMap: Gratis, sin API Key y funciona siempre
    m = folium.Map(
        location=[4.5, -74.0], 
        zoom_start=5.5, 
        tiles="OpenStreetMap", 
        zoom_control=True,
        control_scale=True
    )

    vals = g[col_name].dropna()
    vmin = float(vals.min()) if len(vals) else 0.0
    vmax = float(vals.max()) if len(vals) else 1.0

    palettes = {
        "RdYlGn": ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850"],
        "YlOrRd": ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        "inferno": ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
    }
    palette = palettes.get(cmap_name, palettes["YlOrRd"])

    def value_to_color(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "#1e293b"  # Color para regiones sin datos
        t = 0.5 if vmax == vmin else (float(val) - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))
        n = len(palette) - 1
        i = min(int(t * n), n - 1)
        f = (t * n) - i
        c1, c2 = palette[i].lstrip("#"), palette[i + 1].lstrip("#")
        r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
        r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)
        r = int(r1 + (r2 - r1) * f)
        gg = int(g1 + (g2 - g1) * f)
        b = int(b1 + (b2 - b1) * f)
        return f"#{r:02x}{gg:02x}{b:02x}"

    def style(feat):
        val = feat["properties"].get(col_name)
        return {
            "fillColor": value_to_color(val),
            "color": "#334155",
            "weight": 1,
            "fillOpacity": 0.9 if pd.notna(val) else 0.3,
        }

    folium.GeoJson(
        g.__geo_interface__,
        style_function=style,
        tooltip=GeoJsonTooltip(fields=["departamento", col_name], aliases=["Departamento", alias]),
    ).add_to(m)

    if center_dept and center_dept in set(g["departamento"]):
        cent = g.loc[g["departamento"] == center_dept, "geometry"].centroid.iloc[0]
        folium.Marker(
            [cent.y, cent.x],
            popup=f"Shock: {center_dept}",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    # Encuadre a Colombia completa
    bounds = g.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Barra de calor VERTICAL
    colormap = cm.LinearColormap(colors=palette, vmin=vmin, vmax=vmax, caption=alias)
    m.add_child(colormap)

    return folium_html(m, height=650)


def make_spirf_fig(model, shock_dept, H):
    depts = model.depts
    if shock_dept not in depts:
        shock_dept = depts[0]
    idx = depts.index(shock_dept)
    irf = spirf(model, "nl_growth", idx, H)
    fig = make_subplots(
        rows=2, cols=1,
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
                    line=dict(width=4 if depts[i] == shock_dept else 2, color=colors[j % 6]),
                ),
                row=r, col=1,
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
    nl = load_viirs_csv(VIIRS_CSV)
    if nl.empty:
        raise RuntimeError(
            f"No se encontró {VIIRS_CSV}. Genera el CSV en Colab y súbelo a la raíz del repo."
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


# ---------- Asistente Gemini (google-genai) ----------
def render_assistant(model, df, shock):
    st.subheader("🤖 Asistente LuzEspacial")
    st.caption("Pregunta sobre el modelo, los resultados o la interpretación de los mapas.")

    if not GEMINI_AVAILABLE:
        st.warning("Falta `google-genai` en el entorno. Revisa tu requirements.txt.")
        return

    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        st.info(
            "🔑 Configura `GEMINI_API_KEY` en **Settings → Secrets** de Streamlit Cloud "
            "para activar el asistente."
        )
        return

    # Inicializar el nuevo cliente de Google
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        st.error("Hubo un problema al inicializar la IA. Revisa tu API Key.")
        return

    depts = model.depts
    system_prompt = f"""Eres un asistente experto en econometría espacial que ayuda a interpretar
los resultados de la aplicación LuzEspacial (SpVAR aplicado a Colombia).

Contexto del modelo actual:
- Spatial VAR (SpVAR) con {model.N} departamentos de Colombia.
- Variables endógenas: nl_growth (crecimiento de luces VIIRS) y eco_ind (DANE ITAED).
- Rezagos: {model.lags} trimestres.
- Raíz característica máxima: {model.maxeig:.3f} (si es >= 1, el VAR no es estacionario).
- Período de datos: {df['fecha'].min():%Y-%m} a {df['fecha'].max():%Y-%m}.
- Shock actual: {shock}.
- Departamentos: {', '.join(depts[:12])}{'...' if len(depts) > 12 else ''}.

Responde en español, claro y conciso. Si te falta información, dilo honestamente."""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Pregunta sobre el modelo…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                # Usar la nueva API compatible con claves "AQ."
                response = client.models.generate_content_stream(
                    model="gemini-2.5-flash",
                    contents=f"{system_prompt}\n\nPregunta del usuario: {prompt}",
                )
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            except Exception:
                # Ocultamos el error técnico de Python y mostramos un mensaje amigable
                full_response = "⚠️ Lo siento, en este momento el asistente no puede responder. Por favor, verifica tu API Key o intenta de nuevo en unos minutos."
                placeholder.warning(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})


# ---------- UI principal ----------
st.markdown(
    """
<div class="lz-header">
  <div class="lz-title">🛰️ LuzEspacial</div>
  <p class="lz-sub">VAR Espacial · Luces Nocturnas VIIRS + Actividad Económica (Colombia)</p>
  <div class="lz-pill">DANE ITAED + NOAA VIIRS</div>
</div>
""",
    unsafe_allow_html=True,
)

try:
    model, gdf, df = run_pipeline()
except Exception as e:
    st.error(f"Error al cargar los datos. Intenta recargar la página.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Parámetros del shock")
    shock = st.selectbox("Departamento del shock", model.depts, index=0)
    horizon = st.slider("Horizonte (trimestres)", 6, 16, 12)
    st.markdown("### Resumen del modelo")
    st.markdown(
        f'<span class="lz-badge">✓ SpVAR</span> max|λ| = **{model.maxeig:.3f}** · n = **{model.N}**',
        unsafe_allow_html=True,
    )
    if model.maxeig >= 1.0:
        st.warning("⚠️ El VAR no es estacionario (max|λ| ≥ 1). Interpreta con cautela.")
    st.caption(f"Shock: **{shock}**")
    st.markdown("---")
    st.caption("Spatial VAR (no Panel VAR)")
    st.caption("Fuentes: DANE ITAED + NOAA VIIRS")

# Cálculos
fig, irf = make_spirf_fig(model, shock, horizon)
impact = {model.depts[i]: float(irf["eco_ind"][:, i].sum()) for i in range(model.N)}
growth = df.groupby("departamento")["nl_growth"].mean().to_dict()
rad = df.groupby("departamento")["nl_radiance"].mean().to_dict()

# Tabs
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Luces satelitales",
    "🗺️ Mapas de Impacto",
    "📈 SpIRF",
    "🌡️ Crecimiento y radiancia",
    "🤖 Asistente",
])

# --- Tab 0: Imagen satelital ---
with tab0:
    st.subheader("Luces nocturnas satelitales (VIIRS 2021)")
    if os.path.exists(LUCES_PNG):
        st.image(LUCES_PNG, use_container_width=True)
        st.caption("Composite anual VIIRS 2021 · NOAA / Earth Engine")
    else:
        st.warning(f"No se encontró **{LUCES_PNG}** en el repositorio.")
        st.info("Las pestañas de mapas sí usan datos VIIRS reales del CSV.")

# --- Tab 1: Mapas individuales ---
with tab1:
    st.subheader("Análisis Geográfico del Shock")
    st.caption(f"Shock en **{shock}** · Horizonte {horizon} trimestres · Selecciona una métrica para visualizar en el mapa.")

    mapa_opcion = st.radio(
        "Selecciona la métrica a visualizar:",
        ["Impacto Económico Acumulado", "Crecimiento Promedio de Luces", "Radiancia Promedio", "Ubicación del Shock"],
        horizontal=True
    )

    st.markdown("---")

    if mapa_opcion == "Impacto Económico Acumulado":
        st.markdown('<div class="lz-card"><h4>Impacto acumulado en actividad económica</h4><p>Suma del SpIRF de eco_ind en el horizonte.</p></div>', unsafe_allow_html=True)
        html_impact, h = build_choropleth(gdf, impact, "impacto", "RdYlGn", "Impacto eco", shock)
        components.html(html_impact, height=h, scrolling=False)

    elif mapa_opcion == "Crecimiento Promedio de Luces":
        st.markdown('<div class="lz-card"><h4>Crecimiento promedio de luces</h4><p>Media de nl_growth por departamento en la muestra.</p></div>', unsafe_allow_html=True)
        html_growth, h = build_choropleth(gdf, growth, "nl_growth", "YlOrRd", "Crec. luces %")
        components.html(html_growth, height=h, scrolling=False)

    elif mapa_opcion == "Radiancia Promedio":
        st.markdown('<div class="lz-card"><h4>Radiancia promedio</h4><p>Media de nl_radiance por departamento.</p></div>', unsafe_allow_html=True)
        html_rad, h = build_choropleth(gdf, rad, "nl_radiance", "inferno", "Radiancia")
        components.html(html_rad, height=h, scrolling=False)

    elif mapa_opcion == "Ubicación del Shock":
        st.markdown('<div class="lz-card"><h4>Ubicación del shock</h4><p>Departamento que recibe el shock actualmente.</p></div>', unsafe_allow_html=True)
        g_shock = gdf[gdf["departamento"] == shock]
        
        m_shock = folium.Map(location=[4.5, -74.0], zoom_start=5.5, tiles="OpenStreetMap")
        folium.GeoJson(
            gdf.__geo_interface__,
            style_function=lambda x: {'fillColor': 'transparent', 'color': '#22d3ee', 'weight': 1.5, 'fillOpacity': 0.1}
        ).add_to(m_shock)
        
        if not g_shock.empty:
            cent = g_shock.geometry.centroid.iloc[0]
            folium.Marker(
                [cent.y, cent.x],
                popup=f"Shock: {shock}",
                icon=folium.Icon(color="red", icon="info-sign"),
            ).add_to(m_shock)
            
        bounds = gdf.total_bounds
        m_shock.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        html_shock, h = folium_html(m_shock, height=650)
        components.html(html_shock, height=h, scrolling=False)

    st.markdown("---")
    buf = io.StringIO()
    pd.DataFrame(list(impact.items()), columns=["departamento", "impacto_eco"]).to_csv(buf, index=False)
    st.download_button(
        "⬇️ Descargar impacto por departamento (CSV)",
        buf.getvalue(),
        f"impacto_{shock}.csv",
        "text/csv",
    )

# --- Tab 2: SpIRF ---
with tab2:
    st.subheader(f"SpIRF · Shock en {shock}")
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Crecimiento y radiancia individuales ---
with tab3:
    st.subheader("Crecimiento de luces y radiancia")
    
    st.markdown("### Crecimiento promedio de luces (%)")
    html_g2, h = build_choropleth(gdf, growth, "nl_growth", "YlOrRd", "Crec. luces %")
    components.html(html_g2, height=h, scrolling=False)
    
    st.markdown("---")
    
    st.markdown("### Radiancia promedio")
    html_r2, h = build_choropleth(gdf, rad, "nl_radiance", "inferno", "Radiancia")
    components.html(html_r2, height=h, scrolling=False)

# --- Tab
