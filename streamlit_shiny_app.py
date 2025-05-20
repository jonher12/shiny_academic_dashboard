import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# === CONFIGURACIÓN DE PÁGINA ===
st.set_page_config(page_title="Dashboard Estudiantil", layout="wide")

# === FUNCIÓN PARA CARGAR DATOS ===
@st.cache_data
def load_data_from_gdrive(file_id: str) -> pd.DataFrame:
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(BytesIO(response.content), encoding="latin1")
    else:
        st.error(f"Error al descargar archivo: {response.status_code}")
        return pd.DataFrame()

# === CARGAR DATOS ===
FILE_ID = st.secrets["FILE_ID"]
df = load_data_from_gdrive(FILE_ID)

# === VARIABLES ===
demograficas = ["Procedencia", "1st Fall Enrollment", "Índice General", "Índice Científico", "PCAT"]
notas_cursos = [
    "Español Básico Nota 1", "Español Básico Nota 2",
    "Inglés Básico Nota 1", "Inglés Básico Nota 2",
    "Ciencias Sociales Nota 1", "Ciencias Sociales Nota 2",
    "Economía Nota 1", "Economía Nota 2"
]
continuas = ["Índice General", "Índice Científico", "PCAT"]
categoricas = [col for col in df.columns if col not in demograficas + ["Nombre", "Numero de Estudiante", "Email UPR", "Número de Expediente"] and col not in continuas]

# Mapear notas
nota_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
df[notas_cursos] = df[notas_cursos].apply(lambda col: col.map(lambda x: nota_map.get(str(x).strip().upper(), np.nan)))

# === VALORES DEFAULT ===
default_filters = {
    "col_cat": "1st Fall Enrollment",
    "valor_filtro": "All Enrollment",
    "col_proc": "Todas",
    "col_x": "Índice General",
    "col_y": "Índice Científico"
}

# === GESTIÓN DE SESIÓN ===
if "reset_filters" not in st.session_state:
    st.session_state.reset_filters = False

# === SIDEBAR ===
with st.sidebar:
    st.header("📊 Filtros")

    if st.button("🔄 Resetear filtros"):
        for k, v in default_filters.items():
            st.session_state[k] = v
        st.session_state.slider = None

    col_cat = st.selectbox("Filtrar por categoría", categoricas, key="col_cat")
    valores_cat = sorted(df[col_cat].dropna().astype(str).unique())
    if col_cat == "1st Fall Enrollment":
        valores_cat = ["All Enrollment"] + valores_cat
    valor_filtro = st.selectbox(f"Valor en '{col_cat}'", valores_cat, key="valor_filtro")

    col_proc = st.selectbox("Procedencia", ["Todas"] + sorted(df["Procedencia"].dropna().astype(str).unique()), key="col_proc")

    col_x = st.selectbox("Variable continua (eje X)", continuas, key="col_x")
    col_y_options = [c for c in continuas if c != col_x]
    col_y = st.selectbox("Variable continua (eje Y)", col_y_options, key="col_y")

    min_val = float(df[col_x].min())
    max_val = float(df[col_x].max())
    slider_step = 1.0 if col_x == "PCAT" else 0.1
    selected_range = st.slider(
        f"Rango de '{col_x}'",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=slider_step,
        key="slider"
    )

# === FILTRADO ===
df_filtrado = df.copy()
if col_cat == "1st Fall Enrollment" and valor_filtro != "All Enrollment":
    df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == valor_filtro]
elif col_cat != "1st Fall Enrollment":
    df_filtrado = df_filtrado[df_filtrado[col_cat].astype(str) == valor_filtro]

if col_proc != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Procedencia"].astype(str) == col_proc]

df_filtrado = df_filtrado[(df_filtrado[col_x] >= selected_range[0]) & (df_filtrado[col_x] <= selected_range[1])]

# === MÉTRICAS ===
st.markdown("## 📊 Dashboard Estudiantil")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total registros", f"{len(df_filtrado):,}")
c2.metric("Promedio General", f"{df_filtrado['Índice General'].mean():.2f}")
c3.metric("Promedio Científico", f"{df_filtrado['Índice Científico'].mean():.2f}")
c4.metric("Promedio PCAT", f"{df_filtrado['PCAT'].mean():.2f}")

# === HISTOGRAMA ===
hist = go.Figure()
hist.add_trace(go.Histogram(x=df_filtrado[col_x], nbinsx=10, marker_color="#1f77b4"))
hist.update_layout(title=f"Distribución de {col_x}", xaxis_title=col_x, yaxis_title="Frecuencia")

# === BARRAS ===
valores_barras = df_filtrado[col_cat].dropna().astype(str).value_counts().sort_index()
bars = go.Figure()
bars.add_trace(go.Bar(x=valores_barras.index, y=valores_barras.values, marker_color="#2c3e50"))
bars.update_layout(title=f"Distribución de {col_cat}", xaxis_title=col_cat, yaxis_title="Cantidad", xaxis_type='category')

# === MATRIZ DE CORRELACIÓN ===
columnas_cor = notas_cursos + continuas
corr_df = df_filtrado[columnas_cor].replace({pd.NA: np.nan})
matriz = corr_df.corr()

heatmap = go.Figure(data=go.Heatmap(
    z=matriz.values,
    x=matriz.columns,
    y=matriz.index,
    colorscale="Blues",
    zmin=-1,
    zmax=1,
    colorbar=dict(title="Correlación")
))
heatmap.update_layout(
    title="Correlación entre notas y métricas",
    xaxis=dict(tickangle=45, tickfont=dict(size=10), automargin=True),
    yaxis=dict(tickfont=dict(size=10), automargin=True),
    width=1100,
    height=900,
    margin=dict(t=60, l=200, r=50, b=200)
)

# === SCATTER + REGRESIÓN ===
x_vals = df_filtrado[col_x].dropna().values.reshape(-1, 1)
y_vals = df_filtrado[col_y].dropna().values.reshape(-1, 1)
valid_idx = (~np.isnan(x_vals.flatten())) & (~np.isnan(y_vals.flatten()))
x_clean = x_vals[valid_idx].reshape(-1, 1)
y_clean = y_vals[valid_idx].reshape(-1, 1)

model = LinearRegression()
model.fit(x_clean, y_clean)
y_pred = model.predict(x_clean)
r2 = r2_score(y_clean, y_pred)
slope = model.coef_[0][0]
intercept = model.intercept_[0]
equation = f"y = {slope:.2f}x + {intercept:.2f}<br>R² = {r2:.3f}"

scatter = go.Figure()
scatter.add_trace(go.Scatter(x=x_clean.flatten(), y=y_clean.flatten(), mode='markers', name='Datos'))
scatter.add_trace(go.Scatter(x=x_clean.flatten(), y=y_pred.flatten(), mode='lines', name='Regresión', line=dict(color='orange')))
scatter.update_layout(title=f"{col_x} vs {col_y} con regresión<br><sub>{equation}</sub>", xaxis_title=col_x, yaxis_title=col_y)

# === LAYOUT ===
g1, g2 = st.columns(2)
g1.plotly_chart(hist, use_container_width=True)
g2.plotly_chart(bars, use_container_width=True)

g3, g4 = st.columns(2)
g3.plotly_chart(scatter, use_container_width=True)
g4.plotly_chart(heatmap, use_container_width=True)

# === TABLA DE DATOS ===
st.markdown("### 🧾 Tabla de datos filtrados")
columnas_excluir = ["Nombre", "Numero de Estudiante", "Email UPR", "Número de Expediente"]
columnas_mostrar = [col for col in df_filtrado.columns if col not in columnas_excluir]
st.dataframe(df_filtrado[columnas_mostrar])
