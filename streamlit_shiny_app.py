import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Dashboard Estudiantil", layout="wide")

@st.cache_data
def load_data_from_gdrive(file_id: str) -> pd.DataFrame:
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(BytesIO(response.content), encoding="latin1")
    else:
        st.error(f"Error al descargar archivo: {response.status_code}")
        return pd.DataFrame()

FILE_ID = st.secrets["FILE_ID"]
df = load_data_from_gdrive(FILE_ID)

# === VARIABLES ===
columnas_ocultas = ["Nombre", "Numero de Estudiante", "Email UPR", "Número de Expediente"]
demograficas = ["Procedencia", "1st Fall Enrollment", "Índice General", "Índice Científico", "PCAT"]
notas_cursos = [
    "Español Básico Nota 1", "Español Básico Nota 2", "Inglés Básico Nota 1", "Inglés Básico Nota 2",
    "Ciencias Sociales Nota 1", "Ciencias Sociales Nota 2",
    "Introducción al Estudio de la Cultura de Occidente Nota 1", "Introducción al Estudio de la Cultura de Occidente Nota 2",
    "Economía Nota 1", "Economía Nota 2", "Psicología Nota 1", "Idioma (Inglés o Español) Nota 1",
    "Biología General Nota 1", "Biología General Nota 2", "Biología General Nota 3", "Biología General Nota 4",
    "Biología General (D)", "Química General Nota 1", "Química General Nota 2", "Química General Nota 3", "Química General Nota 4", "Química General (D)",
    "Química Orgánica Nota 1", "Química Orgánica Nota 2", "Química Orgánica Nota 3", "Química Orgánica Nota 4",
    "Química Orgánica (W)", "Química Orgánica (D)", "Química Orgánica (F)",
    "Matemática - Pre-Cálculo Nota 1", "Matemática - Pre-Cálculo Nota 2", "Matemática - Pre-Cálculo (D)", "Matemática - Pre-Cálculo (F)",
    "Cálculo I Nota 1", "Cálculo I Nota 2", "Cálculo I (D)", "Cálculo I (F)",
    "Física General Nota 1", "Física General Nota 2", "Física General Nota 3", "Física General Nota 4",
    "Física General (D)", "Física General (F)", "Lab. Física General Nota 1", "Lab. Física General Nota 2",
    "An. y Fisiología Nota 1", "An. y Fisiología Nota 2", "An. y Fisiología Nota 3", "An. y Fisiología Nota 4"
]
continuas = ["Índice General", "Índice Científico", "PCAT"]

# Columnas categóricas válidas
categoricas = [col for col in df.columns if col not in continuas + columnas_ocultas and col in demograficas + notas_cursos]

nota_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
df[notas_cursos] = df[notas_cursos].apply(lambda col: col.map(lambda x: nota_map.get(str(x).strip().upper(), np.nan)))

# === SIDEBAR ===
with st.sidebar:
    st.header("🎛️ Filtros")

    if st.button("🔄 Resetear filtros"):
        st.rerun()

    col_cat = st.selectbox("Filtrar por categoría", categoricas, index=categoricas.index("1st Fall Enrollment") if "1st Fall Enrollment" in categoricas else 0)
    valores_cat = sorted(df[col_cat].dropna().astype(str).unique())
    if col_cat == "1st Fall Enrollment":
        valores_cat = ["All Enrollment"] + valores_cat
    valor_filtro = st.selectbox(f"Valor en '{col_cat}'", valores_cat, index=0)

    col_proc = st.selectbox("Procedencia", ["Todas"] + sorted(df["Procedencia"].dropna().astype(str).unique()), index=0)

    col_x = st.selectbox("Variable continua (eje X)", continuas, index=0)
    y_options = [col for col in continuas if col != col_x]
    col_y = st.selectbox("Variable continua (eje Y)", y_options, index=0)

    if col_x:
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

df_filtrado = df_filtrado[
    (df_filtrado[col_x] >= selected_range[0]) & (df_filtrado[col_x] <= selected_range[1])
]

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
datos_cor = df_filtrado[columnas_cor].copy()
matriz = datos_cor.corr()

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
    width=1200,
    height=1000,
    margin=dict(t=80, l=200, r=50, b=200)
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
scatter.update_layout(title=f"{col_x} vs {col_y} con regresión<br><sub>{ ​:contentReference[oaicite:0]{index=0}​
