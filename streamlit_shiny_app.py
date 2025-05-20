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
demograficas = [
    "Nombre", "Número de Estudiante", "Email UPR", "Procedencia", "Número de Expediente",
    "1st Fall Enrollment", "Índice General", "Índice Científico", "PCAT"
]

cursos_requeridos = [
    "Español Básico Nota 1", "Español Básico Nota 2",
    "Inglés Básico Nota 1", "Inglés Básico Nota 2",
    "Ciencias Sociales Nota 1", "Ciencias Sociales Nota 2",
    "Introducción al Estudio de la Cultura de Occidente Nota 1", "Introducción al Estudio de la Cultura de Occidente Nota 2",
    "Economía Nota 1", "Economía Nota 2", "Psicología Nota 1", "Idioma (Inglés o Español) Nota 1",
    "Biología General Nota 1", "Biología General Nota 2", "Biología General Nota 3", "Biología General Nota 4",
    "Biología General (D)", "Química General Nota 1", "Química General Nota 2", "Química General Nota 3",
    "Química General Nota 4", "Química General (D)", "Química Orgánica Nota 1", "Química Orgánica Nota 2",
    "Química Orgánica Nota 3", "Química Orgánica Nota 4", "Química Orgánica (W)", "Química Orgánica (D)", "Química Orgánica (F)",
    "Matemática - Pre-Cálculo Nota 1", "Matemática - Pre-Cálculo Nota 2", "Matemática - Pre-Cálculo (D)", "Matemática - Pre-Cálculo (F)",
    "Cálculo I Nota 1", "Cálculo I Nota 2", "Cálculo I (D)", "Cálculo I (F)",
    "Física General Nota 1", "Física General Nota 2", "Física General Nota 3", "Física General Nota 4",
    "Física General (D)", "Física General (F)", "Lab. Física General Nota 1", "Lab. Física General Nota 2",
    "An. y Fisiología Nota 1", "An. y Fisiología Nota 2", "An. y Fisiología Nota 3", "An. y Fisiología Nota 4"
]

categoricas = ["1st Fall Enrollment"] + cursos_requeridos
continuas = ["Índice General", "Índice Científico", "PCAT"]
notas_letra = cursos_requeridos
nota_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
df[notas_letra] = df[notas_letra].apply(lambda col: col.map(lambda x: nota_map.get(str(x).strip().upper(), np.nan)))

# === SIDEBAR ===
with st.sidebar:
    st.header("🎚️ Filtros")

    col_cat = st.selectbox("Filtrar por categoría", categoricas)
    valores_cat = sorted(df[col_cat].dropna().apply(lambda x: str(x).strip()).unique())
    if col_cat == "1st Fall Enrollment":
        valores_cat = ["All Enrollment"] + valores_cat
    valor_filtro = st.selectbox(f"Valor en '{col_cat}'", valores_cat, index=0)

    procedencias = sorted(df["Procedencia"].dropna().apply(lambda x: str(x).strip()).unique())
    selected_procedencia = st.selectbox("Procedencia", ["Todas"] + procedencias)

    col_x = st.selectbox("Variable continua (eje X)", continuas)
    y_options = [col for col in continuas if col != col_x]
    col_y = st.selectbox("Variable continua (eje Y)", y_options)

    if col_x:
        min_val = float(df[col_x].min())
        max_val = float(df[col_x].max())
        slider_step = 1.0 if col_x == "PCAT" else 0.1
        selected_range = st.slider(
            f"Rango de '{col_x}'",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=slider_step
        )
    else:
        selected_range = (None, None)

# === FILTRO PRINCIPAL ===
df_temp = df.copy()
if col_cat == "1st Fall Enrollment" and valor_filtro != "All Enrollment":
    df_temp = df_temp[df_temp[col_cat].apply(lambda x: str(x).strip()) == valor_filtro]
elif col_cat != "1st Fall Enrollment":
    df_temp = df_temp[df_temp[col_cat].apply(lambda x: str(x).strip()) == valor_filtro]

if selected_procedencia != "Todas":
    df_temp = df_temp[df_temp["Procedencia"].apply(lambda x: str(x).strip()) == selected_procedencia]

df_filtrado = df_temp[
    (df_temp[col_x] >= selected_range[0]) &
    (df_temp[col_x] <= selected_range[1])
]
corr_df = df_filtrado.copy()
barras_df = df_temp.copy()

# === MÉTRICAS ===
st.markdown("## 📊 Dashboard Estudiantil")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total registros", f"{len(df):,}")
c2.metric("Promedio General", f"{df['Índice General'].mean():.2f}")
c3.metric("Promedio Científico", f"{df['Índice Científico'].mean():.2f}")
c4.metric("Promedio PCAT", f"{df['PCAT'].mean():.2f}")

# === HISTOGRAMA ===
hist = go.Figure()
hist.add_trace(go.Histogram(x=df_filtrado[col_x], nbinsx=10, marker_color="#1f77b4"))
hist.update_layout(title=f"Distribución de {col_x}", xaxis_title=col_x, yaxis_title="Frecuencia")

# === BARRAS ===
valores_barras = barras_df[col_cat].dropna().apply(lambda x: str(x).strip()).value_counts().sort_index()
bars = go.Figure()
bars.add_trace(go.Bar(x=valores_barras.index.astype(str), y=valores_barras.values, marker_color="#2c3e50"))
bars.update_layout(
    title=f"Distribución de {col_cat}",
    xaxis_title=col_cat,
    yaxis_title="Cantidad",
    xaxis_type='category'
)

# === MATRIZ DE CORRELACIÓN ===
columnas_cor = notas_letra + continuas
datos_cor = corr_df[columnas_cor].dropna(how='any')

if datos_cor.shape[0] >= 2:
    data_corr = datos_cor.corr()
    heatmap = go.Figure(data=go.Heatmap(
        z=data_corr.values,
        x=data_corr.columns,
        y=data_corr.index,
        colorscale="Blues",
        zmin=-1,
        zmax=1
    ))
    heatmap.update_layout(
        title="Correlación entre notas y métricas",
        width=1200,
        height=1000,
        margin=dict(l=250, r=50, b=250, t=50),
        xaxis_tickangle=45
    )
    show_heatmap = True
else:
    show_heatmap = False

# === REGRESIÓN LINEAL ===
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

# === SCATTER PLOT ===
scatter = go.Figure()
scatter.add_trace(go.Scatter(
    x=x_clean.flatten(),
    y=y_clean.flatten(),
    mode='markers',
    name='Datos'
))
scatter.add_trace(go.Scatter(
    x=x_clean.flatten(),
    y=y_pred.flatten(),
    mode='lines',
    name='Regresión',
    line=dict(color='orange')
))
scatter.update_layout(
    title=f"{col_x} vs {col_y} con regresión<br><sub>{equation}</sub>",
    xaxis_title=col_x,
    yaxis_title=col_y
)

# === LAYOUT ===
g1, g2 = st.columns(2)
g1.plotly_chart(hist, use_container_width=True)
g2.plotly_chart(bars, use_container_width=True)

g3, g4 = st.columns(2)
g3.plotly_chart(scatter, use_container_width=True)
if show_heatmap:
    g4.plotly_chart(heatmap, use_container_width=True)
else:
    g4.warning("⚠️ No hay suficientes datos completos para calcular la matriz de correlación.")

# === TABLA ===
st.markdown("### 🧾 Tabla de datos filtrados")
st.dataframe(df_filtrado)
