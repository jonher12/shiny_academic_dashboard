import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from io import BytesIO

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

# Columnas y mapeo
categoricas = [
    "1st Fall Enrollment", "Español Básico Nota 1", "Español Básico Nota 2",
    "Inglés Básico Nota 1", "Inglés Básico Nota 2",
    "Ciencias Sociales Nota 1", "Ciencias Sociales Nota 2"
]
continuas = ["Índice General", "Índice Científico", "PCAT"]
notas_letra = [
    "Español Básico Nota 1", "Español Básico Nota 2",
    "Inglés Básico Nota 1", "Inglés Básico Nota 2",
    "Ciencias Sociales Nota 1", "Ciencias Sociales Nota 2"
]
nota_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
df[notas_letra] = df[notas_letra].apply(lambda col: col.map(lambda x: nota_map.get(str(x).strip().upper(), np.nan)))

# Sidebar
with st.sidebar:
    st.header("🎚️ Filtros")
    col_cat = st.selectbox("Filtrar por categoría", categoricas)
    valores_cat = sorted(df[col_cat].dropna().apply(lambda x: str(x).strip()).unique())
    valor_filtro = st.selectbox(f"Valor en '{col_cat}'", valores_cat)

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

# Filtro principal
df_filtrado = df[
    (df[col_cat].apply(lambda x: str(x).strip()) == valor_filtro) &
    (df[col_x] >= selected_range[0]) &
    (df[col_x] <= selected_range[1])
]

# Título y métricas
st.markdown("## 📊 Dashboard Estudiantil")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total registros", f"{len(df):,}")
c2.metric("Promedio General", f"{df['Índice General'].mean():.2f}")
c3.metric("Promedio Científico", f"{df['Índice Científico'].mean():.2f}")
c4.metric("Promedio PCAT", f"{df['PCAT'].mean():.2f}")

# Histograma
hist = go.Figure()
hist.add_trace(go.Histogram(x=df_filtrado[col_x], nbinsx=10, marker_color="#1f77b4"))
hist.update_layout(title=f"Distribución de {col_x}", xaxis_title=col_x, yaxis_title="Frecuencia")

# Barras
valores = df[col_cat].apply(lambda x: str(x).strip()).value_counts().sort_index()
bars = go.Figure()
bars.add_trace(go.Bar(x=valores.index, y=valores.values, marker_color="#2c3e50"))
bars.update_layout(title=f"Distribución de {col_cat}", xaxis_title=col_cat, yaxis_title="Cantidad")

# Correlación
columnas_cor = notas_letra + continuas
datos_cor = df[columnas_cor].replace({pd.NA: np.nan})
matriz = datos_cor.corr()
heatmap = go.Figure(data=go.Heatmap(
    z=matriz.values, x=matriz.columns, y=matriz.index,
    colorscale="Blues", zmin=-1, zmax=1
))
heatmap.update_layout(title="Correlación entre notas y métricas")

# Scatter con regresión
scatter = px.scatter(
    df_filtrado,
    x=col_x,
    y=col_y,
    trendline="ols",
    title=f"{col_x} vs {col_y} con regresión"
)

# Layout en grid
g1, g2 = st.columns(2)
g1.plotly_chart(hist, use_container_width=True)
g2.plotly_chart(bars, use_container_width=True)

g3, g4 = st.columns(2)
g3.plotly_chart(scatter, use_container_width=True)
g4.plotly_chart(heatmap, use_container_width=True)

# Tabla final
st.markdown("### 🧾 Tabla de datos filtrados")
st.dataframe(df_filtrado)
