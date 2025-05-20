import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import requests
from io import BytesIO

# === CONFIGURACIÓN DE LA PÁGINA ===
st.set_page_config(page_title="Dashboard Estudiantil", layout="wide")

# === FUNCIÓN PARA CARGAR DATOS DESDE GOOGLE DRIVE ===
@st.cache_data
def load_data_from_gdrive(file_id: str) -> pd.DataFrame:
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(BytesIO(response.content), encoding="latin1")
    else:
        st.error(f"Error al descargar archivo: {response.status_code}")
        return pd.DataFrame()

# === LEER FILE_ID DESDE SECRETS ===
FILE_ID = st.secrets["FILE_ID"]
df = load_data_from_gdrive(FILE_ID)

# === VARIABLES Y MAPEO ===
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

# === SIDEBAR ===
with st.sidebar:
    st.header("🎚️ Filtros")
    
    # Filtro categórico
    col_cat = st.selectbox("Filtrar por categoría", categoricas)
    valores_cat = sorted(df[col_cat].dropna().apply(lambda x: str(x).strip()).unique())
    valor_filtro = st.selectbox(f"Valor en '{col_cat}'", valores_cat)

    # Filtro variable continua
    col_cont = st.selectbox("Variable continua a graficar", continuas)

    # Slider para variable continua
    if col_cont:
        min_val = float(df[col_cont].min())
        max_val = float(df[col_cont].max())
        slider_step = 1.0 if col_cont == "PCAT" else 0.1
        selected_range = st.slider(
            f"Rango de '{col_cont}'",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val),
            step=slider_step
        )
    else:
        selected_range = (None, None)

# === FILTRADO DINÁMICO ===
df_filtrado = df[
    (df[col_cat].apply(lambda x: str(x).strip()) == valor_filtro) &
    (df[col_cont] >= selected_range[0]) &
    (df[col_cont] <= selected_range[1])
]

# === TÍTULO ===
st.markdown("## 📊 Dashboard Estudiantil")

# === MÉTRICAS EN TARJETAS ===
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros", f"{len(df):,}")
col2.metric("Promedio General", f"{df['Índice General'].mean():.2f}")
col3.metric("Promedio Científico", f"{df['Índice Científico'].mean():.2f}")
col4.metric("Promedio PCAT", f"{df['PCAT'].mean():.2f}")

# === GRÁFICO CONTINUO FILTRADO ===
st.markdown("### 📈 Distribución de variable continua (filtrada)")
fig1 = go.Figure()
fig1.add_trace(go.Histogram(
    x=df_filtrado[col_cont],
    nbinsx=10,
    marker_color="#1f77b4"
))
fig1.update_layout(title=f"Distribución de {col_cont}", xaxis_title=col_cont, yaxis_title="Frecuencia")
st.plotly_chart(fig1, use_container_width=True)

# === GRÁFICO CATEGÓRICO TOTAL ===
st.markdown("### 📊 Distribución total de la categoría")
valores = df[col_cat].apply(lambda x: str(x).strip()).value_counts().sort_index()
fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=valores.index,
    y=valores.values,
    marker_color="#2c3e50"
))
fig2.update_layout(title=f"Distribución de {col_cat}", xaxis_title=col_cat, yaxis_title="Cantidad")
st.plotly_chart(fig2, use_container_width=True)

# === MATRIZ DE CORRELACIÓN ===
st.markdown("### 🔗 Matriz de Correlación (notas vs. métricas)")
columnas_cor = notas_letra + continuas
datos_cor = df[columnas_cor].replace({pd.NA: np.nan})
matriz = datos_cor.corr()
fig3 = go.Figure(data=go.Heatmap(
    z=matriz.values,
    x=matriz.columns,
    y=matriz.index,
    colorscale="Blues",
    zmin=-1,
    zmax=1
))
fig3.update_layout(title="Correlación entre notas y métricas")
st.plotly_chart(fig3, use_container_width=True)

# === TABLA FINAL ===
st.markdown("### 🧾 Tabla de datos filtrados")
st.dataframe(df_filtrado)
