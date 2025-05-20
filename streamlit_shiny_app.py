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
def reset_defaults():
    st.session_state.col_cat = "1st Fall Enrollment"
    st.session_state.valor_filtro = "All Enrollment"
    st.session_state.procedencia = "Todas"
    st.session_state.col_x = "Índice General"
    st.session_state.col_y = "Índice Científico"
    st.session_state.slider = (float(df["Índice General"].min()), float(df["Índice General"].max()))

with st.sidebar:
    st.header("🎚️ Filtros")

    if st.button("🔄 Resetear filtros"):
        reset_defaults()

    col_cat = st.selectbox("Filtrar por categoría", categoricas, key="col_cat")
    valores_cat = sorted(df[col_cat].dropna().apply(lambda x: str(x).strip()).unique())
    if col_cat == "1st Fall Enrollment":
        valores_cat = ["All Enrollment"] + valores_cat
    valor_filtro = st.selectbox(f"Valor en '{col_cat}'", valores_cat, index=0 if "valor_filtro" not in st.session_state else valores_cat.index(st.session_state.valor_filtro), key="valor_filtro")

    procedencias = sorted(df["Procedencia"].dropna().apply(lambda x: str(x).strip()).unique())
    selected_procedencia = st.selectbox("Procedencia", ["Todas"] + procedencias, index=0 if "procedencia" not in st.session_state else (["Todas"] + procedencias).index(st.session_state.procedencia), key="procedencia")

    col_x = st.selectbox("Variable continua (eje X)", continuas, key="col_x")
    y_options = [col for col in continuas if col != col_x]
    col_y = st.selectbox("Variable continua (eje Y)", y_options, key="col_y")

    if col_x:
        min_val = float(df[col_x].min())
        max_val = float(df[col_x].max())
        slider_step = 1.0 if col_x == "PCAT" else 0.1
        selected_range = st.slider(
            f"Rango de '{col_x}'",
            min_value=min_val,
            max_value=max_val,
            value=st.session_state.get("slider", (min_val, max_val)),
            step=slider_step,
            key="slider"
        )
    else:
        selected_range = (None, None)

# (rest of the code remains unchanged)
