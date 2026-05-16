"""
Tablero Seguro Agrícola Indexado producción de Café
Modelo LASSO — Nariño y Cundinamarca

Requiere:
- data/base_narino_cundinamarca.csv
- models/modelo_narino_cundinamarca.joblib
"""

import streamlit as st
import pandas as pd
import joblib
import os
import altair as alt


# 1. Configuración de página

st.set_page_config(
    page_title="Seguro Indexado Producción de Café",
    page_icon="🌾",
    layout="wide",
)

alt.themes.enable("default")


# 2. Estilos visuales

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f6f8;
            color: #263238;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 5.5rem;
            padding-right: 5.5rem;
            max-width: 100%;
        }

        @media screen and (max-width: 1200px) {
            .block-container {
                padding-left: 3rem;
                padding-right: 3rem;
            }
        }

        header[data-testid="stHeader"] {
            background-color: #11151b;
        }

        section[data-testid="stSidebar"] {
            display: none;
        }

        .encabezado {
            background: linear-gradient(135deg, #3e2723, #6d4c41);
            color: white;
            padding: 28px;
            border-radius: 18px;
            margin-bottom: 22px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }

        .encabezado h1 {
            font-size: 30px;
            margin-bottom: 8px;
            font-weight: 700;
        }

        .encabezado p {
            font-size: 15px;
            opacity: 0.92;
            margin: 0;
        }

        .card {
            background: white;
            padding: 22px;
            border-radius: 18px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 20px;
        }

        .card h2 {
            font-size: 20px;
            margin-bottom: 16px;
            color: #3e2723;
        }

        .card p {
            font-size: 13px;
            color: #78909c;
            line-height: 1.5;
        }

        .card-contexto {
            background: white;
            padding: 14px 18px;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            margin-bottom: 18px;
        }

        .card-contexto h2 {
            font-size: 16px;
            margin-bottom: 8px;
            color: #3e2723;
        }

        .card-contexto p {
            font-size: 12px;
            color: #78909c;
            line-height: 1.4;
            margin: 0;
        }

        .st-key-parametros_prediccion {
            background-color: #ffffff !important;
            border: 1px solid #d7dde2 !important;
            border-radius: 18px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            padding: 22px 24px 18px 24px !important;
            margin-bottom: 22px !important;
        }

        .st-key-parametros_prediccion div {
            background-color: transparent !important;
        }

        .st-key-parametros_prediccion div[data-baseweb="select"] > div,
        .st-key-parametros_prediccion .stNumberInput input {
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        .st-key-comportamiento_historico,
        .st-key-regla_activacion {
            background-color: #ffffff !important;
            border: 1px solid #d7dde2 !important;
            border-radius: 18px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            padding: 22px 24px 18px 24px !important;
            margin-bottom: 22px !important;
            min-height: auto !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        .st-key-comportamiento_historico div,
        .st-key-regla_activacion div {
            box-sizing: border-box !important;
        }

        .st-key-comportamiento_historico [data-testid="stSelectbox"],
        .st-key-comportamiento_historico [data-testid="stVegaLiteChart"],
        .st-key-comportamiento_historico div[data-baseweb="select"] {
            max-width: 100% !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        .st-key-comportamiento_historico div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            max-width: 100% !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }

        .st-key-comportamiento_historico div[data-testid="stVegaLiteChart"] {
            max-width: 100% !important;
            width: 100% !important;
            overflow: hidden !important;
        }

        .st-key-comportamiento_historico div[data-testid="stVegaLiteChart"] > div {
            max-width: 100% !important;
            width: 100% !important;
            overflow: hidden !important;
        }

        .st-key-comportamiento_historico canvas,
        .st-key-comportamiento_historico svg {
            max-width: 100% !important;
            width: 100% !important;
            overflow: visible !important;
        }

        .st-key-simulador_seguro {
            background-color: #ffffff !important;
            border: 1px solid #d7dde2 !important;
            border-radius: 18px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
            padding: 22px 24px 18px 24px !important;
            margin-bottom: 22px !important;
        }

        .kpi-card {
            background: white;
            padding: 20px;
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
            border-left: 6px solid #795548;
            min-height: 135px;
        }

        .kpi-title {
            font-size: 13px;
            color: #607d8b;
            margin-bottom: 10px;
            font-weight: 600;
        }

        .kpi-value {
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 6px;
            color: #263238;
        }

        .kpi-detail {
            font-size: 12px;
            color: #78909c;
            line-height: 1.35;
        }

        .sim-card {
            background-color: #e9e0da !important;
            border-radius: 18px !important;
            padding: 22px 16px !important;
            text-align: center !important;
            min-height: 118px !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: none !important;
        }

        .sim-card h3 {
            font-size: 16px !important;
            color: #5d4037 !important;
            margin-bottom: 8px !important;
            font-weight: 700 !important;
        }

        .sim-card p {
            font-size: 30px !important;
            font-weight: 800 !important;
            color: #3e2723 !important;
            margin: 0 !important;
        }

        .nota {
            font-size: 12px;
            color: #78909c;
            margin-top: 12px;
            line-height: 1.5;
        }

        .formula-box {
            background: #fafafa;
            border: 1px solid #eceff1;
            border-radius: 14px;
            padding: 14px 16px;
            margin: 10px 0 16px 0;
        }

        .formula-box p {
            font-size: 13px;
            color: #607d8b;
            line-height: 1.55;
            margin: 0;
        }

        .stSelectbox label,
        .stNumberInput label,
        .stSlider label {
            color: #000000 !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            text-transform: uppercase;
        }

        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            border: 1px solid #d7dde2 !important;
            border-radius: 12px !important;
            color: #111827 !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="select"] input {
            color: #111827 !important;
        }

        div[data-baseweb="select"] svg {
            fill: #111827 !important;
        }

        div[data-baseweb="popover"] {
            background-color: transparent !important;
            z-index: 999999 !important;
        }

        ul[role="listbox"] {
            background-color: #ffffff !important;
            border: 1px solid #d7dde2 !important;
            border-radius: 12px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15) !important;
            overflow: hidden !important;
        }

        ul[role="listbox"] li {
            background-color: #ffffff !important;
            color: #111827 !important;
        }

        ul[role="listbox"] li:hover {
            background-color: #f3f4f6 !important;
            color: #111827 !important;
        }

        ul[role="listbox"] li[aria-selected="true"] {
            background-color: #f1f5f9 !important;
            color: #111827 !important;
            font-weight: 700 !important;
        }

        .stTextInput input,
        .stTextArea textarea {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #d7dde2 !important;
            border-radius: 10px !important;
        }

        input::placeholder,
        textarea::placeholder {
            color: #616161 !important;
            opacity: 1 !important;
        }

        div[data-testid="stNumberInput"] > div {
            border: 1px solid #d7dde2 !important;
            border-radius: 10px !important;
            background-color: #ffffff !important;
            overflow: hidden !important;
        }

        div[data-testid="stNumberInput"] input {
            border: none !important;
            border-radius: 10px 0 0 10px !important;
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        div[data-testid="stNumberInput"] button {
            border-top: none !important;
            border-bottom: none !important;
            border-right: none !important;
            border-left: 1px solid #d7dde2 !important;
            border-radius: 0 !important;
            background-color: #ffffff !important;
            color: #000000 !important;
        }

        div[data-testid="stNumberInput"] button:last-child {
            border-radius: 0 10px 10px 0 !important;
        }

        .stSlider * {
            color: #000000 !important;
        }

        .stSlider [data-testid="stTickBar"] {
            color: #000000 !important;
        }

        .stSlider [data-testid="stTickBar"] * {
            color: #000000 !important;
        }

        div[data-baseweb="slider"] span {
            color: #000000 !important;
        }

        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #795548 !important;
            border: 2px solid #795548 !important;
            box-shadow: none !important;
        }

        .tabla-clara-wrapper {
            width: 100%;
            overflow-x: auto;
            margin-top: 12px;
            margin-bottom: 16px;
        }

        .tabla-clara {
            width: 100%;
            border-collapse: collapse;
            background-color: #ffffff !important;
            color: #263238 !important;
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #e0e0e0;
            font-size: 13px;
        }

        .tabla-clara th {
            background-color: #f5f7f9 !important;
            color: #3e2723 !important;
            font-weight: 700;
            text-align: left;
            padding: 12px 14px;
            border-bottom: 1px solid #e0e0e0;
        }

        .tabla-clara td {
            background-color: #ffffff !important;
            color: #263238 !important;
            padding: 11px 14px;
            border-bottom: 1px solid #eceff1;
            vertical-align: top;
        }

        .tabla-clara tr:last-child td {
            border-bottom: none;
        }

        .tabla-clara tr:hover td {
            background-color: #fafafa !important;
        }

        .st-key-regla_activacion .tabla-clara-wrapper {
            max-width: 100%;
        }

        div[data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 14px !important;
            overflow: hidden !important;
        }

        div[data-testid="stExpander"] details {
            background-color: #ffffff !important;
        }

        div[data-testid="stExpander"] summary {
            color: #263238 !important;
            background-color: #ffffff !important;
        }

        div[data-testid="stExpander"] div {
            background-color: transparent !important;
        }

        /* Ajustar ancho del selectbox dentro de comportamiento histórico */
        .st-key-comportamiento_historico div[data-testid="stSelectbox"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        .st-key-comportamiento_historico div[data-testid="stSelectbox"] > div {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }

        .st-key-comportamiento_historico div[data-baseweb="select"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        .st-key-comportamiento_historico div[data-baseweb="select"] > div {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        .st-key-comportamiento_historico div[data-baseweb="select"] span {
            max-width: calc(100% - 28px) !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# 3. Encabezado

st.markdown(
    """
    <div class="encabezado">
        <h1>Tablero de Seguro Indexado para la Producción de Café</h1>
        <p>Predicción del rendimiento porcentual y valoración preliminar del riesgo para Cundinamarca y Nariño.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# 4. Carga de datos y modelo

BASE_PATH = "data/base_narino_cundinamarca.csv"
MODELO_PATH = "models/modelo_narino_cundinamarca.joblib"


@st.cache_data
def cargar_base():
    df = pd.read_csv(BASE_PATH)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource
def cargar_modelo():
    if os.path.exists(MODELO_PATH):
        return joblib.load(MODELO_PATH)
    return None


df_base = cargar_base()
modelo = cargar_modelo()


# 5. Variables del modelo

VARS_NUMERICAS = [
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "altitud_media_m",
    "SPI1_floracion",
    "SPI1_llenado",
    "Rendimiento_lag1",
]

VARS_CATEGORICAS = [
    "Departamento",
    "Municipio",
]

VARIABLES_MODELO = VARS_NUMERICAS + VARS_CATEGORICAS

COLUMNAS_REQUERIDAS = [
    "Año",
    "Departamento",
    "Municipio",
    "Rendimiento",
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "altitud_media_m",
    "SPI1_floracion",
    "SPI1_llenado",
]

columnas_faltantes = [
    col for col in COLUMNAS_REQUERIDAS
    if col not in df_base.columns
]

if len(columnas_faltantes) > 0:
    st.error("Faltan columnas requeridas en la base.")
    st.write(columnas_faltantes)
    st.write("Columnas disponibles:")
    st.write(list(df_base.columns))
    st.stop()


# 6. Limpieza básica de tipos

df_base["Año"] = pd.to_numeric(df_base["Año"], errors="coerce")

columnas_numericas_base = [
    "Rendimiento",
    "Precipitación acumulada anual (mm/año)",
    "Temperatura media anual (°C)",
    "Máximo de la temperatura media mensual (°C)",
    "Mínimo de la temperatura media mensual (°C)",
    "Humedad relativa media anual (%)",
    "Radiación solar acumulada anual (MJ/m²/año)",
    "altitud_media_m",
    "SPI1_floracion",
    "SPI1_llenado",
]

for col in columnas_numericas_base:
    df_base[col] = pd.to_numeric(df_base[col], errors="coerce")

df_base["Departamento"] = df_base["Departamento"].astype(str)
df_base["Municipio"] = df_base["Municipio"].astype(str)


# 7. Base para históricos

MIN_ANIOS_HISTORICOS = 3

df_historico = df_base.dropna(subset=["Rendimiento"]).copy()

historico_mun = (
    df_historico
    .groupby(["Departamento", "Municipio"])["Rendimiento"]
    .agg(
        rend_medio="mean",
        rend_std="std",
        rend_min="min",
        rend_max="max",
        n_obs="count"
    )
    .reset_index()
)

historico_mun_valido = historico_mun[
    historico_mun["n_obs"] >= MIN_ANIOS_HISTORICOS
].copy()


# 8. Funciones

def calcular_activacion(rend_estimado, rend_hist_medio, suma_asegurada, alpha):
    trigger = alpha * rend_hist_medio

    if trigger <= 0:
        deficit_rel = 0.0
        pct_pago = 0.0
        nivel = "Bajo"

    elif rend_estimado >= trigger:
        deficit_rel = 0.0
        pct_pago = 0.0
        nivel = "Bajo"

    else:
        deficit_rel = (trigger - rend_estimado) / trigger
        pct_pago = deficit_rel

        if deficit_rel < 0.15:
            nivel = "Medio"
        elif deficit_rel < 0.35:
            nivel = "Alto"
        else:
            nivel = "Crítico"

    indemnizacion = suma_asegurada * pct_pago
    deficit_pp = (rend_hist_medio - rend_estimado) * 100

    return trigger, deficit_pp, pct_pago, indemnizacion, nivel


def estilo_variacion(valor):
    if valor > 0:
        return {
            "icono": "↑",
            "color": "#2e7d32",
            "fondo": "#e8f5e9",
            "borde": "#66bb6a"
        }
    elif valor < 0:
        return {
            "icono": "↓",
            "color": "#c62828",
            "fondo": "#ffebee",
            "borde": "#ef5350"
        }
    else:
        return {
            "icono": "→",
            "color": "#616161",
            "fondo": "#f5f5f5",
            "borde": "#bdbdbd"
        }


def estilo_riesgo(nivel):
    if nivel == "Bajo":
        return {
            "color": "#2e7d32",
            "fondo": "#e8f5e9",
            "borde": "#66bb6a",
            "icono": "🟢"
        }
    elif nivel == "Medio":
        return {
            "color": "#ef6c00",
            "fondo": "#fff3e0",
            "borde": "#ffb74d",
            "icono": "🟡"
        }
    elif nivel == "Alto":
        return {
            "color": "#d32f2f",
            "fondo": "#ffebee",
            "borde": "#ef5350",
            "icono": "🔴"
        }
    else:
        return {
            "color": "#8e0000",
            "fondo": "#fdecea",
            "borde": "#e57373",
            "icono": "🔴"
        }


def render_tabla_clara(df):
    tabla_html = df.to_html(
        index=False,
        escape=False,
        classes="tabla-clara"
    )

    return f"""
    <div class="tabla-clara-wrapper">
        {tabla_html}
    </div>
    """


# 9. Filtros dentro del recuadro de parámetros

with st.container(key="parametros_prediccion"):
    st.markdown(
        """
        <h2 style="
            font-size:16px;
            text-transform:uppercase;
            letter-spacing:0.5px;
            margin:0 0 22px 0;
            color:#5d4037;
        ">
            ⚙️ Parámetros de predicción
        </h2>
        """,
        unsafe_allow_html=True
    )

    f1, f2, f3, f4, f5, f6 = st.columns(6)

    with f1:
        departamentos_disponibles = sorted(
            historico_mun_valido["Departamento"].dropna().unique()
        )

        if len(departamentos_disponibles) == 0:
            st.error(
                f"No hay departamentos con municipios que tengan al menos {MIN_ANIOS_HISTORICOS} años de rendimiento histórico."
            )
            st.stop()

        departamento = st.selectbox(
            "Departamento",
            options=departamentos_disponibles
        )

    municipios_disponibles = sorted(
        historico_mun_valido[
            historico_mun_valido["Departamento"] == departamento
        ]["Municipio"].dropna().unique()
    )

    if len(municipios_disponibles) == 0:
        st.error(
            f"No hay municipios con al menos {MIN_ANIOS_HISTORICOS} años de rendimiento histórico para {departamento}."
        )
        st.stop()

    with f2:
        municipio = st.selectbox(
            "Municipio",
            options=municipios_disponibles
        )

    anios_disponibles = sorted(
        df_base[
            (df_base["Departamento"] == departamento) &
            (df_base["Municipio"] == municipio)
        ]["Año"].dropna().unique()
    )

    with f3:
        anio_base = st.selectbox(
            "Año de análisis",
            options=anios_disponibles,
            index=len(anios_disponibles) - 1
        )

    with f4:
        rendimiento_anterior_pct = st.number_input(
            "Rendimiento año anterior (%)",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=0.5
        )

    rendimiento_lag1_manual = rendimiento_anterior_pct / 100

    with f5:
        suma_asegurada = st.number_input(
            "Valor asegurado (COP)",
            min_value=100_000,
            max_value=500_000_000,
            value=5_000_000,
            step=100_000,
            format="%d"
        )

    with f6:
        alpha_pct = st.slider(
            "Trigger (%)",
            min_value=65,
            max_value=90,
            value=80,
            step=5
        )

alpha = alpha_pct / 100


# 10. Datos históricos del municipio

datos_hist = historico_mun_valido[
    (historico_mun_valido["Departamento"] == departamento) &
    (historico_mun_valido["Municipio"] == municipio)
]

if datos_hist.empty:
    st.error(
        f"No hay al menos {MIN_ANIOS_HISTORICOS} años de rendimiento histórico para {municipio}."
    )
    st.stop()

rend_hist_medio = float(datos_hist["rend_medio"].iloc[0])
rend_hist_std = float(datos_hist["rend_std"].iloc[0])
rend_hist_min = float(datos_hist["rend_min"].iloc[0])
rend_hist_max = float(datos_hist["rend_max"].iloc[0])
n_obs = int(datos_hist["n_obs"].iloc[0])

if pd.isna(rend_hist_std):
    rend_hist_std = 0.0


# 11. Datos del municipio y año seleccionado

datos_mun = df_base[
    (df_base["Departamento"] == departamento) &
    (df_base["Municipio"] == municipio)
].sort_values("Año").copy()

fila_pred = df_base[
    (df_base["Departamento"] == departamento) &
    (df_base["Municipio"] == municipio) &
    (df_base["Año"] == anio_base)
].copy()

if fila_pred.empty:
    st.error("No hay registros para el municipio y año seleccionado.")
    st.stop()

fila_pred = fila_pred.iloc[0].copy()


# 12. Preparación de datos para predicción

X_input = pd.DataFrame([fila_pred])

X_input["Rendimiento_lag1"] = rendimiento_lag1_manual
X_input["Departamento"] = departamento
X_input["Municipio"] = municipio

X_input = X_input[VARIABLES_MODELO].copy()

for col in VARS_NUMERICAS:
    X_input[col] = pd.to_numeric(X_input[col], errors="coerce")

variables_vacias = X_input[VARS_NUMERICAS].isna().sum()
variables_vacias = variables_vacias[variables_vacias > 0]

if len(variables_vacias) > 0:
    st.error("Hay variables numéricas vacías para realizar la predicción.")
    st.write(variables_vacias)
    st.stop()


# 13. Predicción

if modelo is not None:
    try:
        rend_estimado = float(modelo.predict(X_input)[0])
    except Exception as e:
        st.error(f"Error al ejecutar la predicción con el modelo: {e}")
        st.write("Revisa que las variables coincidan con las usadas al entrenar el modelo.")
        st.write("Variables enviadas al modelo:")
        st.write(list(X_input.columns))
        st.stop()
else:
    st.warning(
        "⚠️ No se encontró modelo_narino_cundinamarca.joblib. "
        "Se usa Rendimiento_lag1 como aproximación.",
        icon="⚠️"
    )
    rend_estimado = rendimiento_lag1_manual

rend_estimado = max(0.0, rend_estimado)

if rend_estimado > 1:
    rend_estimado = rend_estimado / 100


# 14. Cálculo de activación

trigger, deficit_pp, pct_pago, indemnizacion, nivel = calcular_activacion(
    rend_estimado=rend_estimado,
    rend_hist_medio=rend_hist_medio,
    suma_asegurada=suma_asegurada,
    alpha=alpha
)

rend_estimado_pct = rend_estimado * 100
rend_historico_pct = rend_hist_medio * 100
trigger_pct = trigger * 100
variacion_lag_pct = rend_estimado_pct - rendimiento_anterior_pct

estilo_var = estilo_variacion(variacion_lag_pct)
estilo_risk = estilo_riesgo(nivel)


# 15. Tarjeta de contexto

st.markdown(
    f"""
    <div class="card-contexto">
        <h2>📍 {municipio}, {departamento.upper()} · Año de análisis {int(anio_base)}</h2>
        <p>
            El tablero calcula el rendimiento estimado de acuerdo con el modelo, compara contra el histórico municipal
            y estima la posible activación del seguro indexado.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# 16. KPIs principales

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Rendimiento año anterior</div>
            <div class="kpi-value">{rendimiento_anterior_pct:.1f}%</div>
            <div class="kpi-detail">Dato ingresado por el usuario</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Rendimiento estimado</div>
            <div class="kpi-value">{rend_estimado_pct:.1f}%</div>
            <div class="kpi-detail">Predicción del modelo LASSO</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Variación esperada</div>
            <div class="kpi-value" style="color:{estilo_var['color']};">
                {estilo_var['icono']} {variacion_lag_pct:+.1f} p.p.
            </div>
            <div class="kpi-detail">Frente al rendimiento ingresado</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Riesgo climático</div>
            <div class="kpi-value" style="color:{estilo_risk['color']};">
                {estilo_risk['icono']} {nivel}
            </div>
            <div class="kpi-detail">Según déficit frente al trigger</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k5:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Indemnización potencial</div>
            <div class="kpi-value">${indemnizacion:,.0f}</div>
            <div class="kpi-detail">Pago estimado si se activa el índice</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)


# 17. Comportamiento histórico y regla de activación

col_hist, col_regla = st.columns([1.25, 1])

with col_hist:
    with st.container(key="comportamiento_historico"):
        st.markdown(
            """
            <h2 style="
                font-size:20px;
                margin:0 0 8px 0;
                color:#3e2723;
            ">
                Comportamiento histórico de variables
            </h2>

            <p style="
                font-size:13px;
                color:#78909c;
                margin:0 0 18px 0;
                line-height:1.5;
            ">
                Seleccione una variable para comparar el comportamiento histórico del municipio
                frente al promedio del departamento.
            </p>
            """,
            unsafe_allow_html=True
        )

        variables_graficar = [
            "Rendimiento",
            "Precipitación acumulada anual (mm/año)",
            "Temperatura media anual (°C)",
            "Máximo de la temperatura media mensual (°C)",
            "Mínimo de la temperatura media mensual (°C)",
            "Humedad relativa media anual (%)",
            "Radiación solar acumulada anual (MJ/m²/año)",
            "SPI1_floracion",
            "SPI1_llenado"
        ]

        variable_seleccionada = st.selectbox(
            "Variable histórica a visualizar",
            options=variables_graficar,
            key="variable_historica"
        )

        if variable_seleccionada == "Rendimiento":
            df_municipio = datos_mun[["Año", "Rendimiento"]].dropna().copy()
            df_municipio["Año"] = df_municipio["Año"].astype(int)
            df_municipio["Valor"] = df_municipio["Rendimiento"] * 100
            df_municipio["Serie"] = "Municipio"

            df_departamento = (
                df_base[
                    (df_base["Departamento"] == departamento) &
                    (df_base["Rendimiento"].notna())
                ]
                .groupby("Año", as_index=False)["Rendimiento"]
                .mean()
            )

            df_departamento["Año"] = df_departamento["Año"].astype(int)
            df_departamento["Valor"] = df_departamento["Rendimiento"] * 100
            df_departamento["Serie"] = "Departamento"

            df_prediccion = pd.DataFrame({
                "Año": [int(anio_base)],
                "Valor": [rend_estimado_pct],
                "Serie": ["Predicción"]
            })

            df_grafica = pd.concat(
                [
                    df_municipio[["Año", "Valor", "Serie"]],
                    df_departamento[["Año", "Valor", "Serie"]]
                ],
                ignore_index=True
            )

            df_grafica["Año"] = df_grafica["Año"].astype(int)
            df_prediccion["Año"] = df_prediccion["Año"].astype(int)

            anio_min_graf = int(min(df_grafica["Año"].min(), df_prediccion["Año"].min()))
            anio_max_graf = int(max(df_grafica["Año"].max(), df_prediccion["Año"].max()))

            eje_x = alt.X(
                "Año:Q",
                title=None,
                scale=alt.Scale(
                    domain=[anio_min_graf - 0.5, anio_max_graf + 1.2],
                    nice=False,
                    zero=False
                ),
                axis=alt.Axis(
                    values=list(range(anio_min_graf, anio_max_graf + 1)),
                    format="d",
                    labelAngle=-45,
                    labelOverlap=False,
                    labelLimit=80,
                    labelFlush=False
                )
            )

            chart_lineas = alt.Chart(df_grafica).mark_line(
                point=True,
                strokeWidth=4
            ).encode(
                x=eje_x,
                y=alt.Y("Valor:Q",title="Rendimiento (%)", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=["Municipio", "Departamento", "Predicción"],
                        range=["#795548", "#b0bec5", "#ff8f00"]
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                        labelColor="#607d8b",
                        symbolType="circle"
                    )
                ),
                tooltip=[
                    alt.Tooltip("Año:O", title="Año"),
                    alt.Tooltip("Serie:N", title="Serie"),
                    alt.Tooltip("Valor:Q", title="Valor", format=".1f")
                ]
            )

            punto_prediccion = alt.Chart(df_prediccion).mark_point(
                size=180,
                shape="diamond",
                filled=True
            ).encode(
                x=eje_x,
                y=alt.Y("Valor:Q"),
                color=alt.Color(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=["Municipio", "Departamento", "Predicción"],
                        range=["#795548", "#b0bec5", "#ff8f00"]
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                        labelColor="#607d8b",
                        symbolType="circle"
                    )
                ),
                tooltip=[
                    alt.Tooltip("Año:O", title="Año"),
                    alt.Tooltip("Serie:N", title="Serie"),
                    alt.Tooltip("Valor:Q", title="Predicción", format=".1f")
                ]
            )

            chart_variable = (
                chart_lineas + punto_prediccion
            ).properties(
                height=300,
                background="#ffffff",
                padding={
                    "left": 25,
                    "right": 55,
                    "top": 20,
                    "bottom": 35
                },
                autosize=alt.AutoSizeParams(
                    type="fit",
                    contains="padding"
                )
            ).configure_view(
                fill="#ffffff",
                stroke="#eceff1"
            ).configure_axis(
                labelColor="#607d8b",
                titleColor="#5d4037",
                gridColor="#eceff1",
                domainColor="#cfd8dc",
                tickColor="#cfd8dc"
            )

            st.altair_chart(chart_variable, use_container_width=True)

        else:
            df_municipio = datos_mun[["Año", variable_seleccionada]].dropna().copy()
            df_municipio["Año"] = df_municipio["Año"].astype(int)
            df_municipio["Valor"] = df_municipio[variable_seleccionada]
            df_municipio["Serie"] = "Municipio"

            df_departamento = (
                df_base[
                    (df_base["Departamento"] == departamento) &
                    (df_base[variable_seleccionada].notna())
                ]
                .groupby("Año", as_index=False)[variable_seleccionada]
                .mean()
            )

            df_departamento["Año"] = df_departamento["Año"].astype(int)
            df_departamento["Valor"] = df_departamento[variable_seleccionada]
            df_departamento["Serie"] = "Departamento"

            df_grafica = pd.concat(
                [
                    df_municipio[["Año", "Valor", "Serie"]],
                    df_departamento[["Año", "Valor", "Serie"]]
                ],
                ignore_index=True
            )

            df_grafica["Año"] = df_grafica["Año"].astype(int)

            anio_min_graf = int(df_grafica["Año"].min())
            anio_max_graf = int(df_grafica["Año"].max())

            titulo_variable = variable_seleccionada

            if variable_seleccionada == "Precipitación acumulada anual (mm/año)":
                titulo_variable = "Precipitación acumulada anual"
            elif variable_seleccionada == "Temperatura media anual (°C)":
                titulo_variable = "Temperatura media anual"
            elif variable_seleccionada == "Máximo de la temperatura media mensual (°C)":
                titulo_variable = "Máximo de temperatura mensual"
            elif variable_seleccionada == "Mínimo de la temperatura media mensual (°C)":
                titulo_variable = "Mínimo de temperatura mensual"
            elif variable_seleccionada == "Humedad relativa media anual (%)":
                titulo_variable = "Humedad relativa media anual"
            elif variable_seleccionada == "Radiación solar acumulada anual (MJ/m²/año)":
                titulo_variable = "Radiación solar acumulada anual"

            eje_x = alt.X(
                "Año:Q",
                title=None,
                scale=alt.Scale(
                    domain=[anio_min_graf - 0.5, anio_max_graf + 0.8],
                    nice=False,
                    zero=False
                ),
                axis=alt.Axis(
                    values=list(range(anio_min_graf, anio_max_graf + 1)),
                    format="d",
                    labelAngle=-45,
                    labelOverlap=False,
                    labelLimit=80,
                    labelFlush=False
                )
            )

            chart_variable = alt.Chart(df_grafica).mark_line(
                point=True,
                strokeWidth=4
            ).encode(
                x=eje_x,
                y=alt.Y(
                    "Valor:Q",
                    title=titulo_variable,
                    scale=alt.Scale(zero=False)
                ),
                color=alt.Color(
                    "Serie:N",
                    scale=alt.Scale(
                        domain=["Municipio", "Departamento"],
                        range=["#795548", "#b0bec5"]
                    ),
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                        labelColor="#607d8b",
                        symbolType="circle"
                    )
                ),
                tooltip=[
                    alt.Tooltip("Año:O", title="Año"),
                    alt.Tooltip("Serie:N", title="Serie"),
                    alt.Tooltip("Valor:Q", title="Valor", format=".2f")
                ]
            ).properties(
                height=300,
                background="#ffffff",
                padding={
                    "left": 25,
                    "right": 55,
                    "top": 20,
                    "bottom": 35
                },
                autosize=alt.AutoSizeParams(
                    type="fit",
                    contains="padding"
                )
            ).configure_view(
                fill="#ffffff",
                stroke="#eceff1"
            ).configure_axis(
                labelColor="#607d8b",
                titleColor="#5d4037",
                gridColor="#eceff1",
                domainColor="#cfd8dc",
                tickColor="#cfd8dc"
            )

            st.altair_chart(chart_variable, use_container_width=True)

with col_regla:
    with st.container(key="regla_activacion"):
        st.markdown(
            f"""
            <h2 style="
                font-size:20px;
                margin:0 0 10px 0;
                color:#3e2723;
            ">
                Regla de activación
            </h2>

            <div class="formula-box">
                <p>
                    <strong>Fórmula aplicada</strong><br>
                    Trigger = {alpha_pct}% × {rend_historico_pct:.1f}% = <strong>{trigger_pct:.2f}%</strong><br><br>
                    Si Ŷ ≥ Trigger → sin pago.<br>
                    Si Ŷ &lt; Trigger → Pago = Suma asegurada × (Trigger − Ŷ) / Trigger.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        tabla_regla = pd.DataFrame({
            "Condición": [
                "Ŷ ≥ Trigger",
                "Déficit relativo < 15%",
                "Déficit relativo 15% – 34%",
                "Déficit relativo ≥ 35%"
            ],
            "Pago proporcional": [
                "0%",
                "1% – 14%",
                "15% – 34%",
                "35% – 100%"
            ],
            "Riesgo": [
                "🟢 Bajo",
                "🟡 Medio",
                "🔴 Alto",
                "🔴 Crítico"
            ]
        })

        st.markdown(
            render_tabla_clara(tabla_regla),
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <p class="nota">
                El riesgo actual estimado para el municipio seleccionado es <strong>{nivel}</strong>.
            </p>
            """,
            unsafe_allow_html=True
        )


# 18. Simulador de seguro indexado, estadísticas y variables usadas

with st.container(key="simulador_seguro"):
    st.markdown(
        """
        <h2 style="
            font-size:20px;
            margin:0 0 18px 0;
            color:#3e2723;
        ">
            Simulador de seguro indexado
        </h2>
        """,
        unsafe_allow_html=True
    )

    s1, s2, s3 = st.columns(3)

    with s1:
        st.markdown(
            f"""
            <div class="sim-card">
                <h3>Activación del seguro</h3>
                <p>{"Sí" if pct_pago > 0 else "No"}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            f"""
            <div class="sim-card">
                <h3>Porcentaje de pago</h3>
                <p>{pct_pago:.0%}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with s3:
        st.markdown(
            f"""
            <div class="sim-card">
                <h3>Pago estimado</h3>
                <p>${indemnizacion:,.0f}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📊 Ver estadísticas históricas del municipio"):
        stats_df = pd.DataFrame({
            "Estadístico": [
                "Media histórica",
                f"Trigger α = {alpha_pct}%",
                "Desv. estándar",
                "Mínimo histórico",
                "Máximo histórico",
                "Años observados"
            ],
            "Valor": [
                f"{rend_historico_pct:.1f}%",
                f"{trigger_pct:.1f}%",
                f"{rend_hist_std * 100:.1f} p.p.",
                f"{rend_hist_min * 100:.1f}%",
                f"{rend_hist_max * 100:.1f}%",
                str(n_obs)
            ]
        })

        st.markdown(
            render_tabla_clara(stats_df),
            unsafe_allow_html=True
        )

    with st.expander("🔍 Ver variables exactas usadas en la predicción"):
        registros = []

        for var in VARIABLES_MODELO:
            if var in VARS_CATEGORICAS:
                fuente = "Selección del usuario"
                valor_mostrado = str(X_input[var].iloc[0])
            elif var == "Rendimiento_lag1":
                fuente = "Ingresado manualmente en el tablero"
                valor_mostrado = f"{rendimiento_anterior_pct:.1f}%"
            else:
                fuente = f"Base consolidada - año {int(anio_base)}"
                valor_mostrado = f"{X_input[var].iloc[0]:.4f}"

            registros.append({
                "Variable": var,
                "Fuente": fuente,
                "Valor usado": valor_mostrado
            })

        variables_df = pd.DataFrame(registros)

        st.markdown(
            render_tabla_clara(variables_df),
            unsafe_allow_html=True
        )


# 19. Nota metodológica

st.markdown(
    f"""
    <div class="card">
        <h2>Información Técnica</h2>
        <p>
            Este tablero permite evaluar el riesgo agroclimático asociado a la producción de café
            en municipios de Cundinamarca y Nariño mediante un
            <strong>modelo predictivo LASSO</strong>. A partir de la selección del departamento,
            municipio, año de análisis, rendimiento del año anterior, valor asegurado y porcentaje
            de trigger, la herramienta estima el rendimiento esperado del cultivo y lo compara con
            el rendimiento histórico municipal.
        </p>
        <p>
            Con esta comparación, el tablero determina si se activa o no el seguro indexado,
            clasifica el nivel de riesgo climático y calcula una indemnización potencial proporcional
            al déficit estimado frente al umbral de activación. Además, permite al usuario analizar
            el comportamiento histórico del rendimiento y de las principales variables agroclimáticas,
            comparar el municipio frente al promedio departamental y consultar las variables exactas
            utilizadas en la predicción.
        </p>
        <p>
            <strong>Regla de activación:</strong>
            Trigger = {alpha_pct}% × histórico del municipio = {trigger_pct:.2f}%.
            Si el rendimiento estimado cae por debajo del trigger, el pago es proporcional al déficit:
            Pago = Suma asegurada × (Trigger − Ŷ) / Trigger.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)