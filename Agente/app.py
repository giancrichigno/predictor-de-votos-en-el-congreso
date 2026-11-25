import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

CARPETA = Path(__file__).resolve().parent.parent
MAYORIA = 129

PROMPT_LEY = """Sos un especialista en el Congreso argentino. Te paso el título o resumen de un proyecto que se trata en la Cámara de Diputados.

Devolvé un JSON con tres claves:
- "origen": quién presentó el proyecto. "PE" si lo envió el Poder Ejecutivo, "JGM" si lo envió la Jefatura de Gabinete, "S" si viene del Senado (en revisión), "D" si lo presentó un diputado, "SIN_DATO" si no se puede saber.
- "es_mocion": 1 si es un trámite (moción de orden, tratamiento sobre tablas, vuelta a comisión, cambio del plan de labor), 0 si se vota la ley o sus artículos.
- "resumen": una oración que diga de qué trata.
"""

st.set_page_config(page_title="Predictor de votos", page_icon="🏛️", layout="wide")


@st.cache_data(show_spinner="Calculando el perfil de cada diputado...")
def cargar_perfiles():
    df = pd.read_csv(CARPETA / "Datos" / "dataset_features.csv", encoding="utf-8-sig", low_memory=False)
    df["fecha_dt"] = pd.to_datetime(df["fecha_dt"])
    df = df.sort_values(["fecha_dt", "votacion_id"])

    ultima = df["votacion_id"].iloc[-1]
    en_ejercicio = df.loc[df["votacion_id"] == ultima, "diputado"]
    fecha_ultima = df["fecha_dt"].iloc[-1]

    grupo = df[df["diputado"].isin(en_ejercicio)].groupby("diputado")
    perfiles = pd.DataFrame({
        "tasa_afirmativos_previos": grupo["es_afirmativo"].mean() * 100,
        "tasa_negativos_previos": grupo["es_negativo"].mean() * 100,
        "tasa_abstenciones_previas": grupo["es_abstencion"].mean() * 100,
        "tasa_ausencias_previas": grupo["es_ausente"].mean() * 100,
        "tasa_afirmativos_recientes": grupo["es_afirmativo"].apply(lambda s: s.tail(50).mean() * 100),
        "tasa_ausencias_recientes": grupo["es_ausente"].apply(lambda s: s.tail(50).mean() * 100),
        "total_votos_previos": grupo.size(),
        "macrocoalicion": grupo["macrocoalicion"].last(),
        "provincia": grupo["provincia"].last(),
    }).reset_index()
    perfiles["es_oficialismo"] = (perfiles["macrocoalicion"] == "LLA").astype(int)
    return perfiles, fecha_ultima


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo():
    return joblib.load(CARPETA / "modelo_random_forest.pkl")


def prioridad_modelo(nombre):
    puntaje = 100
    if "flash" in nombre: puntaje -= 50
    if "pro" in nombre: puntaje -= 20
    if "preview" in nombre or "latest" in nombre: puntaje += 30
    return puntaje


@st.cache_resource(show_spinner="Conectando con Gemini...")
def conectar_gemini(api_key):
    client = genai.Client(api_key=api_key)
    excluir = ["tts", "image", "embedding", "live", "robotics"]
    candidatos = [
        m.name for m in client.models.list()
        if "gemini" in m.name
        and not any(x in m.name for x in excluir)
        and "generateContent" in (m.supported_actions or [])
    ]
    for candidato in sorted(candidatos, key=prioridad_modelo):
        try:
            client.models.generate_content(model=candidato, contents="Respondé OK")
            return client, candidato
        except Exception:
            continue
    raise RuntimeError("Ningún modelo de Gemini respondió con esta clave.")


def clasificar_ley(client, modelo_gemini, texto):
    respuesta = client.models.generate_content(
        model=modelo_gemini,
        contents=[PROMPT_LEY, texto],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    ley = json.loads(respuesta.text)
    if isinstance(ley, list):
        ley = ley[0]

    origen = ley.get("origen", "SIN_DATO")
    if origen not in ["PE", "JGM", "S", "D"]:
        origen = "SIN_DATO"
    return origen, int(ley.get("es_mocion", 0)), ley.get("resumen", "")


def predecir(perfiles, modelo, origen, es_mocion):
    hoy = pd.Timestamp.today()
    entrada = perfiles.copy()
    entrada["origen"] = origen
    entrada["es_mocion"] = es_mocion
    entrada["anio"] = hoy.year
    entrada["mes"] = hoy.month
    entrada["probabilidad"] = modelo.predict_proba(entrada)[:, 1]
    entrada["a_favor"] = (entrada["probabilidad"] > 0.5).astype(int)
    return entrada


ORIGENES = {
    "PE": "Poder Ejecutivo",
    "JGM": "Jefatura de Gabinete",
    "S": "Senado (en revisión)",
    "D": "Un diputado",
    "SIN_DATO": "Sin dato",
}

# ---------------------------------------------------------------- página
st.title("¿Cómo votaría la Cámara de Diputados?")
st.caption(
    "Pegá el título o resumen de un proyecto. Gemini identifica quién lo presentó y si es una moción; "
    "un Random Forest entrenado con 764.389 votos (1993-2026) estima el voto de cada diputado en ejercicio."
)

with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Clave de Gemini", type="password", help="Se crea en aistudio.google.com/api-keys. No se guarda.")
    st.markdown("[Crear una clave](https://aistudio.google.com/api-keys)")
    st.divider()
    st.caption(
        "El modelo acierta 70,2% en votaciones de julio 2023 a octubre 2026, contra 57,3% de decir siempre afirmativo. "
        "No conoce el tema del proyecto: la predicción depende sobre todo de cada diputado."
    )

texto = st.text_area(
    "Proyecto",
    placeholder="Ej.: Proyecto de ley del Poder Ejecutivo de modernización laboral: modifica la Ley de Contrato de Trabajo.",
    height=110,
)
boton = st.button("Predecir", type="primary", disabled=not texto.strip())

if boton:
    if not api_key:
        st.warning("Ingresá tu clave de Gemini en la barra lateral.")
        st.stop()
    try:
        client, modelo_gemini = conectar_gemini(api_key)
        with st.spinner("Gemini está leyendo el proyecto..."):
            origen, es_mocion, resumen = clasificar_ley(client, modelo_gemini, texto)
    except Exception as error:
        st.error(f"No se pudo consultar a Gemini: {str(error)[:200]}")
        st.stop()

    perfiles, fecha_ultima = cargar_perfiles()
    resultado = predecir(perfiles, cargar_modelo(), origen, es_mocion)
    st.session_state["resultado"] = (resultado, origen, es_mocion, resumen, fecha_ultima)

if "resultado" in st.session_state:
    resultado, origen, es_mocion, resumen, fecha_ultima = st.session_state["resultado"]
    total = int(resultado["a_favor"].sum())
    diputados = len(resultado)
    dudosos = int(resultado["probabilidad"].between(0.4, 0.6).sum())

    st.info(f"**Gemini entendió:** {resumen}  \nOrigen: {ORIGENES[origen]} · {'Moción o trámite' if es_mocion else 'Votación de la ley'}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Votos a favor esperados", f"{total} de {diputados}")
    c2.metric(f"Mayoría absoluta ({MAYORIA})", "Sí" if total >= MAYORIA else "No",
              delta=f"{total - MAYORIA:+d} votos", delta_color="normal" if total >= MAYORIA else "inverse")
    c3.metric("Casos dudosos", dudosos, help="Diputados con probabilidad entre 40% y 60%.")

    por_bloque = (
        resultado.groupby("macrocoalicion")
        .agg(diputados=("diputado", "size"), a_favor=("a_favor", "sum"), probabilidad=("probabilidad", "mean"))
        .reset_index()
        .sort_values("diputados", ascending=False)
    )
    por_bloque["en_contra"] = por_bloque["diputados"] - por_bloque["a_favor"]
    largo = por_bloque.melt(
        id_vars="macrocoalicion", value_vars=["a_favor", "en_contra"], var_name="voto", value_name="cantidad"
    )
    largo["voto"] = largo["voto"].map({"a_favor": "A favor", "en_contra": "No a favor"})
    orden = por_bloque["macrocoalicion"].tolist()

    st.subheader("Por macrocoalición")
    grafico = (
        alt.Chart(largo)
        .mark_bar()
        .encode(
            y=alt.Y("macrocoalicion:N", sort=orden, title=None),
            x=alt.X("cantidad:Q", title="Diputados"),
            color=alt.Color(
                "voto:N",
                scale=alt.Scale(domain=["A favor", "No a favor"], range=["#2a78d6", "#c9ced6"]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            order=alt.Order("voto:N", sort="ascending"),
            tooltip=["macrocoalicion", "voto", "cantidad"],
        )
        .properties(height=280)
    )
    st.altair_chart(grafico, width="stretch")

    st.subheader("Diputado por diputado")
    filtro = st.multiselect("Filtrar por macrocoalición", orden)
    tabla = resultado if not filtro else resultado[resultado["macrocoalicion"].isin(filtro)]
    st.dataframe(
        tabla[["diputado", "macrocoalicion", "provincia", "probabilidad", "tasa_afirmativos_recientes"]]
        .sort_values("probabilidad", ascending=False),
        hide_index=True,
        width="stretch",
        column_config={
            "diputado": "Diputado",
            "macrocoalicion": "Macrocoalición",
            "provincia": "Provincia",
            "probabilidad": st.column_config.ProgressColumn(
                "Probabilidad a favor", format="percent", min_value=0, max_value=1
            ),
            "tasa_afirmativos_recientes": st.column_config.NumberColumn(
                "Afirmativos en sus últimos 50 votos", format="%.0f%%"
            ),
        },
    )
    st.caption(f"Diputados en ejercicio según la última votación registrada ({fecha_ultima:%d/%m/%Y}).")
