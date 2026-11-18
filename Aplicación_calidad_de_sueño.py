"""
Aplicación de Streamlit: calidad de sueño, hábitos cotidianos y trastornos del sueño.

Ejecución (terminal de VS Code, con el entorno virtual activo):
    python -m pip install streamlit pandas numpy scikit-learn joblib
    python -m streamlit run sleep_app.py        (también sirve:  python sleep_app.py)

(Altair, que usan los gráficos, se instala solo junto con Streamlit.)

Pensada para cualquier persona que quiera conocer su probabilidad de presentar algún trastorno del sueño.
Incluye una pestaña de introducción y colores pensados para distinguirse también con daltonismo.
El tema (fondo blanco, cuadros oscuros con letra blanca) y los colores van integrados en este archivo: no hace falta config.toml.
La evaluación es guiada, en 4 pasos:
  1. Tus datos:        un cuestionario corto con valores típicos por defecto.
  2. Tu resultado:     nivel de trastorno del sueño más probable y la probabilidad de cada nivel (resultado principal),
                       más la probabilidad de sentirse descansado y el rendimiento mental estimado.
  3. Tu posición:      comparación con otras personas (todas, de tu edad o de tu ocupación).
  4. Cómo mejorar:     qué cambios de hábitos ayudarían más y un simulador "¿y si...?".
Además, la pestaña "Sobre el modelo" explica en lenguaje sencillo cómo funciona y qué tan confiable es.

Preparación para publicar la aplicación:
  * Coloca sleep_health_dataset.csv y modelo_sueno.joblib junto a este script en el repositorio.
  * Los usuarios no necesitan subir archivos ni entrenar modelos; ambos recursos se cargan desde el servidor.

Modelos (HistGradientBoosting de scikit-learn), cada objetivo entrenado dos veces:
  * "completo": usa las 28 variables (incluye mediciones del sueño). Más preciso.
  * "hábitos":  usa 22 variables (sin calidad, REM, sueño profundo, latencia, despertares ni diferencia de
                fin de semana). Es el que usa la app por defecto y el simulador: si se dejaran las mediciones
                fijas, cambiar la cafeína o el alcohol casi no movería el resultado, porque esas mediciones son
                justamente la consecuencia de los hábitos.
Objetivos: riesgo de trastorno del sueño (4 niveles; resultado principal), probabilidad de sentirse descansado
y puntaje cognitivo del día siguiente.

Importante: los datos son SINTÉTICOS. Es una demostración técnica, no una herramienta de diagnóstico.
"""

from __future__ import annotations

import base64
import io
import mimetypes
import time
from html import escape as escapar_html
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.compose import make_column_transformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             log_loss, mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
SEMILLA = 42
CARPETA = Path(__file__).resolve().parent
RUTA_DATOS = CARPETA / "sleep_health_dataset.csv"
RUTA_MODELO = CARPETA / "modelo_sueno.joblib"
RUTA_IMAGEN_BANNER = CARPETA / "banner_sueno.jpg"  # Imagen de fondo del banner en alta calidad.

OBJ_RIESGO = "sleep_disorder_risk"
OBJ_DESCANSO = "felt_rested"
OBJ_COG = "cognitive_performance_score"

NIVELES = ["Healthy", "Mild", "Moderate", "Severe"]  # de menor a mayor riesgo
ETIQ_NIVEL = {"Healthy": "Sano", "Mild": "Leve", "Moderate": "Moderado", "Severe": "Severo"}
# Semáforo apto para daltonismo: azul (sano) -> amarillo -> naranja -> rojo (severo). Del frío al caliente, de calma a alarma;
# el azul se separa del resto aunque no se distinga el verde del rojo, y amarillo/naranja/rojo difieren también en luminosidad.
# Siempre van acompañados de texto. Se validó con simulación de protanopia, deuteranopia y tritanopia.
# (versión clara, para dibujar sobre los cuadros oscuros: barras, bordes y gráficos). Las tarjetas grandes usan COLORES_HERO.
# El rojo de Severo lleva algo de rosa: para quien no distingue el rojo se vuelve gris azulado y no se confunde con el naranja.
COLOR_NIVEL = {"Healthy": "#5787EB", "Mild": "#FEEC00", "Moderate": "#F49A31", "Severe": "#FF567B"}
ORDEN_ETIQ = [ETIQ_NIVEL[n] for n in NIVELES]
COLORES_ETIQ = [COLOR_NIVEL[n] for n in NIVELES]
TITULO_APP = "Descubre como tus hábitos y estilo de vida afectan tu calidad de sueño"

VARS_CAT = ["gender", "occupation", "country", "chronotype", "mental_health_condition",
            "season", "day_type"]
VARS_BIN = ["exercise_day", "sleep_aid_used", "shift_work"]
# Mediciones del sueño: son consecuencia de los hábitos, no hábitos en sí
VARS_MEDICIONES = ["sleep_quality_score", "rem_percentage", "deep_sleep_percentage",
                   "sleep_latency_mins", "wake_episodes_per_night", "weekend_sleep_diff_hrs"]
VARS_HABITOS = [
    "age", "gender", "occupation", "country", "bmi", "chronotype", "mental_health_condition",
    "heart_rate_resting_bpm", "caffeine_mg_before_bed", "alcohol_units_before_bed",
    "screen_time_before_bed_mins", "sleep_aid_used", "room_temperature_celsius",
    "exercise_day", "steps_that_day", "nap_duration_mins", "stress_score",
    "work_hours_that_day", "shift_work", "sleep_duration_hrs", "season", "day_type",
]
VARS_COMPLETO = VARS_HABITOS + VARS_MEDICIONES
VARS_NUM = [v for v in VARS_COMPLETO if v not in VARS_CAT]  # incluye las binarias (0/1)
COLUMNAS_REQUERIDAS = VARS_COMPLETO + [OBJ_RIESGO, OBJ_DESCANSO, OBJ_COG]

ETIQ_VAR = {
    "age": "Edad", "gender": "Género", "occupation": "Ocupación", "country": "País",
    "bmi": "IMC", "chronotype": "Cronotipo", "mental_health_condition": "Salud mental",
    "heart_rate_resting_bpm": "Frecuencia cardíaca en reposo (lpm)",
    "caffeine_mg_before_bed": "Cafeína antes de dormir (mg)",
    "alcohol_units_before_bed": "Alcohol antes de dormir (unidades)",
    "screen_time_before_bed_mins": "Pantallas antes de dormir (min)",
    "sleep_aid_used": "Usó ayuda para dormir", "room_temperature_celsius": "Temperatura del cuarto (°C)",
    "exercise_day": "Hizo ejercicio ese día", "steps_that_day": "Pasos del día",
    "nap_duration_mins": "Siesta (min)", "stress_score": "Estrés (1–10)",
    "work_hours_that_day": "Horas de trabajo del día", "shift_work": "Trabaja por turnos",
    "sleep_duration_hrs": "Horas de sueño", "season": "Estación", "day_type": "Tipo de día",
    "sleep_quality_score": "Calidad del sueño (1–10)", "rem_percentage": "Sueño REM (%)",
    "deep_sleep_percentage": "Sueño profundo (%)", "sleep_latency_mins": "Latencia para dormirse (min)",
    "wake_episodes_per_night": "Despertares por noche",
    "weekend_sleep_diff_hrs": "Diferencia de sueño en fin de semana (h)",
}
ETIQ_VALOR = {  # traducción de valores categóricos (solo para mostrar)
    "Male": "Masculino", "Female": "Femenino", "Other": "Otro",
    "Software Engineer": "Ing. de software", "Nurse": "Enfermería", "Doctor": "Medicina",
    "Teacher": "Docencia", "Lawyer": "Abogacía", "Driver": "Conducción", "Retired": "Jubilación",
    "Student": "Estudiante", "Freelancer": "Freelance", "Homemaker": "Labores del hogar",
    "Manager": "Gerencia", "Sales": "Ventas",
    "Morning": "Matutino", "Evening": "Vespertino", "Neutral": "Neutral",
    "Healthy": "Sana", "Anxiety": "Ansiedad", "Depression": "Depresión", "Both": "Ansiedad y depresión",
    "Spring": "Primavera", "Summer": "Verano", "Autumn": "Otoño", "Winter": "Invierno",
    "Weekday": "Entre semana", "Weekend": "Fin de semana",
}
PASOS = {
    "bmi": 0.1, "sleep_duration_hrs": 0.25, "sleep_quality_score": 0.1, "rem_percentage": 0.5,
    "deep_sleep_percentage": 0.5, "stress_score": 0.5, "work_hours_that_day": 0.5,
    "room_temperature_celsius": 0.5, "weekend_sleep_diff_hrs": 0.25, "alcohol_units_before_bed": 1.0,
    "caffeine_mg_before_bed": 10, "steps_that_day": 500, "screen_time_before_bed_mins": 1,
    "nap_duration_mins": 5,
}
FORMATOS = {"alcohol_units_before_bed": "%.0f", "sleep_duration_hrs": "%.2f",
            "weekend_sleep_diff_hrs": "%.2f"}

PERFILES_EJEMPLO = {
    "Perfil típico": {},
    "Abogado con mucho estrés": {
        "occupation": "Lawyer", "age": 38, "stress_score": 7.5, "work_hours_that_day": 11.0,
        "sleep_duration_hrs": 5.5, "caffeine_mg_before_bed": 200, "alcohol_units_before_bed": 2.0,
        "screen_time_before_bed_mins": 100, "exercise_day": False,
        "mental_health_condition": "Anxiety"},
    "Jubilado descansado": {
        "occupation": "Retired", "age": 66, "stress_score": 2.5, "work_hours_that_day": 0.0,
        "sleep_duration_hrs": 8.0, "caffeine_mg_before_bed": 0, "alcohol_units_before_bed": 0.0,
        "screen_time_before_bed_mins": 40, "exercise_day": True, "shift_work": False,
        "mental_health_condition": "Healthy"},
    "Enfermera con turnos": {
        "occupation": "Nurse", "gender": "Female", "shift_work": True, "stress_score": 7.0,
        "work_hours_that_day": 11.0, "sleep_duration_hrs": 5.0, "caffeine_mg_before_bed": 150,
        "screen_time_before_bed_mins": 90, "chronotype": "Evening",
        "mental_health_condition": "Anxiety"},
}

# Variables que se pueden mover en el simulador
PALANCAS_SIM = ["sleep_duration_hrs", "work_hours_that_day", "caffeine_mg_before_bed",
                "alcohol_units_before_bed", "screen_time_before_bed_mins", "stress_score", "exercise_day"]
# Escenarios rápidos. Valor fijo, ("rel", delta) = sumar/restar, ("tope", x) = como máximo x
ESCENARIOS = {
    "Sin cafeína": {"caffeine_mg_before_bed": 0},
    "Sin alcohol": {"alcohol_units_before_bed": 0},
    "Pantallas ≤ 30 min": {"screen_time_before_bed_mins": ("tope", 30)},
    "Dormir 7.5 h": {"sleep_duration_hrs": 7.5},
    "Estrés −2 puntos": {"stress_score": ("rel", -2)},
    "Hacer ejercicio": {"exercise_day": 1},
    "Jornada ≤ 8 h": {"work_hours_that_day": ("tope", 8)},
}


# --------------------------------------------------------------------------- #
# Lectura y preparación de datos
# --------------------------------------------------------------------------- #
def _detectar_separador(contenido: bytes) -> str:
    muestra = contenido[:65536].decode("utf-8", errors="ignore")
    primera = next((l for l in muestra.splitlines() if l.strip()), "")
    conteos = {s: primera.count(s) for s in [",", ";", "\t", "|"]}
    mejor = max(conteos, key=conteos.get)
    return mejor if conteos[mejor] > 0 else ","


def _leer_csv_bytes(contenido: bytes) -> pd.DataFrame:
    sep = _detectar_separador(contenido)
    ultimo_error: Exception | None = None
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(contenido), sep=sep, encoding=codificacion,
                               low_memory=False, keep_default_na=False, na_values=[""])
        except UnicodeDecodeError as e:
            ultimo_error = e
    raise ultimo_error  # type: ignore[misc]


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Valida columnas, fija los tipos y descarta filas sin objetivos válidos."""
    faltan = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if faltan:
        raise ValueError("Faltan estas columnas: " + ", ".join(faltan))
    d = df.copy()
    for c in VARS_NUM + [OBJ_COG, OBJ_DESCANSO]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in VARS_CAT + [OBJ_RIESGO]:
        s = d[c].astype("string").str.strip()
        d[c] = s.astype(object).where(s.notna(), np.nan)  # nulos como np.nan para scikit-learn
    d = d.dropna(subset=[OBJ_RIESGO, OBJ_DESCANSO, OBJ_COG])
    d = d[d[OBJ_RIESGO].isin(NIVELES)].reset_index(drop=True)
    if d.empty:
        raise ValueError("No quedaron filas válidas (revisa la columna sleep_disorder_risk).")
    d[OBJ_DESCANSO] = d[OBJ_DESCANSO].astype(int)
    return d


@st.cache_resource(show_spinner="Leyendo datos...")
def leer_bytes(contenido: bytes) -> pd.DataFrame:
    return preparar_datos(_leer_csv_bytes(contenido))


@st.cache_resource(show_spinner="Leyendo datos...")
def leer_ruta(ruta: str, modificado: float) -> pd.DataFrame:
    return preparar_datos(_leer_csv_bytes(Path(ruta).read_bytes()))


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def _ordenar_columnas(variables: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Categóricas primero y luego numéricas (el orden importa para el preprocesador)."""
    cat = [v for v in variables if v in VARS_CAT]
    num = [v for v in variables if v not in VARS_CAT]
    return cat + num, cat, num


def _pipeline_arboles(cat: list[str], num: list[str], clase):
    prep = make_column_transformer(
        (OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
        remainder="passthrough")
    modelo = clase(categorical_features=[True] * len(cat) + [False] * len(num),
                   max_iter=200, random_state=SEMILLA)
    return make_pipeline(prep, modelo)


def _pipeline_lineal(cat: list[str], num: list[str]):
    prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore"), cat),
                                   (StandardScaler(), num))
    return make_pipeline(prep, LogisticRegression(max_iter=1000))


def _pct(x: float) -> str:
    return f"{x:.1%}"


def entrenar_todo(df: pd.DataFrame) -> dict:
    """Entrena los 6 modelos (3 objetivos x 2 conjuntos de variables) y evalúa en datos de prueba."""
    t0 = time.time()
    tr, te = train_test_split(df, test_size=0.2, random_state=SEMILLA, stratify=df[OBJ_RIESGO])
    y_te = te[OBJ_RIESGO]
    moda = tr[OBJ_RIESGO].mode().iloc[0]
    azar = {"acc": float((y_te == moda).mean()),
            "f1": float(f1_score(y_te, [moda] * len(y_te), average="macro", zero_division=0))}

    modelos, columnas, metricas, importancia = {}, {}, {}, {}
    for nombre, variables in [("completo", VARS_COMPLETO), ("habitos", VARS_HABITOS)]:
        cols, cat, num = _ordenar_columnas(variables)
        riesgo = _pipeline_arboles(cat, num, HistGradientBoostingClassifier).fit(tr[cols], tr[OBJ_RIESGO])
        descanso = _pipeline_arboles(cat, num, HistGradientBoostingClassifier).fit(tr[cols], tr[OBJ_DESCANSO])
        cognitivo = _pipeline_arboles(cat, num, HistGradientBoostingRegressor).fit(tr[cols], tr[OBJ_COG])
        lineal = _pipeline_lineal(cat, num).fit(tr[cols], tr[OBJ_RIESGO])

        pred = riesgo.predict(te[cols])
        proba = riesgo.predict_proba(te[cols])
        pred_lin = lineal.predict(te[cols])
        etiquetas = [ETIQ_NIVEL[n] for n in NIVELES]
        reporte = pd.DataFrame(classification_report(
            y_te, pred, labels=NIVELES, output_dict=True, zero_division=0)).T
        reporte = reporte.loc[NIVELES, ["precision", "recall", "f1-score", "support"]].round(3)
        reporte.index = etiquetas
        matriz = pd.DataFrame(confusion_matrix(y_te, pred, labels=NIVELES),
                              index=[f"Real: {e}" for e in etiquetas],
                              columns=[f"Pred: {e}" for e in etiquetas])
        pred_cog = cognitivo.predict(te[cols])
        metricas[nombre] = {
            "acc": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
            "logloss": float(log_loss(y_te, proba, labels=list(riesgo.classes_))),
            "acc_lineal": float(accuracy_score(y_te, pred_lin)),
            "f1_lineal": float(f1_score(y_te, pred_lin, average="macro", zero_division=0)),
            "auc_descanso": float(roc_auc_score(te[OBJ_DESCANSO], descanso.predict_proba(te[cols])[:, 1])),
            "rmse_cog": float(mean_squared_error(te[OBJ_COG], pred_cog) ** 0.5),
            "r2_cog": float(r2_score(te[OBJ_COG], pred_cog)),
            "reporte": reporte, "matriz": matriz,
        }
        muestra = te.sample(min(3000, len(te)), random_state=SEMILLA)
        imp = permutation_importance(riesgo, muestra[cols], muestra[OBJ_RIESGO], n_repeats=3,
                                     random_state=SEMILLA, scoring="f1_macro", n_jobs=1)
        importancia[nombre] = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
        modelos[nombre] = {"riesgo": riesgo, "descanso": descanso, "cognitivo": cognitivo}
        columnas[nombre] = cols

    rangos, perfil_tipico = {}, {}
    for v in VARS_NUM:
        entero = bool(pd.api.types.is_integer_dtype(df[v]))
        mediana = float(df[v].median())
        rangos[v] = {"min": float(df[v].min()), "max": float(df[v].max()), "entero": entero}
        perfil_tipico[v] = (bool(mediana >= 0.5) if v in VARS_BIN
                            else int(round(mediana)) if entero else round(mediana, 2))
    niveles = {c: sorted(df[c].dropna().unique().tolist()) for c in VARS_CAT}
    for c in VARS_CAT:
        perfil_tipico[c] = df[c].mode().iloc[0]

    return {"id": str(time.time()), "modelos": modelos, "columnas": columnas, "metricas": metricas,
            "importancia": importancia, "rangos": rangos, "niveles": niveles,
            "perfil_tipico": perfil_tipico, "azar": azar, "n_train": len(tr), "n_test": len(te),
            "segundos": time.time() - t0, "version_sklearn": sklearn.__version__}


def guardar_modelo(bundle: dict) -> bool:
    try:
        joblib.dump(bundle, RUTA_MODELO)
        return True
    except OSError:
        return False


def cargar_modelo_guardado() -> dict | None:
    """Carga el modelo guardado si existe y se creó con la misma versión de scikit-learn."""
    if not RUTA_MODELO.exists():
        return None
    try:
        bundle = joblib.load(RUTA_MODELO)
        return bundle if bundle.get("version_sklearn") == sklearn.__version__ else None
    except Exception:
        return None


def predecir_perfil(bundle: dict, perfil: dict, conjunto: str, solo_riesgo: bool = False) -> dict:
    """Probabilidades por nivel de riesgo (+ prob. de sentirse descansado y puntaje cognitivo)."""
    m, cols = bundle["modelos"][conjunto], bundle["columnas"][conjunto]
    fila = pd.DataFrame([perfil])[cols]
    riesgo = m["riesgo"]
    probs = pd.Series(riesgo.predict_proba(fila)[0], index=riesgo.classes_).reindex(NIVELES)
    res = {"probs": probs, "descanso": np.nan, "cognitivo": np.nan}
    if not solo_riesgo:
        d = m["descanso"]
        res["descanso"] = float(d.predict_proba(fila)[0, list(d.classes_).index(1)])
        res["cognitivo"] = float(np.clip(m["cognitivo"].predict(fila)[0], 0, 100))
    return res


def nivel_esperado(res: dict) -> float:
    """Nivel de riesgo esperado, de 0 (sano) a 3 (severo): sirve para ordenar cambios, no se muestra como porcentaje."""
    return float(sum(i * res["probs"][n] for i, n in enumerate(NIVELES)))


# --------------------------------------------------------------------------- #
# Perfil (estado de los controles) y escenarios
# --------------------------------------------------------------------------- #
def _para_widget(var: str, valor, bundle: dict):
    """Ajusta un valor al tipo y rango que admite el control correspondiente."""
    if var in VARS_BIN:
        return bool(valor)
    if var in VARS_CAT:
        niveles = bundle["niveles"][var]
        return valor if valor in niveles else niveles[0]
    r = bundle["rangos"][var]
    v = min(max(float(valor), r["min"]), r["max"])
    return int(round(v)) if r["entero"] else float(round(v, 4))


def inicializar_perfil(bundle: dict) -> None:
    """Carga el perfil típico en los controles: completo si hay un modelo nuevo y, si no, solo lo que falte."""
    nuevo = st.session_state.get("perfil_bundle_id") != bundle["id"]
    for v in VARS_COMPLETO:
        clave = f"p_{v}"
        if nuevo or clave not in st.session_state:
            st.session_state[clave] = _para_widget(v, bundle["perfil_tipico"][v], bundle)
    if nuevo or "usar_mediciones" not in st.session_state:
        st.session_state["usar_mediciones"] = False
    if nuevo:
        st.session_state["perfil_bundle_id"] = bundle["id"]
        st.session_state["paso"] = 1
        st.session_state.pop("sim_firma", None)


def leer_perfil() -> dict:
    perfil = {}
    for v in VARS_COMPLETO:
        x = st.session_state[f"p_{v}"]
        perfil[v] = int(x) if v in VARS_BIN else x
    return perfil


def aplicar_perfil_ejemplo(nombre: str) -> None:
    """Callback de los botones de perfiles de ejemplo."""
    bundle = st.session_state["bundle"]
    valores = dict(bundle["perfil_tipico"])
    valores.update(PERFILES_EJEMPLO[nombre])
    for v in VARS_COMPLETO:
        st.session_state[f"p_{v}"] = _para_widget(v, valores[v], bundle)
    st.session_state["usar_mediciones"] = False


def resolver_cambio(var: str, spec, base, bundle: dict):
    if isinstance(spec, tuple):
        tipo, x = spec
        nuevo = base + x if tipo == "rel" else min(base, x)
    else:
        nuevo = spec
    valor = _para_widget(var, nuevo, bundle)
    return int(valor) if var in VARS_BIN else valor


def aplicar_escenario(nombre: str) -> None:
    """Callback de los botones de escenarios: mueve los controles del simulador."""
    bundle = st.session_state["bundle"]
    perfil = leer_perfil()
    for var, spec in ESCENARIOS[nombre].items():
        actual = st.session_state.get(f"s_{var}", perfil[var])
        st.session_state[f"s_{var}"] = _para_widget(
            var, resolver_cambio(var, spec, actual, bundle), bundle)


def restablecer_simulador() -> None:
    """Callback: fuerza que los controles del simulador vuelvan al perfil base."""
    st.session_state.pop("sim_firma", None)


def sincronizar_simulador(perfil: dict, bundle: dict) -> None:
    """Si cambió el perfil base (o faltan valores), los controles del simulador vuelven al perfil base."""
    firma = tuple(sorted((k, str(v)) for k, v in perfil.items()))
    faltan = any(f"s_{var}" not in st.session_state for var in PALANCAS_SIM)
    if faltan or st.session_state.get("sim_firma") != firma:
        for var in PALANCAS_SIM:
            st.session_state[f"s_{var}"] = _para_widget(var, perfil[var], bundle)
        st.session_state["sim_firma"] = firma


def ranking_palancas(bundle: dict, perfil: dict, conjunto: str = "habitos") -> pd.DataFrame:
    """Efecto de aplicar, de a uno, cada escenario: cuánto cambia el nivel esperado y en qué nivel quedaría la persona."""
    base = predecir_perfil(bundle, perfil, conjunto, solo_riesgo=True)
    filas = []
    for nombre, cambios in ESCENARIOS.items():
        nuevo = dict(perfil)
        for var, spec in cambios.items():
            nuevo[var] = resolver_cambio(var, spec, perfil[var], bundle)
        if all(nuevo[v] == perfil[v] for v in cambios):
            continue  # el perfil ya cumple ese escenario
        r = predecir_perfil(bundle, nuevo, conjunto, solo_riesgo=True)
        nivel = r["probs"].idxmax()
        filas.append({"cambio": nombre, "delta": nivel_esperado(r) - nivel_esperado(base),
                      "nivel": nivel, "p_nivel": float(r["probs"][nivel])})
    columnas = ["cambio", "delta", "nivel", "p_nivel"]
    if not filas:
        return pd.DataFrame(columns=columnas)
    return pd.DataFrame(filas, columns=columnas).sort_values("delta").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Estilo visual (sin emojis): tarjetas, banner y barras en HTML/CSS
# --------------------------------------------------------------------------- #
INK = "#0B2A30"        # texto sobre fondo blanco (turquesa casi negro)
PANEL = "#0B434C"      # fondo de los cuadros oscuros (turquesa oscuro)
BLANCO = "#FFFFFF"     # texto sobre los cuadros oscuros
COLORES_HERO = {       # (inicio del degradado, fin, color del texto) de la tarjeta principal según el nivel
    "Healthy": ("#3965F2", "#1D59E3", BLANCO), "Mild": ("#FFD600", "#EFC80A", INK),
    "Moderate": ("#F08522", "#E07810", INK), "Severe": ("#C4234F", "#A8053E", BLANCO),
}
# Paleta única. Los pares que deben distinguirse (niveles de riesgo, mejora/empeora, tú/otras personas) se eligieron
# para que se vean distintos con visión normal y con los tipos más comunes de daltonismo; no dependen solo del color.
PALETA = {
    "ink": INK, "panel": PANEL, "panel_rgb": "11,67,76", "panel2": "#0F5A66", "teal": "#0F766E", "menta": "#2DD4BF",
    "prim": "#2DD4BF", "tu": BLANCO, "grupo": "#8FB0B6", "descanso": "#5EEAD4", "cognitivo": "#E2E8F0",
    "nivel": COLOR_NIVEL, "hero": COLORES_HERO,
    "mejor": "#5787EB", "peor": "#F49A31", "similar": "#94A3B8",
    "mejor_txt": "#9DB8FF", "peor_txt": "#FDBA74", "similar_txt": "#CBD5E1",
    "bueno": "#5787EB", "malo": "#F49A31", "bueno_txt": "#9DB8FF", "malo_txt": "#FDBA74",
    "pico": "#CBD5E1", "neutro": "#94A3B8",
}


def paleta() -> dict:
    return PALETA


ETAPAS_FLUJO = ["Tus datos", "Tu resultado", "Tu posición", "Cómo mejorar"]
ETQ_NIVEL_CORTA = {"Healthy": "tener un sueño saludable", "Mild": "padecer un trastorno del sueño leve",
                   "Moderate": "padecer un trastorno del sueño moderado", "Severe": "padecer un trastorno del sueño severo"}
ETQ_NIVEL_PROB = {n: f"Probabilidad de {texto}" for n, texto in ETQ_NIVEL_CORTA.items()}
ETQ_NIVEL_BREVE = {"Healthy": "sueño saludable", "Mild": "trastorno leve", "Moderate": "trastorno moderado",
                   "Severe": "trastorno severo"}

ESTILOS = """
html{font-size:120% !important;color-scheme:light !important}
body{color-scheme:light !important}
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stMainBlockContainer"],[data-testid="stHeader"],[data-testid="stBottom"],[data-testid="stBottomBlockContainer"]{background:#FFFFFF !important}
[data-testid="stSidebar"],[data-testid="stSidebar"] > div{background:#F1F5F9 !important}
:where(.stApp,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] > p,[data-testid="stMarkdownContainer"] > ul,[data-testid="stMarkdownContainer"] > ol,[data-testid="stMarkdownContainer"] > blockquote,[data-testid="stMarkdownContainer"] li,[data-testid="stMarkdownContainer"] > h1,[data-testid="stMarkdownContainer"] > h2,[data-testid="stMarkdownContainer"] > h3,[data-testid="stMarkdownContainer"] > h4,[data-testid="stHeading"] *,[data-testid="stText"],[data-testid="stMetric"] *,[data-testid="stSpinner"] *,[data-testid="stHeader"] *,[data-testid="stWidgetLabel"] *,[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *,label[data-baseweb="checkbox"] *,label[data-baseweb="radio"] *,[data-testid="stAlert"] *,[data-testid="stFileUploaderDropzone"] *){color:__INK__ !important}
:where([data-testid="stHeader"] svg,[data-testid="stSidebar"] svg){fill:__INK__ !important}
/* El banner comienza en el borde superior; la barra de Streamlit queda transparente encima. */
[data-testid="stHeader"],[data-testid="stToolbar"]{background:transparent !important}
/* Estilo exclusivo del menú de tres puntos: botón circular claro, sin el cuadro oscuro. */
[data-testid="stMainMenu"]{background:transparent !important}
[data-testid="stMainMenu"] button,
[data-testid="stHeader"] button[aria-label*="Main menu"],
[data-testid="stToolbar"] button[aria-label*="Main menu"]{
    background:rgba(255,255,255,.96) !important;
    color:#0B2A30 !important;
    border:1px solid rgba(11,42,48,.22) !important;
    border-radius:50% !important;
    width:2.25rem !important;height:2.25rem !important;min-width:2.25rem !important;
    padding:.45rem !important;
    box-shadow:0 1px 5px rgba(0,0,0,.2) !important;
}
[data-testid="stMainMenu"] button svg,
[data-testid="stMainMenu"] button svg *,
[data-testid="stHeader"] button[aria-label*="Main menu"] svg,
[data-testid="stToolbar"] button[aria-label*="Main menu"] svg{
    color:#0B2A30 !important;fill:#0B2A30 !important;stroke:#0B2A30 !important;
}
[data-testid="stMainBlockContainer"]{padding-top:0 !important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{color:#475569 !important}
[data-testid="stAlert"]{border-radius:12px}
[data-testid="stExpander"] details,details[data-testid="stExpander"]{border:2px solid #475569 !important;border-radius:12px !important;background:#FFFFFF !important}
[data-testid="stFileUploaderDropzone"]{background:#FFFFFF !important;border:2px dashed #94A3B8 !important}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="select"] > div{background:#FFFFFF !important;border:1px solid #64748B !important;border-radius:8px !important}
[data-baseweb="base-input"]{background:#FFFFFF !important}
[data-baseweb="input"] input,[data-baseweb="base-input"] input,[data-baseweb="textarea"] textarea,[data-baseweb="select"] input,[data-baseweb="select"] div,[data-baseweb="select"] span{color:__INK__ !important;-webkit-text-fill-color:__INK__ !important}
[data-baseweb="select"] svg,[data-baseweb="input"] svg{fill:__INK__ !important}
[data-baseweb="input"] button,[data-testid="stNumberInputStepUp"],[data-testid="stNumberInputStepDown"]{background:#F1F5F9 !important;border:none !important}
[data-baseweb="input"] button *,[data-testid="stNumberInputStepUp"] *,[data-testid="stNumberInputStepDown"] *{color:__INK__ !important;fill:__INK__ !important}
label[data-baseweb="checkbox"] > span:first-of-type{border-color:#475569 !important}
[data-baseweb="popover"],[data-baseweb="popover"] > div,[data-baseweb="menu"],[data-baseweb="popover"] ul,[data-baseweb="tooltip"],[data-testid="stTooltipContent"]{background:#FFFFFF !important;color:__INK__ !important}
[data-baseweb="popover"] > div{border:2px solid __INK__ !important;border-radius:10px !important}
[data-baseweb="popover"] *,[data-testid="stTooltipContent"] *{color:__INK__ !important}
[data-baseweb="popover"] li:hover,[data-baseweb="popover"] [aria-selected="true"]{background:#CCFBF1 !important}
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] *{font-size:1.05rem !important}
[data-testid="stNumberInputField"],[data-testid="stSelectbox"] input,[data-testid="stSelectbox"] [data-baseweb="select"] div,[data-testid="stSelectbox"] [data-baseweb="select"] span,[data-testid="stTextInput"] input{font-size:1.05rem !important}
[data-testid="stSliderThumbValue"],[data-testid="stSliderThumbValue"] *{font-size:1.05rem !important;font-weight:600 !important}
[data-testid="stSliderTickBar"],[data-testid="stSliderTickBar"] *,[data-testid="stTickBarMin"],[data-testid="stTickBarMax"]{font-size:1rem !important}
[data-testid="stRadio"] label,[data-testid="stRadio"] label *,[data-testid="stCheckbox"] label,[data-testid="stCheckbox"] label *,label[data-baseweb="checkbox"] *,label[data-baseweb="radio"] *{font-size:1.05rem !important}
[data-testid="stSelectboxVirtualDropdown"],[data-testid="stSelectboxVirtualDropdown"] *,[role="listbox"] [role="option"],[role="listbox"] [role="option"] *,[data-baseweb="popover"] li,[data-baseweb="popover"] li *{font-size:1.05rem !important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{font-size:1.05rem !important}
[data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] *,[data-testid="stTooltipContent"],[data-testid="stTooltipContent"] *{font-size:1rem !important}
[data-testid="stTooltipIcon"] svg{width:1.1rem !important;height:1.1rem !important}
[role="tablist"],[data-baseweb="tab-list"]{gap:.7rem !important;border-bottom:none !important;box-shadow:none !important;padding:.3rem .2rem .8rem .2rem !important;background:transparent !important}
[role="tablist"]::before,[role="tablist"]::after,[role="tab"]::before,[role="tab"]::after,[data-testid="stTab"]::before,[data-testid="stTab"]::after{display:none !important}
[role="tablist"] div:empty,[role="tablist"] span:empty,[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none !important}
[role="tab"],[data-testid="stTab"]{height:auto !important;padding:.6rem 1.4rem !important;border:2px solid __TEAL__ !important;border-radius:12px !important;background:#FFFFFF !important;box-shadow:0 2px 6px rgba(__PANEL_RGB__,.2) !important;margin:0 !important;opacity:1 !important;cursor:pointer}
[role="tab"],[role="tab"] *,[data-testid="stTab"],[data-testid="stTab"] *{color:__PANEL__ !important;font-size:1.1rem !important;font-weight:700 !important}
[role="tab"]:hover,[data-testid="stTab"]:hover{background:#CCFBF1 !important}
[role="tab"][aria-selected="true"],[data-testid="stTab"][aria-selected="true"],[role="tab"][data-selected="true"]{background:linear-gradient(135deg,__PANEL__,__TEAL__) !important;border-color:__PANEL__ !important;box-shadow:0 3px 10px rgba(__PANEL_RGB__,.4) !important}
[role="tab"][aria-selected="true"],[role="tab"][aria-selected="true"] *,[data-testid="stTab"][aria-selected="true"],[data-testid="stTab"][aria-selected="true"] *,[role="tab"][data-selected="true"],[role="tab"][data-selected="true"] *{color:#FFFFFF !important}
button[kind="primary"],button[data-testid="stBaseButton-primary"]{background:linear-gradient(135deg,__PANEL__,__TEAL__) !important;border:2px solid __PANEL__ !important;border-radius:10px !important}
button[kind="primary"],button[kind="primary"] *,button[data-testid="stBaseButton-primary"],button[data-testid="stBaseButton-primary"] *{color:#FFFFFF !important}
button[kind="secondary"],button[kind="tertiary"],button[data-testid="stBaseButton-secondary"],button[data-testid="stBaseButton-tertiary"]{background:#FFFFFF !important;border:2px solid __TEAL__ !important;border-radius:10px !important}
button[kind="secondary"],button[kind="secondary"] *,button[kind="tertiary"],button[kind="tertiary"] *,button[data-testid="stBaseButton-secondary"],button[data-testid="stBaseButton-secondary"] *,button[data-testid="stBaseButton-tertiary"],button[data-testid="stBaseButton-tertiary"] *{color:__PANEL__ !important}
button[kind="secondary"]:hover,button[data-testid="stBaseButton-secondary"]:hover{background:#CCFBF1 !important}
button:disabled{opacity:.55}
[data-testid="stVegaLiteChart"],[data-testid="stArrowVegaLiteChart"]{background:__PANEL__ !important;border-radius:12px;overflow:hidden;box-shadow:0 0 0 2px __TEAL__,0 0 0 5px __MENTA__,0 4px 12px rgba(__PANEL_RGB__,.3)}
#vg-tooltip-element{background:#FFFFFF !important;color:__INK__ !important;border:2px solid __INK__ !important;border-radius:8px !important;font-size:16px !important}
#vg-tooltip-element td,#vg-tooltip-element .key{color:#475569 !important}
#vg-tooltip-element .value{color:__INK__ !important}
.tarjeta,.guia .item,.llamado,.alerta-nivel,.aviso,.panel,.tabla-datos{border:3px solid transparent;border-radius:14px;background:linear-gradient(__PANEL__,__PANEL__) padding-box,linear-gradient(135deg,__MENTA__,__TEAL__) border-box;box-shadow:inset 0 0 0 1px rgba(255,255,255,.22),0 4px 12px rgba(__PANEL_RGB__,.28);color:#FFFFFF}
.banner{width:100%;height:clamp(340px,27.5vw,440px);min-height:340px;box-sizing:border-box;display:flex;flex-direction:column;justify-content:center;background:__BANNER_BG__;background-position:center 44%;background-size:cover;color:#fff;padding:clamp(2rem,5vw,5rem);border-radius:0;margin:0 0 1rem 0;box-shadow:0 4px 14px rgba(__PANEL_RGB__,.3);overflow:hidden}
.banner h1{color:#FFFFFF !important;font-size:clamp(2.1rem,3.2vw,3.5rem);font-weight:800;line-height:1.14;max-width:1150px;text-shadow:0 2px 8px rgba(0,0,0,.38);margin:0 0 .65rem 0;padding:0}
.banner p{color:#CCFBF1 !important;margin:0;font-size:clamp(1rem,1.25vw,1.2rem);line-height:1.45;text-shadow:0 1px 5px rgba(0,0,0,.35)}
.intro-descripcion{font-size:1.12rem;line-height:1.6;color:__INK__;margin:.1rem 0 1rem 0}
.pasos{display:flex;gap:.6rem;flex-wrap:wrap;margin:.2rem 0 1rem 0}
.paso{flex:1;min-width:170px;display:flex;align-items:center;gap:.6rem;padding:.55rem .85rem;border:2px solid transparent;border-radius:12px;background:linear-gradient(__PANEL__,__PANEL__) padding-box,linear-gradient(135deg,__MENTA__,__TEAL__) border-box;color:#CBD5E1}
.paso .num{width:1.7rem;height:1.7rem;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-weight:700;font-size:.9rem;background:#1F6B76;color:#FFFFFF}
.paso .txt{font-weight:600;font-size:.95rem}
.paso.hecho{color:#FFFFFF}
.paso.hecho .num{background:__MENTA__;color:__INK__}
.paso.activo{background:linear-gradient(135deg,__TEAL__,#115E59) padding-box,linear-gradient(135deg,#FFFFFF,__MENTA__) border-box;color:#FFFFFF;box-shadow:0 3px 10px rgba(15,118,110,.45)}
.paso.activo .num{background:#FFFFFF;color:__TEAL__}
.titulo-paso h2{margin:.2rem 0 .1rem 0;padding:0;font-size:1.6rem;color:__INK__ !important}
.titulo-paso p{margin:0 0 .6rem 0;color:#334155 !important}
.seccion{display:flex;align-items:center;gap:.7rem;margin:1.3rem 0 .5rem 0}
.seccion .marca{width:6px;height:2.1rem;border-radius:4px;background:linear-gradient(__MENTA__,__TEAL__)}
.seccion .t{font-size:1.25rem;font-weight:700;line-height:1.2;color:__INK__}
.seccion .s{font-size:1rem;color:#475569}
.hero{color:#fff;border-radius:20px;padding:1.5rem 1.8rem;margin:.2rem 0 1rem 0;box-shadow:0 6px 18px rgba(0,0,0,.18)}
.hero.claro{box-shadow:inset 0 0 0 2px rgba(__PANEL_RGB__,.35),0 6px 18px rgba(0,0,0,.18)}
.hero .etq{font-size:1.05rem;opacity:.95}
.hero .num{font-size:3.67rem;font-weight:800;line-height:1.05;margin:.15rem 0 .6rem 0}
.hero .desglose{margin-top:1rem;display:flex;flex-direction:column;gap:.25rem}
.hero .desglose .item{font-size:1.05rem;line-height:1.4}
.hero .desglose .item b{font-size:1.2rem}
.hero .pista{background:rgba(255,255,255,.3);border-radius:999px;height:14px;overflow:hidden}
.hero .pista .relleno{background:#fff;height:100%;border-radius:999px}
.hero.claro .pista{background:rgba(__PANEL_RGB__,.2)}
.hero.claro .pista .relleno{background:__INK__}
.tarjeta{padding:1rem 1.2rem;border-left:8px solid __MENTA__;margin-bottom:.8rem}
.tarjeta .etq{font-size:.8rem;font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:#CBD5E1}
.tarjeta .num{font-size:1.9rem;font-weight:800;line-height:1.2;margin:.1rem 0;color:#FFFFFF}
.tarjeta .nota{font-size:.88rem;color:#CBD5E1;margin-top:.35rem;line-height:1.4}
.alerta-nivel,.aviso{padding:.9rem 1.1rem;margin:.4rem 0 .8rem 0;line-height:1.5;border-left:8px solid __MENTA__}
.panel{padding:.8rem 1.1rem;margin:.3rem 0 .9rem 0}
.pista-fina{background:rgba(255,255,255,.24);border-radius:999px;height:10px;overflow:hidden;margin-top:.4rem}
.pista-fina .relleno{height:100%;border-radius:999px;box-shadow:inset 0 0 0 1.5px rgba(255,255,255,.6)}
.fila-prob{display:flex;align-items:center;gap:.7rem;margin:.3rem 0;padding:.2rem .5rem;border-radius:10px;color:#FFFFFF;font-size:1.1rem}
.fila-prob.activa{background:rgba(45,212,191,.24)}
.fila-prob .nombre{width:6.6rem;font-weight:600}
.fila-prob .pista-fina{flex:1;margin-top:0;height:12px}
.fila-prob .pct{width:3.8rem;text-align:right;font-weight:700}
.fila-cambio{display:flex;align-items:center;gap:.7rem;margin:.4rem 0;color:#FFFFFF;font-size:1.1rem}
.fila-cambio .nombre{width:12rem;font-weight:600}
.fila-cambio .pista-fina{flex:1;margin-top:0;height:14px}
.fila-cambio .valor{width:15.5rem;text-align:right;font-weight:700}
.guia{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.8rem;margin:.3rem 0 1rem 0}
.guia .item{padding:1rem 1.1rem}
.guia .num{width:2rem;height:2rem;border-radius:50%;background:__MENTA__;color:__INK__;display:flex;align-items:center;justify-content:center;font-weight:700;margin-bottom:.55rem}
.guia .t{font-weight:700;font-size:1.08rem;margin-bottom:.25rem;color:#FFFFFF}
.guia .d{font-size:.95rem;line-height:1.45;color:#CBD5E1}
.chips{display:flex;flex-wrap:wrap;gap:.5rem;margin:.2rem 0 1rem 0}
.chip{color:#FFFFFF;border-radius:999px;padding:.35rem .9rem;font-size:.92rem;border:2px solid transparent;background:linear-gradient(__PANEL__,__PANEL__) padding-box,linear-gradient(135deg,__MENTA__,__TEAL__) border-box}
.llamado{padding:1rem 1.3rem;margin:1rem 0;font-size:1.05rem}
.tabla-datos{overflow-x:auto;margin:.3rem 0 .8rem 0}
.tabla-datos table{border-collapse:collapse;width:100%;font-size:.92rem;color:#FFFFFF}
.tabla-datos th{background:__PANEL2__;color:#FFFFFF;text-align:left;padding:.5rem .7rem;font-weight:700;border-bottom:2px solid __MENTA__}
.tabla-datos td{padding:.45rem .7rem;border-bottom:1px solid #1F6B76;color:#FFFFFF;text-align:left}
.tabla-datos th.idx{background:transparent;border-bottom:1px solid #1F6B76}
.tabla-datos tbody tr:last-child td,.tabla-datos tbody tr:last-child th{border-bottom:none}
@media (max-width:760px){.banner{height:320px;min-height:320px;padding:2rem 1.2rem;background-position:center 42%}.banner h1{font-size:clamp(1.8rem,6vw,2.4rem)}.fila-cambio{flex-wrap:wrap}.fila-cambio .nombre{width:100%}.fila-cambio .pista-fina{min-width:40%}.fila-cambio .valor{width:auto;flex:1;text-align:right}.hero{padding:1.2rem 1.2rem}[role="tab"],[data-testid="stTab"]{padding:.4rem .55rem !important}[role="tab"] *,[data-testid="stTab"] *{font-size:.9rem !important}[role="tablist"],[data-baseweb="tab-list"]{gap:.4rem !important}}
"""


def fondo_banner() -> str:
    """Devuelve el fondo del banner usando la imagen JPEG de alta calidad atenuada."""
    degradado = "linear-gradient(135deg, rgba(11,67,76,.46) 0%, rgba(15,118,110,.42) 100%)"
    if not RUTA_IMAGEN_BANNER.is_file():
        return "linear-gradient(135deg,#0B434C 0%,#0F766E 100%)"

    contenido = RUTA_IMAGEN_BANNER.read_bytes()
    tipo = mimetypes.guess_type(RUTA_IMAGEN_BANNER.name)[0] or "image/jpeg"
    imagen = base64.b64encode(contenido).decode("ascii")
    return f"{degradado}, url('data:{tipo};base64,{imagen}') center/cover no-repeat"


def construir_estilos(pal: dict) -> str:
    """Hoja de estilos con los colores de la paleta activa."""
    return (ESTILOS.replace("__PANEL_RGB__", pal["panel_rgb"]).replace("__PANEL2__", pal["panel2"])
            .replace("__PANEL__", pal["panel"]).replace("__INK__", pal["ink"]).replace("__TEAL__", pal["teal"])
            .replace("__MENTA__", pal["menta"]).replace("__BANNER_BG__", fondo_banner()))


# Etiquetas pensadas para el formulario (lenguaje cotidiano)
ETIQ_FORM = dict(ETIQ_VAR)
ETIQ_FORM.update({
    "age": "Edad (años)", "gender": "Género", "occupation": "Ocupación", "mental_health_condition": "Salud mental",
    "chronotype": "¿Prefieres la mañana o la tarde?", "sleep_duration_hrs": "Horas que duermes por noche",
    "work_hours_that_day": "Horas de trabajo en un día típico", "shift_work": "Trabajo por turnos o de noche",
    "sleep_aid_used": "Uso ayudas para dormir", "caffeine_mg_before_bed": "Cafeína antes de dormir (mg)",
    "alcohol_units_before_bed": "Alcohol antes de dormir (bebidas)",
    "screen_time_before_bed_mins": "Pantallas antes de dormir (minutos)", "stress_score": "Nivel de estrés",
    "exercise_day": "Hago ejercicio en un día típico", "bmi": "Índice de masa corporal (IMC)",
    "heart_rate_resting_bpm": "Pulso en reposo (latidos por minuto)",
    "room_temperature_celsius": "Temperatura del cuarto (°C)", "steps_that_day": "Pasos en un día típico",
    "nap_duration_mins": "Siesta (minutos)", "season": "Estación del año", "day_type": "Tipo de día",
})
AYUDAS = {
    "sleep_duration_hrs": "Promedio de horas que duermes cada noche.",
    "caffeine_mg_before_bed": "Una taza de café tiene alrededor de 95 mg.",
    "alcohol_units_before_bed": "Cantidad de bebidas alcohólicas en las horas previas a dormir.",
    "screen_time_before_bed_mins": "Tiempo con celular, televisión o computadora antes de acostarte.",
    "stress_score": "Valores más altos significan más estrés.",
    "shift_work": "Márcalo si trabajas en horarios rotativos o nocturnos.",
    "sleep_aid_used": "Por ejemplo pastillas, suplementos o infusiones para dormir.",
    "bmi": "Se calcula con tu peso y tu estatura. Si no lo sabes, déjalo como está.",
    "mental_health_condition": "Condición de salud mental diagnosticada, si la hay.",
}
# Bloques del formulario: (título, ayuda, variables). Los controles de la lista SLIDERS_FORM son deslizadores.
BLOQUES_FORM = [
    ("Sobre ti", "Datos básicos de tu perfil.",
     ["age", "gender", "occupation", "mental_health_condition", "chronotype"]),
    ("Tu sueño y tu trabajo", "Cuánto duermes y cómo es tu jornada.",
     ["sleep_duration_hrs", "work_hours_that_day", "shift_work", "sleep_aid_used"]),
    ("Antes de dormir", "Lo que haces en las horas previas a acostarte.",
     ["caffeine_mg_before_bed", "alcohol_units_before_bed", "screen_time_before_bed_mins"]),
    ("Tu día a día", "Estrés y actividad física.", ["stress_score", "exercise_day"]),
]
DETALLES = ["country", "bmi", "heart_rate_resting_bpm", "room_temperature_celsius", "steps_that_day",
            "nap_duration_mins", "season", "day_type"]
SLIDERS_FORM = {"sleep_duration_hrs", "work_hours_that_day", "caffeine_mg_before_bed",
                "alcohol_units_before_bed", "screen_time_before_bed_mins", "stress_score"}
assert sorted(sum((b[2] for b in BLOQUES_FORM), []) + DETALLES) == sorted(VARS_HABITOS)

# Hábitos que se comparan con el promedio: (variable, título, unidad)
HABITOS_COMPARAR = [
    ("sleep_duration_hrs", "Horas de sueño", "h"), ("stress_score", "Estrés", ""),
    ("caffeine_mg_before_bed", "Cafeína antes de dormir", "mg"),
    ("alcohol_units_before_bed", "Alcohol antes de dormir", "bebidas"),
    ("screen_time_before_bed_mins", "Pantallas antes de dormir", "min"),
]
# Factores del gráfico "riesgo según...". Solo los que muestran una relación clara con el riesgo.
FACTORES = {
    "Horas de sueño": {"tipo": "num", "col": "sleep_duration_hrs", "bins": [0, 5, 6, 7, 8, 9, 24],
                       "etiquetas": ["Hasta 5 h", "5 a 6 h", "6 a 7 h", "7 a 8 h", "8 a 9 h", "Más de 9 h"]},
    "Nivel de estrés": {"tipo": "num", "col": "stress_score", "bins": [0, 2, 4, 6, 8, 10.01],
                        "etiquetas": ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]},
    "Salud mental": {"tipo": "cat", "col": "mental_health_condition",
                     "orden": ["Healthy", "Anxiety", "Depression", "Both"]},
    "Horas de trabajo": {"tipo": "num", "col": "work_hours_that_day", "bins": [-1, 0, 6, 9, 12, 24],
                         "etiquetas": ["Sin trabajo", "1 a 6 h", "7 a 9 h", "10 a 12 h", "Más de 12 h"]},
    "Peso (IMC)": {"tipo": "num", "col": "bmi", "bins": [0, 18.5, 25, 30, 100],
                   "etiquetas": ["Bajo peso", "Normal", "Sobrepeso", "Obesidad"]},
    "Cafeína antes de dormir": {"tipo": "num", "col": "caffeine_mg_before_bed", "bins": [-1, 0, 100, 200, 1000],
                                "etiquetas": ["Nada", "Hasta 100 mg", "101 a 200 mg", "Más de 200 mg"]},
    "Alcohol antes de dormir": {"tipo": "num", "col": "alcohol_units_before_bed", "bins": [-1, 0, 1, 2, 100],
                                "etiquetas": ["Nada", "1 bebida", "2 bebidas", "3 o más"]},
    "Ocupación": {"tipo": "cat", "col": "occupation", "orden": None},
}


# --------------------------------------------------------------------------- #
# Componentes HTML
# --------------------------------------------------------------------------- #
TEMA_CLARO = {"theme.base": "light", "theme.primaryColor": "#0F766E", "theme.backgroundColor": "#FFFFFF",
              "theme.secondaryBackgroundColor": "#F1F5F9", "theme.textColor": INK}


def aplicar_tema_claro() -> None:
    """Fija el tema claro desde el propio código, sin config.toml. Usa un recurso interno de Streamlit (no oficial), por
    eso va protegido: si no funciona, la hoja de estilos igual deja todo legible. Recarga la página una sola vez."""
    if st.session_state.get("_tema_claro_intentado"):
        return
    st.session_state["_tema_claro_intentado"] = True
    try:
        cambio = False
        for clave, valor in TEMA_CLARO.items():
            if st.get_option(clave) != valor:
                st._config.set_option(clave, valor)
                cambio = True
        recargar = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
        if cambio and recargar is not None:
            recargar()
    except Exception:
        pass


def tema_oscuro_activo() -> bool:
    """True si Streamlit informa que el tema activo es oscuro (solo versiones recientes; si no se sabe, False)."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def mostrar_html(contenido: str) -> None:
    st.markdown(contenido, unsafe_allow_html=True)


def inyectar_estilos() -> None:
    mostrar_html(f"<style>{construir_estilos(paleta())}</style>")


def _ancho(x: float) -> str:
    return f"{max(0.0, min(1.0, float(x))) * 100:.1f}%"


def pct_txt(p: float) -> str:
    """Porcentaje de un nivel. Evita mostrar «100%» o «0%» por redondeo: usa «>99%» y «<1%» (texto HTML seguro)."""
    p = float(p)
    if 0 < p < 0.005:
        return "&lt;1%"
    if 0.995 <= p < 1:
        return "&gt;99%"
    return f"{p:.0%}"


def _num_txt(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def banner() -> None:
    mostrar_html(f'<div class="banner"><h1>{TITULO_APP}</h1><p>Responde unas preguntas sobre tus hábitos '
                 'y descubre tu probabilidad de presentar algún trastorno del sueño.</p></div>')


def indicador_pasos(actual: int) -> None:
    items = []
    for i, nombre in enumerate(ETAPAS_FLUJO, 1):
        clase = "activo" if i == actual else ("hecho" if i < actual else "")
        items.append(f'<div class="paso {clase}"><span class="num">{i}</span><span class="txt">{nombre}</span></div>')
    mostrar_html('<div class="pasos">' + "".join(items) + "</div>")


def titulo_paso(titulo: str, ayuda: str) -> None:
    mostrar_html(f'<div class="titulo-paso"><h2>{titulo}</h2><p>{ayuda}</p></div>')


def seccion(titulo: str, ayuda: str = "") -> None:
    sub = f'<div class="s">{ayuda}</div>' if ayuda else ""
    mostrar_html(f'<div class="seccion"><div class="marca"></div><div><div class="t">{titulo}</div>{sub}</div></div>')


def tarjeta_resultado(res: dict, nivel: str) -> None:
    """Tarjeta principal: el nivel más probable en grande (con su color) y, debajo, cada uno de los demás por separado."""
    c1, c2, texto = paleta()["hero"][nivel]
    claro = " claro" if texto != "#FFFFFF" else ""
    p = float(res["probs"][nivel])
    otros = "".join(f'<div class="item"><b>{pct_txt(res["probs"][n])}</b> de probabilidad de {ETQ_NIVEL_CORTA[n]}</div>'
                    for n in NIVELES if n != nivel)
    mostrar_html(
        f'<div class="hero{claro}" style="color:{texto};background:linear-gradient(135deg,{c1} 0%,{c2} 100%)">'
        f'<div class="etq">{ETQ_NIVEL_PROB[nivel]}</div><div class="num">{pct_txt(p)}</div>'
        f'<div class="pista"><div class="relleno" style="width:{_ancho(p)}"></div></div>'
        f'<div class="desglose">{otros}</div></div>')


def filas_probabilidades(probs: pd.Series, resaltar: str | None = None) -> None:
    pal = paleta()
    filas = []
    for n in NIVELES:
        p = float(probs[n])
        activa = " activa" if n == resaltar else ""
        filas.append(f'<div class="fila-prob{activa}"><span class="nombre">{ETIQ_NIVEL[n]}</span>'
                     f'<div class="pista-fina"><div class="relleno" style="width:{_ancho(p)};background:{pal["nivel"][n]}"></div></div>'
                     f'<span class="pct">{pct_txt(p)}</span></div>')
    mostrar_html('<div class="panel">' + "".join(filas) + "</div>")


def tarjeta_mini(etiqueta: str, valor: str, fraccion: float, color: str, nota: str = "") -> None:
    nota_html = f'<div class="nota">{nota}</div>' if nota else ""
    mostrar_html(f'<div class="tarjeta" style="border-left-color:{color}"><div class="etq">{etiqueta}</div>'
                 f'<div class="num">{valor}</div><div class="pista-fina"><div class="relleno" '
                 f'style="width:{_ancho(fraccion)};background:{color}"></div></div>{nota_html}</div>')


def tarjeta_info(titulo: str, texto: str, color: str | None = None) -> None:
    color = color or paleta()["prim"]
    mostrar_html(f'<div class="tarjeta" style="border-left-color:{color}"><div class="etq">{titulo}</div>'
                 f'<div class="nota" style="font-size:1rem;color:#FFFFFF">{texto}</div></div>')


def estado_habito(clave: str, valor: float, promedio: float) -> tuple[str, str]:
    """Clasifica un hábito frente al promedio del grupo: (estado, frase)."""
    if clave == "sleep_duration_hrs":
        if 7 <= valor <= 9:
            return "mejor", "Dentro del rango con menos trastornos (7 a 9 h)"
        return "peor", "Fuera del rango con menos trastornos (7 a 9 h)"
    if valor < promedio * 0.95:
        return "mejor", "Por debajo del promedio"
    if valor > promedio * 1.05 and valor > promedio:
        return "peor", "Por encima del promedio"
    return "similar", "Similar al promedio"


def tarjeta_comparacion(titulo: str, tu: float, promedio: float, unidad: str, estado: str, frase: str) -> None:
    pal = paleta()
    color, color_txt = pal[estado], pal[f"{estado}_txt"]
    u = f" {unidad}" if unidad else ""
    mostrar_html(f'<div class="tarjeta" style="border-left-color:{color}"><div class="etq">{titulo}</div>'
                 f'<div class="num">{_num_txt(tu)}{u}</div>'
                 f'<div class="nota">Promedio del grupo: {_num_txt(promedio)}{u}<br><b style="color:{color_txt}">{frase}</b></div></div>')


def barras_cambios(r: pd.DataFrame) -> None:
    """Barras horizontales: cuánto acerca (o aleja) cada cambio de un sueño saludable y en qué nivel quedaría la persona."""
    pal = paleta()
    maximo = max(float(r["delta"].abs().max()), 1e-9)
    filas = []
    for _, fila in r.iterrows():
        d = float(fila["delta"])
        color, color_txt = (pal["bueno"], pal["bueno_txt"]) if d < 0 else (pal["malo"], pal["malo_txt"])
        texto = f'{"Mejora" if d < 0 else "Empeora"} · {ETIQ_NIVEL[fila["nivel"]]} ({pct_txt(fila["p_nivel"])})'
        filas.append(f'<div class="fila-cambio"><span class="nombre">{fila["cambio"]}</span>'
                     f'<div class="pista-fina"><div class="relleno" style="width:{_ancho(abs(d) / maximo)};background:{color}"></div></div>'
                     f'<span class="valor" style="color:{color_txt}">{texto}</span></div>')
    mostrar_html('<div class="panel">' + "".join(filas) + "</div>")


def aviso(tipo: str, texto: str) -> None:
    """Cuadro de aviso (exito, info, alerta, error): cuadro oscuro con letra blanca y borde izquierdo de color."""
    color = {"info": paleta()["menta"], "exito": COLOR_NIVEL["Healthy"], "alerta": COLOR_NIVEL["Mild"],
             "error": COLOR_NIVEL["Severe"]}[tipo]
    mostrar_html(f'<div class="aviso" style="border-left-color:{color}">{" ".join(str(texto).split())}</div>')


def aviso_nivel(nivel: str, texto: str) -> None:
    """Mensaje con el color del nivel (azul, amarillo, naranja o rojo): borde grueso de color sobre cuadro oscuro. El texto lo dice siempre."""
    pal = paleta()
    mostrar_html(f'<div class="alerta-nivel" style="border-left-color:{pal["nivel"][nivel]}">{texto}</div>')


def mensaje_nivel(nivel: str) -> None:
    textos = {
        "Healthy": "Tu perfil se parece al de personas con un sueño saludable. Mantener tus hábitos es la mejor estrategia.",
        "Mild": "Tu perfil se parece al de personas con riesgo leve. Algunos cambios de hábitos podrían ayudarte a descansar mejor.",
        "Moderate": "Tu perfil se parece al de personas con riesgo moderado. Conviene revisar los hábitos que más pesan (paso 4) y, si te preocupa, consultar a un profesional de la salud.",
        "Severe": "Tu perfil se parece al de personas con riesgo severo. Conviene revisar tus hábitos (paso 4) y consultar a un profesional de la salud.",
    }
    aviso_nivel(nivel, textos[nivel])


# --------------------------------------------------------------------------- #
# Utilidades de interfaz
# --------------------------------------------------------------------------- #
def mostrar_grafico(grafico, ancho_completo: bool = True) -> None:
    """st.altair_chart compatible con versiones nuevas y antiguas de Streamlit."""
    if not ancho_completo:
        st.altair_chart(grafico)
        return
    try:
        st.altair_chart(grafico, width="stretch", theme=None)
    except Exception:
        st.altair_chart(grafico, use_container_width=True, theme=None)


def mostrar_tabla(df: pd.DataFrame, indice: bool = False) -> None:
    """Tabla en HTML con el mismo estilo (borde doble, fondo oscuro con letra blanca) que el resto; se ve igual con cualquier tema."""
    def celda(columna, valor) -> str:
        if isinstance(valor, (float, np.floating)):
            return f"{valor:,.0f}" if str(columna).lower() == "support" else f"{valor:.3f}"
        if isinstance(valor, (int, np.integer)):
            return f"{valor:,}"
        return escapar_html(str(valor))

    cabecera = ([""] if indice else []) + [str(c) for c in df.columns]
    thead = "".join(f"<th>{escapar_html(c)}</th>" for c in cabecera)
    filas = []
    for etiqueta, fila in df.iterrows():
        celdas = ([f'<th class="idx">{escapar_html(str(etiqueta))}</th>'] if indice else [])
        celdas += [f"<td>{celda(c, v)}</td>" for c, v in fila.items()]
        filas.append("<tr>" + "".join(celdas) + "</tr>")
    mostrar_html('<div class="tabla-datos"><table><thead><tr>' + thead + "</tr></thead><tbody>" + "".join(filas)
                 + "</tbody></table></div>")


def widget_variable(var: str, bundle: dict, prefijo: str = "p_", slider: bool = False):
    """Control adecuado al tipo de variable. El valor vive en st.session_state[prefijo + var]."""
    clave, etiqueta, ayuda = f"{prefijo}{var}", ETIQ_FORM[var], AYUDAS.get(var)
    if var in VARS_BIN:
        return st.checkbox(etiqueta, key=clave, help=ayuda)
    if var in VARS_CAT:
        return st.selectbox(etiqueta, bundle["niveles"][var], format_func=lambda v: ETIQ_VALOR.get(v, v),
                            key=clave, help=ayuda)
    r = bundle["rangos"][var]
    paso = PASOS.get(var, 1 if r["entero"] else 0.1)
    control = st.slider if slider else st.number_input
    if r["entero"]:
        return control(etiqueta, min_value=int(r["min"]), max_value=int(r["max"]),
                       step=max(1, int(paso)), key=clave, help=ayuda)
    return control(etiqueta, min_value=float(r["min"]), max_value=float(r["max"]), step=float(paso),
                   format=FORMATOS.get(var, "%.1f"), key=clave, help=ayuda)


def datos_paciente(bundle: dict) -> dict:
    """Perfil ingresado en el paso 1 y su predicción (modelo completo si el usuario aportó mediciones)."""
    perfil = leer_perfil()
    con_mediciones = bool(st.session_state.get("usar_mediciones"))
    res = predecir_perfil(bundle, perfil, "completo" if con_mediciones else "habitos")
    return {"perfil": perfil, "con_mediciones": con_mediciones, "res": res, "nivel": res["probs"].idxmax()}


def ir_a(paso: int) -> None:
    """Callback de los botones de navegación."""
    st.session_state["paso"] = paso


def conservar_estado() -> None:
    """Streamlit borra el valor de un control cuando deja de dibujarse (al cambiar de paso).
    Reasignarlo cada vez evita que el perfil ingresado se pierda al navegar."""
    for clave in [f"p_{v}" for v in VARS_COMPLETO] + ["usar_mediciones"]:
        if clave in st.session_state:
            st.session_state[clave] = st.session_state[clave]


NAVEGACION = {  # paso: (texto de "atrás", destino, texto de "siguiente", destino)
    1: (None, None, "Ver mi resultado", 2),
    2: ("Editar mis datos", 1, "Ver cómo me comparo", 3),
    3: ("Volver a mi resultado", 2, "Ver cómo mejorar", 4),
    4: ("Volver a mi posición", 3, "Empezar de nuevo", 1),
}


def botones_navegacion(paso: int, lugar: str) -> None:
    atras, dest_atras, sig, dest_sig = NAVEGACION[paso]
    c1, c2, _ = st.columns([2, 2, 3])
    if atras:
        c1.button(atras, key=f"nav_{paso}_{lugar}_atras", on_click=ir_a, args=(dest_atras,))
    c2.button(sig, key=f"nav_{paso}_{lugar}_sig", type="primary", on_click=ir_a, args=(dest_sig,))


# --------------------------------------------------------------------------- #
# Comparación con la población (con caché) y gráficos
# --------------------------------------------------------------------------- #
def _tabla_factor(d: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """% de personas con algún trastorno en cada grupo del factor."""
    if cfg["tipo"] == "num":
        grupo = pd.cut(d[cfg["col"]], cfg["bins"], labels=cfg["etiquetas"])
    else:
        grupo = d[cfg["col"]].map(lambda v: ETIQ_VALOR.get(v, v))
    g = d.groupby(grupo, observed=True)["alguno"].agg(["mean", "size"]).reset_index()
    g.columns = ["categoria", "pct", "personas"]
    g["categoria"] = g["categoria"].astype(str)
    if cfg["tipo"] == "cat":
        if cfg.get("orden"):
            posicion = {ETIQ_VALOR.get(o, o): i for i, o in enumerate(cfg["orden"])}
            g = g.assign(_o=g["categoria"].map(posicion)).sort_values("_o").drop(columns="_o")
        else:
            g = g.sort_values("pct")
    return g.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def calcular_posicion(_df: pd.DataFrame, firma: str, modo: str, edad: int, ocupacion: str) -> dict:
    """Agrega los datos del grupo de comparación. `firma` identifica el dataset para la caché."""
    d = _df
    if modo == "edad":
        d = d[d["age"].between(edad - 5, edad + 5)]
    elif modo == "ocupacion":
        d = d[d["occupation"] == ocupacion]
    if len(d) < 200:  # muy pocas personas para comparar: se usa toda la población
        d, modo = _df, "todas"
    d = d.assign(alguno=(d[OBJ_RIESGO] != "Healthy").astype(int))
    t = d[OBJ_RIESGO].value_counts().reindex(NIVELES).fillna(0).astype(int)
    h = d.assign(horas=(d["sleep_duration_hrs"] * 2).round() / 2).groupby("horas").agg(
        cognicion=(OBJ_COG, "mean"), personas=("age", "size")).reset_index()
    return {
        "modo": modo, "n": len(d), "alguno": float(d["alguno"].mean()),
        "riesgo": pd.DataFrame({"nivel": ORDEN_ETIQ, "personas": t.values, "pct": t.values / max(int(t.sum()), 1)}),
        "promedios": {c: float(d[c].mean()) for c, _, _ in HABITOS_COMPARAR},
        "factores": {nombre: _tabla_factor(d, cfg) for nombre, cfg in FACTORES.items()},
        "cognicion": h[h["personas"] >= 30],
    }


def categoria_usuario(cfg: dict, perfil: dict) -> str | None:
    """Grupo del factor al que pertenece el usuario (para resaltarlo en el gráfico)."""
    if cfg["tipo"] == "cat":
        valor = perfil[cfg["col"]]
        return ETIQ_VALOR.get(valor, valor)
    grupo = pd.cut([float(perfil[cfg["col"]])], cfg["bins"], labels=cfg["etiquetas"])[0]
    return None if pd.isna(grupo) else str(grupo)


def estilo_grafico(grafico):
    """Fondo turquesa oscuro y texto blanco en todo el gráfico, sin depender del tema de Streamlit (nunca texto oscuro sobre oscuro)."""
    return (grafico.configure(background=PANEL)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFontSize=16, titleFontSize=16, labelColor=BLANCO, titleColor=BLANCO,
                            domainColor="#9FBFC4", tickColor="#9FBFC4", gridColor="#1F6B76")
            .configure_legend(labelFontSize=16, titleFontSize=16, labelColor=BLANCO, titleColor=BLANCO,
                              symbolStrokeColor="#E2E8F0", symbolStrokeWidth=1)
            .configure_title(color=BLANCO))


def grafico_riesgo(t: pd.DataFrame, nivel_usuario: str | None = None) -> alt.Chart:
    """Distribución del riesgo en la población; el nivel más probable del usuario va con contorno y etiqueta."""
    pal = paleta()
    dom = {"scale": alt.Scale(domain=[0, float(t["pct"].max()) * 1.25])}
    base = alt.Chart(t).encode(x=alt.X("nivel:N", sort=ORDEN_ETIQ, title=None, axis=alt.Axis(labelAngle=0)))
    barras = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, stroke="#E2E8F0", strokeWidth=1,
                           strokeOpacity=.8).encode(
        y=alt.Y("pct:Q", axis=alt.Axis(format="%"), title="% de personas", **dom),
        color=alt.Color("nivel:N", scale=alt.Scale(domain=ORDEN_ETIQ, range=[pal["nivel"][n] for n in NIVELES]), legend=None),
        tooltip=[alt.Tooltip("nivel:N", title="Nivel"), alt.Tooltip("personas:Q", title="Personas", format=","),
                 alt.Tooltip("pct:Q", title="Porcentaje", format=".1%")])
    texto = base.mark_text(dy=-10, fontSize=17, color=BLANCO).encode(y=alt.Y("pct:Q", title="% de personas", **dom),
                                                      text=alt.Text("pct:Q", format=".0%"))
    grafico = barras + texto
    if nivel_usuario:
        fila = t[t["nivel"] == ETIQ_NIVEL[nivel_usuario]].assign(texto="Tú estás aquí")
        eje_x = alt.X("nivel:N", sort=ORDEN_ETIQ, title=None)
        eje_y = alt.Y("pct:Q", title="% de personas", **dom)
        contorno = alt.Chart(fila).mark_bar(fillOpacity=0, stroke=pal["tu"], strokeWidth=3).encode(x=eje_x, y=eje_y)
        marca = alt.Chart(fila).mark_text(dy=-32, fontWeight="bold", fontSize=17, color=pal["tu"]).encode(
            x=eje_x, y=eje_y, text="texto:N")
        grafico = grafico + contorno + marca
    return estilo_grafico(grafico.properties(height=320))


def grafico_factor(t: pd.DataFrame, usuario: str | None, angulo: int = 0) -> alt.Chart:
    """% de personas con algún trastorno por grupo; la barra del usuario va en blanco."""
    pal = paleta()
    t = t.assign(grupo=np.where(t["categoria"] == usuario, "Tú", "Otras personas"))
    orden = list(t["categoria"])
    escala_y = alt.Scale(domain=[0, 1])
    base = alt.Chart(t).encode(x=alt.X("categoria:N", sort=orden, title=None,
                                       axis=alt.Axis(labelAngle=angulo, labelLimit=210)))
    barras = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, stroke="#E2E8F0", strokeWidth=1, strokeOpacity=.7).encode(
        y=alt.Y("pct:Q", axis=alt.Axis(format="%"), title="Personas con algún trastorno", scale=escala_y),
        color=alt.Color("grupo:N", scale=alt.Scale(domain=["Otras personas", "Tú"], range=[pal["grupo"], pal["tu"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("categoria:N", title="Grupo"), alt.Tooltip("pct:Q", title="Con algún trastorno", format=".0%"),
                 alt.Tooltip("personas:Q", title="Personas", format=",")])
    texto = base.mark_text(dy=-10, fontSize=17, color=BLANCO).encode(
        y=alt.Y("pct:Q", title="Personas con algún trastorno", scale=escala_y), text=alt.Text("pct:Q", format=".0%"))
    return estilo_grafico((barras + texto).properties(height=340))


def grafico_cognicion(t: pd.DataFrame, horas_usuario: float, cognicion_usuario: float) -> alt.Chart:
    """Rendimiento mental promedio según las horas de sueño, con el punto estimado del usuario."""
    pal = paleta()
    ejes = dict(x=alt.X("horas:Q", title="Horas de sueño"),
                y=alt.Y("cognicion:Q", title="Rendimiento mental promedio (0 a 100)", scale=alt.Scale(zero=False)))
    linea = alt.Chart(t).mark_line(point=True, color=pal["prim"], strokeWidth=3).encode(
        tooltip=[alt.Tooltip("horas:Q", title="Horas"), alt.Tooltip("cognicion:Q", title="Rendimiento", format=".0f"),
                 alt.Tooltip("personas:Q", title="Personas", format=",")], **ejes)
    pico = t.loc[t["cognicion"].idxmax()]
    datos_pico = pd.DataFrame({"horas": [float(pico["horas"])], "cognicion": [float(pico["cognicion"])],
                               "texto": [f"Mejor rendimiento: {float(pico['horas']):g} h"]})
    punto = alt.Chart(datos_pico).mark_point(size=130, filled=True, color=pal["pico"]).encode(**ejes)
    etiqueta = alt.Chart(datos_pico).mark_text(dy=-18, fontWeight="bold", fontSize=17, color=pal["pico"]).encode(text="texto:N", **ejes)
    datos_tu = pd.DataFrame({"horas": [float(horas_usuario)], "cognicion": [float(cognicion_usuario)],
                             "texto": [f"Tú: {cognicion_usuario:.0f}"]})
    punto_tu = alt.Chart(datos_tu).mark_point(size=200, filled=True, color=pal["tu"], stroke=pal["prim"], strokeWidth=2).encode(**ejes)
    etiqueta_tu = alt.Chart(datos_tu).mark_text(dy=26, fontWeight="bold", fontSize=17, color=pal["tu"]).encode(text="texto:N", **ejes)
    return estilo_grafico((linea + punto + etiqueta + punto_tu + etiqueta_tu).properties(height=340))


def grafico_importancia(serie: pd.Series, n: int = 8) -> alt.Chart:
    pal = paleta()
    t = pd.DataFrame({"variable": [ETIQ_VAR.get(v, v) for v in serie.index[:n]], "importancia": serie.values[:n]})
    return estilo_grafico(alt.Chart(t).mark_bar(color=pal["prim"], cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
        x=alt.X("importancia:Q", title="Cuánto empeora el modelo si se ignora este factor"),
        y=alt.Y("variable:N", sort="-x", title=None),
        tooltip=[alt.Tooltip("variable:N", title="Factor"),
                 alt.Tooltip("importancia:Q", title="Importancia", format=".3f")]).properties(height=320))


# --------------------------------------------------------------------------- #
# Pasos de la evaluación guiada
# --------------------------------------------------------------------------- #
def paso_datos(bundle: dict) -> None:
    titulo_paso("Cuéntanos sobre ti", "Responde pensando en tu rutina habitual. Si no sabes un dato, déjalo como está: "
                                      "usaremos un valor típico.")
    st.markdown("**¿Quieres ver cómo funciona? Prueba con un ejemplo:**")
    botones = st.columns(len(PERFILES_EJEMPLO))
    for col, nombre in zip(botones, PERFILES_EJEMPLO):
        with col:
            st.button(nombre, key=f"btn_perfil_{nombre}", on_click=aplicar_perfil_ejemplo, args=(nombre,))
    for titulo, ayuda, variables in BLOQUES_FORM:
        seccion(titulo, ayuda)
        cols = st.columns(3)
        for i, var in enumerate(variables):
            with cols[i % 3]:
                widget_variable(var, bundle, slider=var in SLIDERS_FORM)
    st.write("")
    with st.expander("Más detalles (opcional)"):
        st.caption("Si no conoces estos datos, puedes dejarlos como están.")
        cols = st.columns(3)
        for i, var in enumerate(DETALLES):
            with cols[i % 3]:
                widget_variable(var, bundle)
    with st.expander("Tengo datos de un reloj o aplicación de sueño (opcional)"):
        st.checkbox("Incluir estos datos en mi evaluación (mejora la precisión)", key="usar_mediciones")
        cols = st.columns(3)
        for i, var in enumerate(VARS_MEDICIONES):
            with cols[i % 3]:
                widget_variable(var, bundle)
    botones_navegacion(1, "abajo")


def paso_resultado(bundle: dict, df: pd.DataFrame | None) -> None:
    pal = paleta()
    pac = datos_paciente(bundle)
    res, nivel = pac["res"], pac["nivel"]
    titulo_paso("Tu resultado", "Esta es la estimación del modelo con los datos que ingresaste.")
    botones_navegacion(2, "arriba")
    izq, der = st.columns([3, 2])
    with izq:
        tarjeta_resultado(res, nivel)
        mensaje_nivel(nivel)
    with der:
        seccion("Probabilidad de cada nivel", "Cómo se reparte la probabilidad entre los cuatro niveles.")
        filas_probabilidades(res["probs"], resaltar=nivel)
    seccion("Otros resultados", "Dos estimaciones que acompañan a tu riesgo de trastorno.")
    c1, c2 = st.columns(2)
    nota_d = f"En el conjunto de datos, {df[OBJ_DESCANSO].mean():.0%} de las personas se sintió descansado." if df is not None else ""
    nota_c = (f"Escala de 0 a 100. El promedio del conjunto de datos es {df[OBJ_COG].mean():.0f}."
              if df is not None else "Escala de 0 a 100.")
    with c1:
        tarjeta_mini("Probabilidad de sentirte descansado", f"{res['descanso']:.0%}", res["descanso"], pal["descanso"], nota_d)
    with c2:
        tarjeta_mini("Rendimiento mental estimado al día siguiente", f"{res['cognitivo']:.0f} / 100",
                     res["cognitivo"] / 100, pal["cognitivo"], nota_c)
    base = "tus hábitos y los datos de tu reloj o aplicación" if pac["con_mediciones"] else "tus hábitos"
    st.caption(f"Estimación basada en {base}. Se calcula con datos sintéticos: es una herramienta educativa, "
               "no un diagnóstico médico.")
    botones_navegacion(2, "abajo")


def paso_posicion(bundle: dict, df: pd.DataFrame | None) -> None:
    titulo_paso("Tu posición frente a otras personas",
                "Compara tus hábitos y tu riesgo con los de otras personas del conjunto de datos.")
    botones_navegacion(3, "arriba")
    if df is None:
        aviso("info", "Para comparar tu perfil se necesita el archivo de datos (sleep_health_dataset.csv). "
                "Pide a quien administra la aplicación que lo cargue.")
        return
    pac = datos_paciente(bundle)
    perfil = pac["perfil"]
    opciones = ["Todas las personas", "Personas de mi edad (±5 años)", "Personas de mi ocupación"]
    modos = dict(zip(opciones, ["todas", "edad", "ocupacion"]))
    eleccion = st.radio("Compararme con", opciones, horizontal=True, key="grupo_comparacion")
    edad = int(perfil["age"])
    ocupacion = str(perfil["occupation"])
    firma = f"{len(df)}_{df[OBJ_COG].sum():.2f}"
    d = calcular_posicion(df, firma, modos[eleccion], edad, ocupacion)
    nombre_grupo = {"todas": "todas las personas", "edad": f"personas de {edad - 5} a {edad + 5} años",
                    "ocupacion": f"personas con tu ocupación ({ETIQ_VALOR.get(ocupacion, ocupacion)})"}[d["modo"]]
    if d["modo"] != modos[eleccion]:
        st.caption("Había muy pocas personas en ese grupo, así que se muestra la comparación con todas.")
    st.caption(f"Comparando con {d['n']:,} personas: {nombre_grupo}.")

    seccion("Tus hábitos frente al promedio", "Así se comparan tus hábitos con los del grupo elegido.")
    cols = st.columns(len(HABITOS_COMPARAR))
    for col, (clave, titulo, unidad) in zip(cols, HABITOS_COMPARAR):
        tu, prom = float(perfil[clave]), d["promedios"][clave]
        estado, frase = estado_habito(clave, tu, prom)
        with col:
            tarjeta_comparacion(titulo, tu, prom, unidad, estado, frase)

    seccion("Dónde caes tú entre las demás personas",
            f"En este grupo, {d['alguno']:.0%} de las personas tiene algún trastorno del sueño. "
            f"Tu nivel más probable es {ETIQ_NIVEL[pac['nivel']].lower()} "
            f"({pct_txt(pac['res']['probs'][pac['nivel']])}).")
    mostrar_grafico(grafico_riesgo(d["riesgo"], pac["nivel"]))
    st.caption("Cada barra muestra qué porcentaje de personas está en cada nivel de riesgo. "
               "El contorno blanco marca tu nivel más probable.")

    seccion("Qué factores se relacionan con más trastornos",
            "Elige un factor y mira qué porcentaje de personas tiene algún trastorno en cada grupo.")
    factores = [f for f in FACTORES if not (f == "Ocupación" and d["modo"] == "ocupacion")]
    if st.session_state.get("factor_comparacion") not in factores:
        st.session_state.pop("factor_comparacion", None)
    factor = st.selectbox("Factor", factores, key="factor_comparacion")
    cfg = FACTORES[factor]
    usuario = categoria_usuario(cfg, perfil)
    tabla = d["factores"][factor]
    mostrar_grafico(grafico_factor(tabla, usuario, angulo=-30 if cfg["tipo"] == "cat" else 0))
    fila_tu = tabla[tabla["categoria"] == usuario]
    if len(fila_tu):
        st.caption(f"Tu grupo ({usuario}): {float(fila_tu['pct'].iloc[0]):.0%} de las personas tiene algún trastorno.")

    seccion("Dormir bien también se nota al día siguiente",
            "Rendimiento mental promedio según las horas de sueño (escala de 0 a 100).")
    if len(d["cognicion"]):
        horas = float(perfil["sleep_duration_hrs"])
        mostrar_grafico(grafico_cognicion(d["cognicion"], horas, pac["res"]["cognitivo"]))
        pico = d["cognicion"].loc[d["cognicion"]["cognicion"].idxmax()]
        st.caption(f"El rendimiento promedio es mejor con unas {float(pico['horas']):g} horas de sueño; dormir muy poco "
                   f"o demasiado lo reduce. Tú duermes {horas:g} h y tu rendimiento estimado es {pac['res']['cognitivo']:.0f}.")
    else:
        aviso("info", "Hay pocos datos para mostrar esta vista con el grupo elegido.")
    botones_navegacion(3, "abajo")


def mensaje_cambio(base: dict, esc: dict) -> None:
    """Resume el antes y después nivel por nivel (sin sumar niveles). El tono depende del nivel esperado."""
    cambios = []
    for n in NIVELES:
        antes, despues = float(base["probs"][n]), float(esc["probs"][n])
        if abs(despues - antes) >= 0.005:
            verbo = "sube" if despues > antes else "baja"
            cambios.append(f"{ETQ_NIVEL_BREVE[n]} {verbo} de {pct_txt(antes)} a {pct_txt(despues)}")
    delta = nivel_esperado(esc) - nivel_esperado(base)
    if not cambios or abs(delta) < 0.02:
        aviso("info", "Estos cambios casi no modifican tus probabilidades.")
        return
    if len(cambios) > 1:   # "A, la de B y la de C"
        cambios = [", la de ".join(cambios[:-1]), cambios[-1]]
    texto = "Con estos cambios, la probabilidad de " + " y la de ".join(cambios) + "."
    aviso("exito" if delta < 0 else "alerta", texto)


def paso_mejorar(bundle: dict) -> None:
    pal = paleta()
    pac = datos_paciente(bundle)          # el mismo cálculo del paso 2 (con o sin las mediciones del reloj o aplicación)
    perfil, base = pac["perfil"], pac["res"]
    conjunto = "completo" if pac["con_mediciones"] else "habitos"
    fuente = "tus hábitos y los datos de tu reloj o aplicación" if pac["con_mediciones"] else "tus hábitos"
    titulo_paso("Cómo mejorar", "Mira qué cambios podrían ayudarte más y prueba combinaciones. "
                                f"Son estimaciones del modelo con base en {fuente}.")
    botones_navegacion(4, "arriba")
    sincronizar_simulador(perfil, bundle)

    seccion("Los cambios que más podrían ayudarte",
            "Cada barra muestra cuánto cambiaría tu resultado si haces solo ese cambio (mientras más larga, mayor el efecto). "
            "A la derecha, si mejora o empeora y el nivel más probable en el que quedarías.")
    ranking = ranking_palancas(bundle, perfil, conjunto)
    ranking = ranking[ranking["delta"].abs() >= 0.02].reset_index(drop=True)
    if ranking.empty:
        aviso("info", "Ningún cambio individual modifica de forma apreciable tu resultado. "
                "Si ya es bueno, mantener tus hábitos es la mejor estrategia.")
    else:
        barras_cambios(ranking)
        st.caption("Es una estimación del modelo: la probabilidad entre paréntesis es la del nivel indicado.")

    seccion("Prueba tus propios cambios", "Elige un cambio rápido o ajusta los controles. Se pueden combinar.")
    nombres = list(ESCENARIOS)
    for fila in (nombres[:4], nombres[4:]):
        cols = st.columns(4)
        for col, nombre in zip(cols, fila):
            with col:
                st.button(nombre, key=f"btn_esc_{nombre}", on_click=aplicar_escenario, args=(nombre,))
    st.button("Volver a mis datos actuales", key="btn_reset_sim", on_click=restablecer_simulador)
    cols = st.columns(3)
    for i, var in enumerate(PALANCAS_SIM):
        with cols[i % 3]:
            widget_variable(var, bundle, prefijo="s_", slider=var != "exercise_day")

    escenario = dict(perfil)
    for var in PALANCAS_SIM:
        x = st.session_state[f"s_{var}"]
        escenario[var] = int(x) if var in VARS_BIN else x
    esc = predecir_perfil(bundle, escenario, conjunto)

    seccion("Antes y después")
    c1, c2 = st.columns(2)
    for col, titulo, res in [(c1, "Ahora", base), (c2, "Con tus cambios", esc)]:
        with col:
            nivel = res["probs"].idxmax()
            p_nivel = float(res["probs"][nivel])
            tarjeta_mini(f"{titulo}: {ETQ_NIVEL_PROB[nivel].lower()}", pct_txt(p_nivel), p_nivel, pal["nivel"][nivel])
            filas_probabilidades(res["probs"], resaltar=nivel)
            st.markdown(f"Descansado: **{res['descanso']:.0%}**  \nRendimiento mental: **{res['cognitivo']:.0f} / 100**")
    mensaje_cambio(base, esc)
    nota_med = (" Los datos de tu reloj o aplicación se mantienen como los ingresaste; solo cambian los hábitos que modifiques."
                if pac["con_mediciones"] else "")
    st.caption(f"Esta simulación usa {fuente}.{nota_med} Son asociaciones que aprendió el modelo con datos sintéticos, "
               "no efectos garantizados ni consejo médico.")
    botones_navegacion(4, "abajo")


def flujo(bundle: dict | None, df: pd.DataFrame | None) -> None:
    if bundle is None:
        aviso("info", "La aplicación todavía no tiene disponible el modelo entrenado. Inténtalo más tarde o contacta con quien administra la herramienta.")
        return
    conservar_estado()
    inicializar_perfil(bundle)
    paso = int(st.session_state.setdefault("paso", 1))
    indicador_pasos(paso)
    if paso == 1:
        paso_datos(bundle)
    elif paso == 2:
        paso_resultado(bundle, df)
    elif paso == 3:
        paso_posicion(bundle, df)
    else:
        paso_mejorar(bundle)


# --------------------------------------------------------------------------- #
# Introducción
# --------------------------------------------------------------------------- #
def pestana_introduccion() -> None:
    """Bienvenida breve: qué hace la aplicación, cómo se usa y por dónde empezar (sin cifras ni resultados)."""
    titulo_paso("Te damos la bienvenida", "")
    mostrar_html('<div class="intro-descripcion">¿Alguna vez ha sentido que no está descansado bien? ¿Se siente cansado al día siguiente o siente que no rinde tanto si no durmió bien la noche anterior? Esto podría deberse a hábitos que practica en el día a día y que afectan directamente su calidad de sueño. Con esta herramienta usted podrá ver claramente qué está afectando su calidad de sueño y cómo podría mejorarla a través de hábitos más saludables</div>')

    seccion("Cómo funciona", "Cuatro pasos sencillos. En cada uno puedes volver atrás.")
    pasos = [
        ("📋", "Tus datos", "Cuéntanos sobre tu rutina: horas de sueño, estrés, cafeína, pantallas y trabajo."),
        ("📊", "Tu resultado", "Conoce tu nivel de riesgo más probable, tu probabilidad de sentirte descansado y tu rendimiento mental."),
        ("👥", "Tu posición", "Mira cómo te comparas con otras personas."),
        ("💡", "Cómo mejorar", "Prueba cambios en tus hábitos y mira cuánto podrían ayudarte."),
    ]
    mostrar_html('<div class="guia">' + "".join(
        f'<div class="item"><div class="num">{i}</div><div class="t">{emoji} {titulo}</div><div class="d">{texto}</div></div>'
        for i, (emoji, titulo, texto) in enumerate(pasos, 1)) + "</div>")

    mostrar_html('<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in [
        "Solo toma unos minutos", "No pide tu nombre ni datos de contacto",
        "Si no sabes un dato, déjalo como está"]) + "</div>")

    mostrar_html('<div class="llamado"><b>¿Listo para empezar?</b> Abre la pestaña «Mi evaluación», arriba, junto a '
                 '«Introducción».</div>')


# --------------------------------------------------------------------------- #
# Sobre el modelo
# --------------------------------------------------------------------------- #
def pagina_modelo(bundle: dict | None) -> None:
    if bundle is None:
        aviso("info", "Aún no hay un modelo entrenado.")
        st.markdown("Consulte el set de datos utilizado para entrenar el modelo "
                    "[aquí](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)")
        return
    pal = paleta()
    mc, mh, azar = bundle["metricas"]["completo"], bundle["metricas"]["habitos"], bundle["azar"]
    titulo_paso("Cómo funciona esta herramienta", "Una explicación sencilla de qué hace el modelo y qué tan confiable es.")
    c = st.columns(3)
    with c[0]:
        tarjeta_info("1. Aprende de ejemplos", f"Se entrenó con {bundle['n_train']:,} registros de personas ficticias: "
                     "sus hábitos, su descanso y su nivel de riesgo de trastorno del sueño.")
    with c[1]:
        tarjeta_info("2. Estima probabilidades", "Compara tu perfil con lo aprendido y calcula tres cosas: tu probabilidad de "
                     "cada nivel de riesgo, de sentirte descansado y de tu rendimiento mental.", pal["descanso"])
    with c[2]:
        tarjeta_info("3. Se prueba con datos nuevos", f"Se evaluó con {bundle['n_test']:,} registros que no vio mientras aprendía, "
                     "para medir cuánto acierta de verdad.", pal["cognitivo"])

    seccion("¿Qué tan confiable es?", "Cuántas veces acierta el modelo el nivel de riesgo en personas que no conocía.")
    c = st.columns(3)
    with c[0]:
        tarjeta_mini("Solo con tus hábitos", f"{mh['acc'] * 100:.0f} de cada 100", mh["acc"], pal["prim"],
                     "Es lo que usa la aplicación por defecto.")
    with c[1]:
        tarjeta_mini("Con datos de reloj o aplicación", f"{mc['acc'] * 100:.0f} de cada 100", mc["acc"], pal["cognitivo"],
                     "Más preciso, pero requiere datos que no todos tienen a diario.")
    with c[2]:
        tarjeta_mini("Referencia: decir siempre «Sano»", f"{azar['acc'] * 100:.0f} de cada 100", azar["acc"], pal["neutro"],
                     "Un modelo sin información acertaría esto. El nuestro lo supera.")

    seccion("¿Qué factores pesan más en el riesgo?",
            "Mide cuánto empeora el modelo si se ignora cada factor (modelo basado en hábitos).")
    mostrar_grafico(grafico_importancia(bundle["importancia"]["habitos"]))

    seccion("Qué debes tener en cuenta")
    st.markdown(
        "- Los datos son **sintéticos** (generados por computadora): los resultados no describen a personas reales.\n"
        "- Es una herramienta **educativa**, no un diagnóstico médico. Ante dudas, consulta a un profesional de la salud.\n"
        "- Los cambios del paso 4 son **asociaciones** que aprendió el modelo, no efectos garantizados.\n"
        "- La etiqueta de riesgo parece generada por reglas a partir de las variables, por eso el modelo la reconstruye con tanta precisión.")

    with st.expander("Detalles técnicos (para quien quiera profundizar)"):
        st.markdown("**Riesgo de trastorno del sueño (4 niveles)**")
        mostrar_tabla(pd.DataFrame([
            ("Azar: siempre «Sano»", "—", _pct(azar["acc"]), f"{azar['f1']:.3f}", "—"),
            ("Regresión logística", "Completo", _pct(mc["acc_lineal"]), f"{mc['f1_lineal']:.3f}", "—"),
            ("Gradient Boosting", "Completo (28)", _pct(mc["acc"]), f"{mc['f1']:.3f}", f"{mc['logloss']:.3f}"),
            ("Regresión logística", "Hábitos", _pct(mh["acc_lineal"]), f"{mh['f1_lineal']:.3f}", "—"),
            ("Gradient Boosting", "Hábitos (22)", _pct(mh["acc"]), f"{mh['f1']:.3f}", f"{mh['logloss']:.3f}"),
        ], columns=["Modelo", "Variables", "Accuracy", "F1 macro", "Log loss"]))
        st.markdown("**Otros objetivos**")
        mostrar_tabla(pd.DataFrame([
            ("Completo (28)", f"{mc['auc_descanso']:.3f}", f"{mc['rmse_cog']:.2f}", f"{mc['r2_cog']:.3f}"),
            ("Hábitos (22)", f"{mh['auc_descanso']:.3f}", f"{mh['rmse_cog']:.2f}", f"{mh['r2_cog']:.3f}"),
        ], columns=["Variables", "AUC «se sintió descansado»", "RMSE cognitivo", "R² cognitivo"]))
        sel = st.radio("Ver detalle del modelo", ["habitos", "completo"], horizontal=True,
                       format_func=lambda v: "Hábitos (22 variables)" if v == "habitos" else "Completo (28 variables)")
        m = bundle["metricas"][sel]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Matriz de confusión** (filas: real · columnas: predicho)")
            mostrar_tabla(m["matriz"], indice=True)
        with c2:
            st.markdown("**Métricas por nivel**")
            mostrar_tabla(m["reporte"].rename(columns=str.capitalize), indice=True)
        st.markdown(
            "**Glosario.** *Accuracy*: porcentaje de aciertos. *F1 macro*: calidad promedio en los cuatro niveles, "
            "dando el mismo peso a los poco frecuentes (0 a 1; más alto es mejor). *Log loss*: calidad de las "
            "probabilidades (más bajo es mejor). *AUC*: capacidad de separar los casos «sí» de los «no» (0.5 es azar, "
            "1 es perfecto). *RMSE*: error promedio en puntos del rendimiento mental. *R²*: proporción de la variación "
            "que explica el modelo.")

    st.markdown("Consulte el set de datos utilizado para entrenar el modelo "
                "[aquí](https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset)")


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #
def cargar_datos() -> tuple[pd.DataFrame | None, str]:
    """Carga automáticamente el CSV incluido junto al código en el servidor."""
    try:
        if RUTA_DATOS.exists():
            return leer_ruta(str(RUTA_DATOS), RUTA_DATOS.stat().st_mtime), RUTA_DATOS.name
        return None, ""
    except Exception as e:
        st.sidebar.error(f"No se pudo leer el conjunto de datos de referencia: {escapar_html(str(e))}")
        return None, ""



def mostrar_acerca_de() -> None:
    """Información breve y fuente de datos para quienes quieran conocer el proyecto."""
    with st.sidebar:
        st.markdown("### Acerca de esta herramienta")
        st.markdown(
            "Esta aplicación permite explorar cómo algunos hábitos y aspectos del estilo de vida "
            "se relacionan con la calidad del sueño. A partir de tus respuestas, estima el nivel "
            "de riesgo de trastornos del sueño, la probabilidad de sentirte descansado y el "
            "rendimiento cognitivo del día siguiente."
        )
        st.caption(
            "Esta es una herramienta educativa: usa datos sintéticos (generados por computadora) "
            "y no reemplaza la opinión de un profesional de la salud. Si tienes problemas "
            "persistentes para dormir, consulta a un especialista."
        )
        st.divider()
        st.markdown("**Dataset original**")
        st.markdown(
            "[Consultar el conjunto de datos en Kaggle](https://www.kaggle.com/datasets/"
            "mohankrishnathalla/sleep-health-and-daily-performance-dataset)"
        )


def main() -> None:
    aplicar_tema_claro()
    listo = RUTA_MODELO.exists() and RUTA_DATOS.exists()
    st.set_page_config(page_title=TITULO_APP, layout="wide",
                       initial_sidebar_state="collapsed" if listo else "expanded")
    inyectar_estilos()
    banner()

    # En la versión pública, los recursos se leen directamente desde la carpeta del servidor.
    df, _ = cargar_datos()
    if "bundle" not in st.session_state:
        st.session_state["bundle"] = cargar_modelo_guardado()
    bundle = st.session_state.get("bundle")

    mostrar_acerca_de()

    tab_intro, tab_eval, tab_modelo = st.tabs(["Introducción", "Mi evaluación", "Sobre el modelo"])
    with tab_intro:
        pestana_introduccion()
    with tab_eval:
        flujo(bundle, df)
    with tab_modelo:
        pagina_modelo(bundle)


def _dentro_de_streamlit() -> bool:
    try:
        from streamlit.runtime import exists
        return bool(exists())
    except Exception:
        return True


if __name__ == "__main__":
    if _dentro_de_streamlit():
        main()
    else:      # se ejecutó con «python sleep_app.py»: abre la aplicación con el tema claro ya indicado
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve())]
                       + [f"--{clave}={valor}" for clave, valor in TEMA_CLARO.items()])
