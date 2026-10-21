"""
IT-AI Classifier - Interfaz interactiva (Streamlit)
===================================================

Muestra en tiempo real la matriz de confusión y los indicadores de desempeño
(accuracy, precision, recall, F1) cada vez que se cambia un parámetro del modelo.
También permite clasificar un ticket nuevo y probar un lote de tickets.

--------------------------------------------------------------------------
DESCRIPCIÓN DEL MODELO / PROTOTIPO
--------------------------------------------------------------------------
Objetivo
    Prototipo de clasificación automática de tickets de Soporte TI. A partir del
    texto libre del incidente (`descripcion`) el modelo sugiere la `categoria`
    (Acceso, Aplicaciones, Hardware, Impresoras, Redes, Seguridad, Software,
    Telefonía). Es un asistente: propone la categoría y el técnico la valida;
    la decisión final siempre es humana.

Tipo de problema
    Clasificación supervisada multiclase (8 categorías) sobre texto (NLP).

Entrada y salida
    Entrada : descripción del ticket (texto).
    Salida  : categoría sugerida + probabilidad (confianza) de cada categoría.

Arquitectura (Pipeline de scikit-learn)
    1) TF-IDF (TfidfVectorizer): convierte el texto en números. Pasa a
       minúsculas, elimina tildes y usa unigramas y bigramas (ngram_range=(1, 2)
       por defecto) para capturar palabras sueltas y pares como "no imprime".
       `min_df` descarta términos muy poco frecuentes.
    2) Regresión Logística (LogisticRegression): aprende qué términos se asocian
       a cada categoría y entrega una probabilidad por categoría. `C` controla
       la regularización (menor C = modelo más restringido).
    Se eligió este enfoque por ser simple, rápido e interpretable, y suficiente
    para textos cortos con vocabulario distintivo.

Datos
    Dataset sintético de tickets de Soporte TI (1000 registros, columnas
    `descripcion` y `categoria` como mínimo). Al ser sintético, las descripciones
    siguen plantillas parecidas, por lo que las métricas pueden ser más
    optimistas que con tickets reales.

Componentes del prototipo (pestañas de la interfaz)
    - Evaluación : métricas globales, matriz de confusión y métricas por categoría.
    - Errores    : tickets mal clasificados y confusiones más frecuentes.
    - En vivo    : clasifica un ticket escrito a mano y muestra la confianza.
    - Lote       : prueba un CSV de tickets con el modelo ya entrenado.
    Regla de apoyo al técnico: si la confianza es menor a 50 % se marca el ticket
    para revisión manual.

--------------------------------------------------------------------------
PROCESO DE ENTRENAMIENTO / VALIDACIÓN
--------------------------------------------------------------------------
    1) Carga y limpieza : se leen `descripcion` y `categoria` y se eliminan las
                          filas sin alguno de los dos valores.
    2) División         : 80 % entrenamiento / 20 % prueba (hold-out), con
                          `stratify` para conservar la proporción de categorías
                          y `random_state` fijo para que sea reproducible.
    3) Entrenamiento    : el Pipeline (TF-IDF + Regresión Logística) se ajusta
                          SOLO con el conjunto de entrenamiento; el vocabulario
                          TF-IDF también se aprende solo ahí (sin filtrar
                          información del conjunto de prueba).
    4) Validación       : se predicen los tickets de prueba (que el modelo no
                          vio) y se comparan con la categoría real mediante
                          accuracy, precision, recall y F1 (macro y por
                          categoría) y la matriz de confusión.
    5) Análisis de error: se revisan los tickets mal clasificados para entender
                          qué categorías se confunden.
    6) Prueba externa   : la pestaña "Lote" permite validar con tickets
                          adicionales sin reentrenar el modelo.
    Resultado de referencia (parámetros por defecto: 80/20, random_state=42,
    n-gramas 1-2, min_df=2, C=1.0): accuracy ≈ 0.98, con 4 errores de 200, todos
    Hardware clasificados como Impresoras.

    Limitaciones: una sola división (sin validación cruzada), conjunto de prueba
    pequeño (200 tickets) y datos sintéticos.

Ejecución:
    pip install streamlit pandas scikit-learn matplotlib joblib
    python -m streamlit run app_it_ai_classifier.py
"""

import io
import unicodedata

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CSV_POR_DEFECTO = "IT_AI_Classifier_dataset_sintetico.csv"

st.set_page_config(page_title="IT-AI Classifier", page_icon="🎫", layout="wide")
st.title("🎫 IT-AI Classifier")
st.caption(
    "Clasificación automática de tickets de Soporte TI · TF-IDF + Regresión Logística. "
    "El modelo sugiere la categoría; la decisión final es del técnico."
)

# --------------------------------------------------------------------------
# 1. Datos
# --------------------------------------------------------------------------
st.sidebar.header("📂 Datos")
archivo = st.sidebar.file_uploader("CSV de ENTRENAMIENTO (opcional, el dataset completo)", type="csv")


def _limpiar_columna(c) -> str:
    """'Descripción ' -> 'descripcion' (sin tildes, minúsculas, sin espacios)."""
    c = unicodedata.normalize("NFKD", str(c)).encode("ascii", "ignore").decode()
    return c.strip().lower().replace(" ", "_")


def leer_csv(contenido: bytes, requeridas: tuple) -> pd.DataFrame:
    """Lee un CSV probando separadores y codificaciones habituales (Excel usa ';')."""
    vistas = []
    for enc in ("utf-8-sig", "latin-1"):
        for sep in (",", ";", "\t", "|"):
            try:
                d = pd.read_csv(io.BytesIO(contenido), sep=sep, encoding=enc)
            except Exception:
                continue
            d.columns = [_limpiar_columna(c) for c in d.columns]
            if set(requeridas) <= set(d.columns):
                return d
            if not vistas:
                vistas = list(d.columns)
    raise ValueError(
        f"No encontré las columnas {list(requeridas)}. "
        f"Columnas que leí en el archivo: {vistas}"
    )


@st.cache_data(show_spinner=False)
def cargar_datos(contenido: bytes | None, ruta: str) -> pd.DataFrame:
    if contenido is None:
        with open(ruta, "rb") as f:
            contenido = f.read()
    df = leer_csv(contenido, ("descripcion", "categoria"))
    df = df.dropna(subset=["descripcion", "categoria"]).copy()
    df["descripcion"] = df["descripcion"].astype(str)
    df["categoria"] = df["categoria"].astype(str)
    return df


try:
    df = cargar_datos(archivo.getvalue() if archivo else None, CSV_POR_DEFECTO)
except FileNotFoundError:
    st.error(
        f"No encuentro `{CSV_POR_DEFECTO}`. Colócalo junto a este script o súbelo desde la barra lateral."
    )
    st.stop()
except ValueError as e:
    st.error(f"Problema con el CSV de entrenamiento. {e}")
    if "categoria_real" in str(e):
        st.info(
            "Este archivo parece una hoja de VALIDACIÓN (tiene `categoria_real` y `categoria_predicha`), "
            "no el dataset de entrenamiento. Quítalo de la barra lateral con la ✕ (la app volverá a usar "
            f"`{CSV_POR_DEFECTO}` de la carpeta) y súbelo en la pestaña 📦 Probar lote de tickets."
        )
    st.stop()

if len(df) < 100 or df["categoria"].value_counts().min() < 5:
    st.warning(
        f"El CSV de entrenamiento tiene solo {len(df)} tickets. Para entrenar usa el dataset completo "
        "(1000 tickets); los lotes pequeños se prueban en la pestaña 📦 Probar lote de tickets."
    )

# --------------------------------------------------------------------------
# 2. Parámetros (cada cambio recalcula todo al instante)
# --------------------------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del modelo")
test_size = st.sidebar.slider("Porcentaje de prueba", 0.10, 0.50, 0.20, 0.05)
seed = st.sidebar.number_input("random_state", 0, 9999, 42, 1)
ngram_max = st.sidebar.selectbox("N-gramas máximos", [1, 2, 3], index=1)
min_df = st.sidebar.slider("min_df (frecuencia mínima)", 1, 10, 2)
C = st.sidebar.select_slider(
    "C (regularización, mayor = menos)", options=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0], value=1.0
)
normalizar = st.sidebar.checkbox("Normalizar la matriz de confusión (%)", value=False)


# --------------------------------------------------------------------------
# PROCESO DE ENTRENAMIENTO / VALIDACIÓN (implementación)
# --------------------------------------------------------------------------
# `entrenar` ejecuta los pasos 2, 3 y 4 descritos al inicio del archivo:
#   división -> entrenamiento -> predicción sobre el conjunto de prueba.
# Se guarda en caché (`cache_resource`): si los parámetros no cambian, se
# reutiliza el modelo ya entrenado y la interfaz responde de inmediato.
@st.cache_resource(show_spinner="Entrenando modelo...")
def entrenar(textos, etiquetas, test_size, seed, ngram_max, min_df, C):
    # Paso 2 - División hold-out. `stratify` mantiene la proporción de cada
    # categoría en ambos conjuntos; `random_state` hace el resultado reproducible.
    X_train, X_test, y_train, y_test = train_test_split(
        textos, etiquetas, test_size=test_size, random_state=seed, stratify=etiquetas
    )
    # DESCRIPCIÓN DEL MODELO: Pipeline = TF-IDF (texto -> números) +
    # Regresión Logística (números -> categoría y probabilidades).
    modelo = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, ngram_max),
                    min_df=min_df,
                ),
            ),
            ("clasificador", LogisticRegression(C=C, max_iter=2000, random_state=seed)),
        ]
    )
    # Paso 3 - Entrenamiento: tanto el vocabulario TF-IDF como los coeficientes
    # de la regresión se aprenden únicamente con X_train / y_train.
    modelo.fit(X_train, y_train)
    # Paso 4 - Validación: predicciones sobre tickets que el modelo NO vio.
    y_pred = modelo.predict(X_test)
    return modelo, X_train, X_test, y_train, y_test, y_pred


modelo, X_train, X_test, y_train, y_test, y_pred = entrenar(
    tuple(df["descripcion"]), tuple(df["categoria"]), test_size, seed, ngram_max, min_df, C
)
clases = list(modelo.classes_)

# --------------------------------------------------------------------------
# 3. Indicadores globales  (VALIDACIÓN: métricas calculadas sobre el 20 % de prueba)
# --------------------------------------------------------------------------
# accuracy  : proporción de tickets clasificados correctamente.
# precision : de lo que el modelo asigna a una categoría, cuánto es correcto.
# recall    : de lo que realmente es de una categoría, cuánto logra encontrar.
# F1        : promedio armónico de precision y recall.
# "macro"   : se calcula por categoría y se promedia sin ponderar, para que
#             todas las categorías pesen igual.
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
c2.metric("Precision (macro)", f"{precision_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
c3.metric("Recall (macro)", f"{recall_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
c4.metric("F1 (macro)", f"{f1_score(y_test, y_pred, average='macro', zero_division=0):.3f}")
c5.metric("Tickets en prueba", f"{len(y_test)}", f"{len(X_train)} en entrenamiento", delta_color="off")

tab_eval, tab_errores, tab_vivo, tab_lote = st.tabs(
    ["📊 Evaluación", "🔍 Errores del modelo", "⚡ Clasificar ticket en vivo", "📦 Probar lote de tickets"]
)

# --------------------------------------------------------------------------
# 4. Evaluación: matriz de confusión + métricas por categoría  (VALIDACIÓN)
# --------------------------------------------------------------------------
# Matriz de confusión: filas = categoría real, columnas = categoría predicha.
# La diagonal son los aciertos; lo que queda fuera son los errores y muestra
# con qué categoría se confunde el modelo.
with tab_eval:
    izq, der = st.columns([1.2, 1])

    with izq:
        st.subheader("Matriz de confusión")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=clases,
            normalize="true" if normalizar else None,
            values_format=".0%" if normalizar else "d",
            xticks_rotation=45,
            cmap="Blues",
            ax=ax,
            colorbar=False,
        )
        ax.set_xlabel("Categoría predicha")
        ax.set_ylabel("Categoría real")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with der:
        st.subheader("Métricas por categoría")
        rep = pd.DataFrame(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        ).T
        rep["support"] = rep["support"].astype(int)
        por_clase = rep.loc[clases, ["precision", "recall", "f1-score", "support"]]
        st.dataframe(
            por_clase.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}"})
            .background_gradient(subset=["precision", "recall", "f1-score"], cmap="RdYlGn", vmin=0.5, vmax=1.0),
            use_container_width=True,
        )

        fig2, ax2 = plt.subplots(figsize=(6, 3.2))
        x = np.arange(len(clases))
        w = 0.27
        ax2.bar(x - w, por_clase["precision"], w, label="Precision")
        ax2.bar(x, por_clase["recall"], w, label="Recall")
        ax2.bar(x + w, por_clase["f1-score"], w, label="F1")
        ax2.set_xticks(x)
        ax2.set_xticklabels(clases, rotation=45, ha="right")
        ax2.set_ylim(0, 1.05)
        ax2.legend(ncol=3, fontsize=8, loc="lower center")
        ax2.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        with st.expander("Reporte de clasificación completo"):
            st.code(classification_report(y_test, y_pred, zero_division=0))

# --------------------------------------------------------------------------
# 5. Errores  (VALIDACIÓN - Paso 5: análisis de los tickets mal clasificados)
# --------------------------------------------------------------------------
with tab_errores:
    resultados = pd.DataFrame(
        {
            "descripcion": X_test.values if hasattr(X_test, "values") else list(X_test),
            "categoria_real": list(y_test),
            "categoria_predicha": list(y_pred),
        }
    )
    resultados["acierto"] = resultados["categoria_real"] == resultados["categoria_predicha"]
    errores = resultados[~resultados["acierto"]]

    st.subheader(f"Tickets mal clasificados: {len(errores)} de {len(resultados)}")
    if errores.empty:
        st.success("El modelo no cometió errores en el conjunto de prueba con estos parámetros.")
    else:
        pares = (
            errores.groupby(["categoria_real", "categoria_predicha"])
            .size()
            .reset_index(name="casos")
            .sort_values("casos", ascending=False)
        )
        st.markdown("**Confusiones más frecuentes**")
        st.dataframe(pares, use_container_width=True, hide_index=True)
        st.markdown("**Detalle de los tickets**")
        st.dataframe(errores.drop(columns="acierto"), use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        resultados.to_csv(index=False).encode("utf-8-sig"),
        file_name="resultados_prueba.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------
# 6. Clasificación en vivo  (USO DEL PROTOTIPO)
# --------------------------------------------------------------------------
# El modelo entrenado devuelve una probabilidad por categoría; la mayor es la
# categoría sugerida. Con confianza < 50 % se pide revisión manual del técnico.
with tab_vivo:
    st.subheader("Escribe un ticket y obtén la categoría sugerida")
    ejemplo = "No puedo iniciar sesión porque mi contraseña fue bloqueada."
    texto = st.text_area("Descripción del incidente", value=ejemplo, height=100)

    if texto.strip():
        probas = modelo.predict_proba([texto])[0]
        orden = np.argsort(probas)[::-1]
        mejor = clases[orden[0]]
        st.success(f"Categoría sugerida: **{mejor}** ({probas[orden[0]]:.1%} de confianza)")

        prob_df = pd.DataFrame({"categoria": [clases[i] for i in orden], "probabilidad": probas[orden]})
        st.bar_chart(prob_df.set_index("categoria"))

        if probas[orden[0]] < 0.5:
            st.warning("Confianza baja: conviene que el técnico revise esta categoría manualmente.")

    if st.button("💾 Guardar modelo entrenado (.joblib)"):
        joblib.dump(modelo, "IT_AI_Classifier_modelo_categoria.joblib")
        st.info("Modelo guardado como IT_AI_Classifier_modelo_categoria.joblib")

# --------------------------------------------------------------------------
# 7. Probar un lote de tickets (CSV aparte, no reentrena el modelo)
#    (VALIDACIÓN - Paso 6: prueba externa con tickets adicionales)
# --------------------------------------------------------------------------
with tab_lote:
    st.subheader("Probar un lote de tickets con el modelo ya entrenado")
    st.caption(
        "Sube un CSV con la columna `descripcion`. Si también trae `categoria`, "
        "se compara con la predicción y se calculan las métricas del lote."
    )
    lote_file = st.file_uploader("CSV con los tickets a probar", type="csv", key="lote")

    if lote_file is not None:
        try:
            lote = leer_csv(lote_file.getvalue(), ("descripcion",))
        except ValueError as e:
            st.error(str(e))
            lote = None
        if lote is not None:
            lote = lote.dropna(subset=["descripcion"]).copy()
            if "categoria" not in lote.columns and "categoria_real" in lote.columns:
                lote = lote.rename(columns={"categoria_real": "categoria"})
            if "categoria_predicha" in lote.columns:
                lote = lote.rename(columns={"categoria_predicha": "categoria_predicha_archivo"})
            probas_l = modelo.predict_proba(lote["descripcion"].astype(str))
            lote["categoria_predicha"] = np.array(clases)[probas_l.argmax(axis=1)]
            lote["confianza"] = probas_l.max(axis=1).round(3)
            lote["revisar_manual"] = np.where(lote["confianza"] < 0.5, "Sí", "")

            if "categoria" in lote.columns:
                lote["acierto"] = lote["categoria"] == lote["categoria_predicha"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Tickets en el lote", len(lote))
                m2.metric("Accuracy del lote", f"{lote['acierto'].mean():.3f}")
                m3.metric("Para revisión manual (<50 %)", int((lote["confianza"] < 0.5).sum()))

                figl, axl = plt.subplots(figsize=(7, 6))
                ConfusionMatrixDisplay.from_predictions(
                    lote["categoria"], lote["categoria_predicha"], labels=clases,
                    xticks_rotation=45, cmap="Blues", ax=axl, colorbar=False,
                )
                axl.set_xlabel("Categoría predicha")
                axl.set_ylabel("Categoría real")
                axl.set_title("Matriz de confusión del lote")
                figl.tight_layout()
                st.pyplot(figl)
                plt.close(figl)

            cols = [c for c in ["ticket_id", "descripcion", "categoria", "categoria_predicha",
                                "categoria_predicha_archivo", "confianza", "revisar_manual", "acierto"]
                    if c in lote.columns]
            st.dataframe(lote[cols], use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Descargar predicciones (CSV)",
                lote.to_csv(index=False).encode("utf-8-sig"),
                file_name="predicciones_lote.csv",
                mime="text/csv",
            )
