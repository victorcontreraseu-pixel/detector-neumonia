import os
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib
import cv2
import plotly.graph_objects as go
from fpdf import FPDF

# ---------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_neumonia.keras")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "out_relu"  # última capa convolucional de MobileNetV2
SAMPLES_DIR = "samples"

st.set_page_config(
    page_title="Detector de neumonía",
    page_icon="🫁",
    layout="wide",
)

# ---------------------------------------------------------------
# Estilos propios (branding)
# ---------------------------------------------------------------
st.markdown(
    """
    <style>
        .app-header {
            padding: 1.2rem 1.5rem;
            border-radius: 14px;
            background: linear-gradient(90deg, #0f4c81 0%, #1a7fb3 100%);
            color: white;
            margin-bottom: 1.2rem;
        }
        .app-header h1 { margin: 0; font-size: 1.6rem; }
        .app-header p { margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }
        .result-card {
            padding: 1rem 1.2rem;
            border-radius: 12px;
            margin-top: 0.5rem;
            font-size: 1.05rem;
        }
        .result-pneumonia {
            background-color: #fdecea;
            border: 1px solid #f5c2c0;
            color: #7a1f1a !important;
        }
        .result-normal {
            background-color: #eaf7ee;
            border: 1px solid #bfe6c9;
            color: #1e5c31 !important;
        }
        .result-card b { color: inherit; }
    </style>
    <div class="app-header">
        <h1>🫁 Detector de neumonía en radiografías de tórax</h1>
        <p>Herramienta de apoyo con fines educativos. No reemplaza el diagnóstico de un profesional de la salud.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------
# Estado de sesión (historial)
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # cada item: dict con nombre, resultado, prob, hora, thumb


# ---------------------------------------------------------------
# Carga del modelo (se hace una sola vez y queda en caché)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(path: str):
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------
# Preprocesamiento: debe ser IGUAL al usado en el entrenamiento
# ---------------------------------------------------------------
def apply_clahe(image: Image.Image) -> Image.Image:
    """Ecualiza el contraste localmente para que imágenes de distinta
    calidad/resolución se vean más parecidas entre sí antes de entrar al modelo."""
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return Image.fromarray(gray_eq).convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    image = apply_clahe(image)
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0  # rescale=1./255
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ---------------------------------------------------------------
# Grad-CAM: muestra qué zonas influyeron en la predicción
# ---------------------------------------------------------------
def make_gradcam(model, img_array: np.ndarray) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        target = preds[:, 0]  # probabilidad de PNEUMONIA
    grads = tape.gradient(target, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha=0.4) -> Image.Image:
    image = apply_clahe(image).resize(IMG_SIZE)
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[:, :, :3]
    colored = np.uint8(255 * colored)
    blended = (1 - alpha) * np.asarray(image) + alpha * colored
    return Image.fromarray(np.uint8(blended))


# ---------------------------------------------------------------
# Medidor visual (gauge) de probabilidad
# ---------------------------------------------------------------
def make_gauge(prob: float, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Probabilidad de neumonía"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f4c81"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#c9edd1"},
                    {"range": [threshold * 100, 100], "color": "#f7c9c7"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ---------------------------------------------------------------
# Reporte PDF descargable
# ---------------------------------------------------------------
def generate_pdf(image: Image.Image, label: str, prob: float, threshold: float, source_name: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Reporte - Detector de neumonia", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Archivo: {source_name}", ln=True)
    pdf.cell(0, 8, f"Fecha: {datetime.now():%Y-%m-%d %H:%M}", ln=True)
    pdf.cell(0, 8, f"Resultado: {label}", ln=True)
    pdf.cell(0, 8, f"Probabilidad de neumonia: {prob:.1%}", ln=True)
    pdf.cell(0, 8, f"Umbral usado: {threshold:.0%}", ln=True)
    pdf.ln(4)

    img_copy = image.convert("RGB").resize((300, 300))
    pdf.image(img_copy, w=90)

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6,
        "Este resultado es generado por un modelo automatico con fines educativos "
        "y no reemplaza el diagnostico de un profesional de la salud."
    )
    return bytes(pdf.output())


# ---------------------------------------------------------------
# Procesa una imagen y dibuja todo su bloque de resultados
# ---------------------------------------------------------------
def process_and_display(image: Image.Image, source_name: str, threshold: float, show_cam: bool):
    img_array = preprocess(image)

    with st.spinner(f"Analizando {source_name}..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])

    is_pneumonia = prob >= threshold
    label = "NEUMONÍA" if is_pneumonia else "NORMAL"
    confidence = prob if is_pneumonia else 1 - prob

    st.markdown(f"#### {source_name}")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.image(image, caption="Radiografía", use_container_width=True)

    if show_cam:
        try:
            heatmap = make_gradcam(model, img_array)
            with col2:
                st.image(
                    overlay_heatmap(image, heatmap),
                    caption="Grad-CAM (zonas relevantes)",
                    use_container_width=True,
                )
        except Exception as e:
            col2.warning(f"No se pudo generar el mapa de calor: {e}")

    with col3:
        st.plotly_chart(make_gauge(prob, threshold), use_container_width=True, key=f"gauge_{source_name}_{len(st.session_state.history)}")

    css_class = "result-pneumonia" if is_pneumonia else "result-normal"
    st.markdown(
        f'<div class="result-card {css_class}"><b>Resultado: {label}</b> — Confianza: {confidence:.1%}</div>',
        unsafe_allow_html=True,
    )

    pdf_bytes = generate_pdf(image, label, prob, threshold, source_name)
    st.download_button(
        "📄 Descargar reporte PDF",
        data=pdf_bytes,
        file_name=f"reporte_{source_name.split('.')[0]}.pdf",
        mime="application/pdf",
        key=f"pdf_{source_name}_{len(st.session_state.history)}",
    )

    st.session_state.history.append(
        {
            "Archivo": source_name,
            "Resultado": label,
            "Probabilidad": f"{prob:.1%}",
            "Hora": datetime.now().strftime("%H:%M:%S"),
        }
    )
    st.divider()


# ---------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    threshold = st.slider(
        "Umbral de decisión",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Si la probabilidad de neumonía supera este valor, se clasifica como NEUMONÍA.",
    )
    show_cam = st.checkbox("Mostrar mapa de calor (Grad-CAM)", value=True)

    st.divider()
    st.header("🖼️ Imágenes de ejemplo")
    sample_files = {
        "Ejemplo normal": os.path.join(SAMPLES_DIR, "normal.jpg"),
        "Ejemplo con neumonía": os.path.join(SAMPLES_DIR, "pneumonia.jpg"),
    }
    sample_choice = None
    any_sample_available = any(os.path.exists(p) for p in sample_files.values())
    if any_sample_available:
        for label, path in sample_files.items():
            if os.path.exists(path) and st.button(label, use_container_width=True):
                sample_choice = path
    else:
        st.caption(
            "Agrega imágenes en la carpeta `samples/` "
            "(`normal.jpg`, `pneumonia.jpg`) para habilitar esta opción."
        )

    if st.session_state.history:
        st.divider()
        if st.button("🗑️ Borrar historial", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ---------------------------------------------------------------
# Verificación del modelo
# ---------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"No se encontró el modelo en `{MODEL_PATH}`. Copia el archivo "
        "`modelo_neumonia.keras` junto a `app.py` o define la variable de entorno MODEL_PATH."
    )
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------
# Carga de imágenes (varias a la vez) + ejemplo elegido
# ---------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Sube una o varias radiografías de tórax",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if sample_choice:
    sample_image = Image.open(sample_choice)
    process_and_display(sample_image, os.path.basename(sample_choice), threshold, show_cam)

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        process_and_display(image, file.name, threshold, show_cam)

if not uploaded_files and not sample_choice:
    st.info("Sube una imagen o prueba con un ejemplo desde la barra lateral para comenzar.")

# ---------------------------------------------------------------
# Historial de la sesión
# ---------------------------------------------------------------
if st.session_state.history:
    st.subheader("📋 Historial de esta sesión")
    df = pd.DataFrame(st.session_state.history[::-1])  # más reciente primero
    st.dataframe(df, use_container_width=True, hide_index=True)
