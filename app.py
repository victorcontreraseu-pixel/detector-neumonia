import os
import io
import glob
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib
import plotly.graph_objects as go
from PIL import Image, ImageOps
from fpdf import FPDF, XPos, YPos

# ---------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------
IMG_SIZE = (224, 224)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(APP_DIR, "samples")

st.set_page_config(page_title="Detector de neumonía", page_icon="🫁", layout="wide")

st.markdown(
    """
    <style>
        .app-header {
            padding: 1.2rem 1.5rem; border-radius: 14px;
            background: linear-gradient(90deg, #0f4c81 0%, #1a7fb3 100%);
            color: #ffffff; margin-bottom: 0.8rem;
        }
        .app-header h1 { margin: 0; font-size: 1.6rem; color: #ffffff; }
        .app-header p { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 0.95rem; color: #ffffff; }
        .result-card { padding: 1rem 1.2rem; border-radius: 12px; margin-top: 0.6rem; font-size: 1.05rem; }
        .result-pneu { background: #fdecea; border: 1px solid #f5c2c0; color: #7a1f1a !important; }
        .result-norm { background: #eaf7ee; border: 1px solid #bfe6c9; color: #1e5c31 !important; }
        .result-unk  { background: #fff6dc; border: 1px solid #f0d98a; color: #6b5200 !important; }
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
# Estado de sesión
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = {}
if "sample_path" not in st.session_state:
    st.session_state.sample_path = None


# ---------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------
def find_model_path():
    """Usa MODEL_PATH si existe; si no, toma el .keras con nombre más alto (v5 > v4 > v3)."""
    env = os.environ.get("MODEL_PATH")
    if env and os.path.exists(env):
        return env
    candidates = sorted(glob.glob(os.path.join(APP_DIR, "*.keras")))
    return candidates[-1] if candidates else None


@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(path: str):
    return tf.keras.models.load_model(path, compile=False)


def last_conv_layer_name(model):
    try:
        model.get_layer("out_relu")
        return "out_relu"
    except ValueError:
        pass
    for layer in reversed(model.layers):
        shape = getattr(getattr(layer, "output", None), "shape", None)
        if shape is not None and len(shape) == 4:
            return layer.name
    return None


# ---------------------------------------------------------------
# Imagen: lectura y preprocesamiento (igual que en el entrenamiento:
# RGB, 224x224, bilinear, /255, sin CLAHE)
# ---------------------------------------------------------------
def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        rng = arr.max() - arr.min()
        arr = (arr - arr.min()) / (rng + 1e-8) * 255.0
        img = Image.fromarray(arr.astype(np.uint8))
    return img.convert("RGB")


def to_input(img: Image.Image, flip: bool = False) -> np.ndarray:
    im = img.resize(IMG_SIZE, Image.BILINEAR)
    if flip:
        im = ImageOps.mirror(im)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def quality_notes(img: Image.Image):
    notes = []
    rgb = np.asarray(img, dtype=np.float32)
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    if saturation > 18:
        notes.append("La imagen tiene color marcado. Las radiografías suelen ser en escala de grises: "
                     "¿es una foto o una captura de pantalla? Los resultados pueden ser poco confiables.")
    w, h = img.size
    if min(w, h) < 256:
        notes.append(f"Resolución baja ({w}x{h}px). El modelo puede perder detalle.")
    if max(w, h) / max(1, min(w, h)) > 1.5:
        notes.append("Proporciones inusuales para una radiografía de tórax frontal. Revisa que no esté recortada.")
    mean = rgb.mean()
    if mean < 40:
        notes.append("La imagen está muy oscura.")
    elif mean > 215:
        notes.append("La imagen está muy clara o sobreexpuesta.")
    return notes


# ---------------------------------------------------------------
# Predicción y Grad-CAM (con caché para no recalcular en cada interacción)
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=128)
def predict_prob(data: bytes, use_tta: bool, model_path: str) -> float:
    model = load_model(model_path)
    img = open_image(data)
    probs = [float(model.predict(to_input(img), verbose=0)[0][0])]
    if use_tta:
        probs.append(float(model.predict(to_input(img, flip=True), verbose=0)[0][0]))
    return float(np.mean(probs))


@st.cache_data(show_spinner=False, max_entries=64)
def compute_heatmap(data: bytes, model_path: str):
    model = load_model(model_path)
    layer_name = last_conv_layer_name(model)
    if layer_name is None:
        return None
    img = open_image(data)
    arr = tf.convert_to_tensor(to_input(img))
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(arr, training=False)
        target = preds[:, 0]
    grads = tape.gradient(target, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0)
    heat = heat / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def overlay(img: Image.Image, heat: np.ndarray, alpha: float = 0.4) -> Image.Image:
    base = img.resize(IMG_SIZE, Image.BILINEAR)
    h = Image.fromarray(np.uint8(255 * heat)).resize(IMG_SIZE, Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(h) / 255.0)[:, :, :3] * 255
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * colored
    return Image.fromarray(np.uint8(blended))


# ---------------------------------------------------------------
# Decisión, medidor y reporte
# ---------------------------------------------------------------
def decide(prob: float, thr: float, margin: float):
    if prob >= thr + margin:
        return "NEUMONÍA", "pneu"
    if prob <= thr - margin:
        return "NORMAL", "norm"
    return "INDETERMINADO", "unk"


def make_gauge(prob: float, thr: float, margin: float) -> go.Figure:
    lo = max(0.0, thr - margin) * 100
    hi = min(1.0, thr + margin) * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Probabilidad de neumonía"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0f4c81"},
                "steps": [
                    {"range": [0, lo], "color": "#c9edd1"},
                    {"range": [lo, hi], "color": "#fff1c2"},
                    {"range": [hi, 100], "color": "#f7c9c7"},
                ],
            },
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def latin(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(img, heat_img, name, label, prob, thr, margin, model_name) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, latin("Reporte - Detector de neumonía"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    lines = [
        f"Archivo: {name}",
        f"Fecha: {datetime.now():%Y-%m-%d %H:%M}",
        f"Resultado: {label}",
        f"Probabilidad de neumonía: {prob:.1%}",
        f"Umbral: {thr:.2f} (zona indeterminada +/- {margin:.2f})",
        f"Modelo: {model_name}",
    ]
    for line in lines:
        pdf.cell(0, 7, latin(line), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)
    y = pdf.get_y()
    pdf.image(img.resize((300, 300)), x=10, y=y, w=90)
    if heat_img is not None:
        pdf.image(heat_img.resize((300, 300)), x=105, y=y, w=90)
    pdf.set_y(y + 95)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 5,
        latin("Este resultado lo genera un modelo automático con fines educativos y no reemplaza "
              "el diagnóstico de un profesional de la salud. La probabilidad no es una medida clínica."),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )
    return bytes(pdf.output())


def record(key, name, prob, label, thr):
    prev = st.session_state.history.get(key)
    st.session_state.history[key] = {
        "Archivo": name,
        "Probabilidad": f"{prob:.1%}",
        "Resultado": label,
        "Umbral": f"{thr:.2f}",
        "Hora": prev["Hora"] if prev else datetime.now().strftime("%H:%M:%S"),
    }


# ---------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------
model_path = find_model_path()

with st.sidebar:
    st.header("⚙️ Configuración")
    thr = st.slider("Umbral de decisión", 0.10, 0.90, 0.50, 0.05,
                    help="Probabilidad a partir de la cual el resultado se inclina a NEUMONÍA. "
                         "Subirlo reduce falsos positivos, pero puede dejar pasar casos reales.")
    margin = st.slider("Zona indeterminada (±)", 0.00, 0.25, 0.10, 0.05,
                       help="Si la probabilidad cae cerca del umbral, el resultado se marca como "
                            "INDETERMINADO en vez de forzar una respuesta.")
    show_cam = st.checkbox("Mostrar mapa de calor (Grad-CAM)", value=True)
    use_tta = st.checkbox("Promediar con imagen en espejo", value=False,
                          help="Predice la imagen original y su espejo y promedia. Es más estable, pero un poco más lento.")

    sample_files = []
    if os.path.isdir(SAMPLES_DIR):
        sample_files = sorted(
            f for f in glob.glob(os.path.join(SAMPLES_DIR, "*"))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )[:6]
    if sample_files:
        st.divider()
        st.header("🖼️ Imágenes de ejemplo")
        for i, path in enumerate(sample_files):
            if st.button(os.path.basename(path), key=f"sample_{i}", width="stretch"):
                st.session_state.sample_path = path
        if st.session_state.sample_path and st.button("Quitar ejemplo", key="sample_clear", width="stretch"):
            st.session_state.sample_path = None
            st.rerun()

    st.divider()
    if model_path:
        st.caption(f"Modelo: `{os.path.basename(model_path)}`")

if not model_path:
    st.error(
        "No se encontró ningún archivo `.keras` junto a `app.py`. Sube tu modelo al repositorio "
        "(por ejemplo `modelo_neumonia_v5.keras`) o define la variable de entorno MODEL_PATH."
    )
    st.stop()

model_name = os.path.basename(model_path)
load_model(model_path)  # carga una sola vez (queda en caché)


# ---------------------------------------------------------------
# Tarjeta de resultado de una imagen
# ---------------------------------------------------------------
def render_item(idx, name, data, expanded):
    key = hashlib.md5(data).hexdigest()
    img = open_image(data)
    prob = predict_prob(data, use_tta, model_path)
    label, kind = decide(prob, thr, margin)
    record(key, name, prob, label, thr)

    with st.expander(f"{name} — {label} ({prob:.1%})", expanded=expanded):
        for note in quality_notes(img):
            st.warning(note)

        c1, c2, c3 = st.columns(3)
        c1.image(img, caption="Radiografía", width="stretch")

        heat_img = None
        if show_cam:
            try:
                heat = compute_heatmap(data, model_path)
                if heat is not None:
                    heat_img = overlay(img, heat)
                    c2.image(heat_img, caption="Grad-CAM (zonas relevantes)", width="stretch")
            except Exception as e:
                c2.warning(f"No se pudo generar el mapa de calor: {e}")

        c3.plotly_chart(make_gauge(prob, thr, margin), width="stretch", key=f"gauge_{idx}_{key}")

        if kind == "pneu":
            text = f"<b>Resultado: NEUMONÍA (sospecha)</b> — Probabilidad: {prob:.1%}"
        elif kind == "norm":
            text = f"<b>Resultado: NORMAL</b> — Probabilidad de neumonía: {prob:.1%}"
        else:
            text = (f"<b>Resultado: INDETERMINADO</b> — Probabilidad: {prob:.1%}. "
                    "El modelo no está seguro; conviene revisión profesional.")
        st.markdown(f'<div class="result-card result-{kind}">{text}</div>', unsafe_allow_html=True)

        pdf_bytes = build_pdf(img, heat_img, name, label, prob, thr, margin, model_name)
        st.download_button(
            "📄 Descargar reporte PDF",
            data=pdf_bytes,
            file_name=f"reporte_{os.path.splitext(name)[0]}.pdf",
            mime="application/pdf",
            key=f"pdf_{idx}_{key}",
        )

    return {"Archivo": name, "Probabilidad": f"{prob:.1%}", "Resultado": label}


# ---------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------
tab_analizar, tab_historial, tab_info = st.tabs(["🔍 Analizar", "📋 Historial", "ℹ️ Acerca del modelo"])

with tab_analizar:
    uploaded = st.file_uploader(
        "Sube una o varias radiografías de tórax (frontales)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    items = []
    sp = st.session_state.sample_path
    if sp and os.path.exists(sp):
        with open(sp, "rb") as f:
            items.append((os.path.basename(sp), f.read()))
    for f in uploaded or []:
        items.append((f.name, f.getvalue()))

    if not items:
        st.info("Sube una imagen para comenzar. Funciona mejor con radiografías de tórax frontales "
                "(PA/AP), sin recortes y sin fotos de pantalla.")
    else:
        summary = st.container()
        rows = []
        for idx, (name, data) in enumerate(items):
            try:
                rows.append(render_item(idx, name, data, expanded=len(items) == 1))
            except Exception as e:
                st.error(f"No se pudo procesar {name}: {e}")
        if len(rows) > 1:
            with summary:
                st.subheader("Resumen del lote")
                df = pd.DataFrame(rows)
                st.dataframe(df, width="stretch", hide_index=True)
                st.download_button(
                    "⬇️ Descargar resumen (CSV)",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="resumen_lote.csv",
                    mime="text/csv",
                    key="csv_lote",
                )

with tab_historial:
    if st.session_state.history:
        hist = pd.DataFrame(list(st.session_state.history.values())[::-1])
        st.dataframe(hist, width="stretch", hide_index=True)
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Descargar historial (CSV)",
            data=hist.to_csv(index=False).encode("utf-8-sig"),
            file_name="historial.csv",
            mime="text/csv",
            key="csv_hist",
        )
        if c2.button("🗑️ Borrar historial", key="clear_hist"):
            st.session_state.history = {}
            st.rerun()
    else:
        st.info("Aún no has analizado imágenes en esta sesión.")

with tab_info:
    st.markdown(
        """
**Qué hace:** clasifica una radiografía de tórax frontal como *normal* o con *sospecha de neumonía*,
y muestra con un mapa de calor (Grad-CAM) qué zonas influyeron en la decisión.

**Cómo leer el resultado**
- La probabilidad **no está calibrada**: 80 % no significa que haya un 80 % de riesgo clínico.
- **INDETERMINADO** significa que el modelo no está seguro. Es preferible a forzar una respuesta.
- El umbral y la zona indeterminada se ajustan en la barra lateral.
- Si el mapa de calor se concentra fuera de los pulmones (bordes, texto, esquinas), desconfía del resultado.

**Limitaciones**
- Entrenado con radiografías pediátricas y de adultos de datasets públicos; puede rendir distinto con
  otros equipos, hospitales, proyecciones (AP/PA) o poblaciones.
- Puede dar falsos positivos con estructuras normales que se superponen, y falsos negativos.
- No detecta otras enfermedades. No es un dispositivo médico.

**Privacidad:** la app procesa las imágenes en memoria y no las guarda en disco. Aun así, no subas
radiografías con datos identificables de pacientes sin autorización.
        """
    )
