import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib

# ---------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_neumonia.keras")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "out_relu"  # última capa convolucional de MobileNetV2

st.set_page_config(
    page_title="Detector de neumonía",
    page_icon="🫁",
    layout="centered",
)


# ---------------------------------------------------------------
# Carga del modelo (se hace una sola vez y queda en caché)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(path: str):
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------
# Preprocesamiento: debe ser IGUAL al usado en el entrenamiento
# ---------------------------------------------------------------
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
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
    image = image.convert("RGB").resize(IMG_SIZE)
    heat = Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE, Image.BILINEAR)
    colored = matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[:, :, :3]
    colored = np.uint8(255 * colored)
    blended = (1 - alpha) * np.asarray(image) + alpha * colored
    return Image.fromarray(np.uint8(blended))


# ---------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------
st.title("🫁 Detector de neumonía en radiografías de tórax")
st.caption(
    "Herramienta de apoyo con fines educativos. **No reemplaza el diagnóstico "
    "de un profesional de la salud.**"
)

with st.sidebar:
    st.header("Configuración")
    threshold = st.slider(
        "Umbral de decisión",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Si la probabilidad de neumonía supera este valor, se clasifica como NEUMONÍA. "
        "Un umbral menor detecta más casos pero genera más falsos positivos.",
    )
    show_cam = st.checkbox("Mostrar mapa de calor (Grad-CAM)", value=True)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"No se encontró el modelo en `{MODEL_PATH}`. Copia el archivo "
        "`modelo_neumonia.keras` junto a `app.py` o define la variable de entorno MODEL_PATH."
    )
    st.stop()

model = load_model(MODEL_PATH)

uploaded = st.file_uploader(
    "Sube una radiografía de tórax", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded)
    img_array = preprocess(image)

    with st.spinner("Analizando imagen..."):
        prob = float(model.predict(img_array, verbose=0)[0][0])

    is_pneumonia = prob >= threshold
    confidence = prob if is_pneumonia else 1 - prob

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Radiografía subida", use_container_width=True)

    if show_cam:
        try:
            heatmap = make_gradcam(model, img_array)
            with col2:
                st.image(
                    overlay_heatmap(image, heatmap),
                    caption="Zonas que más influyeron (Grad-CAM)",
                    use_container_width=True,
                )
        except Exception as e:
            col2.warning(f"No se pudo generar el mapa de calor: {e}")

    st.divider()
    if is_pneumonia:
        st.error(f"### Resultado: NEUMONÍA  \nConfianza: {confidence:.1%}")
    else:
        st.success(f"### Resultado: NORMAL  \nConfianza: {confidence:.1%}")

    st.progress(prob, text=f"Probabilidad de neumonía: {prob:.1%}")
    st.caption(
        "Ante cualquier síntoma o duda, consulta a un médico. "
        "Este resultado es una estimación automática y puede contener errores."
    )
