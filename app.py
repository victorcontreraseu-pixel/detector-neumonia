import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
import matplotlib
import cv2
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF, XPos, YPos

import db
import auth

# ---------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "modelo_neumonia.keras")
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = "out_relu"  # última capa convolucional de MobileNetV2
SAMPLES_DIR = "samples"

st.set_page_config(page_title="Detector de neumonía", page_icon="🫁", layout="wide")

# ---------------------------------------------------------------
# Base de datos + LOGIN (nada de lo de abajo se ejecuta sin sesión)
# ---------------------------------------------------------------
db.iniciar_db()
sesion = auth.requerir_login()
rol = sesion["rol"]

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
# Carga del modelo (una sola vez, en caché)
# ---------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model(path: str):
    return tf.keras.models.load_model(path)


# ---------------------------------------------------------------
# Preprocesamiento: debe ser IGUAL al usado en el entrenamiento
# ---------------------------------------------------------------
def apply_clahe(image: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    return Image.fromarray(gray_eq).convert("RGB")


def preprocess(image: Image.Image) -> np.ndarray:
    image = apply_clahe(image)
    image = image.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------
def make_gradcam(model, img_array: np.ndarray) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(LAST_CONV_LAYER).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        target = preds[:, 0]
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
# Medidor visual (gauge)
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
# Reporte PDF (ahora incluye paciente y profesional)
# ---------------------------------------------------------------
def _l1(texto) -> str:
    """fpdf con fuentes base solo admite latin-1."""
    return str(texto).encode("latin-1", "replace").decode("latin-1")


def generate_pdf(image: Image.Image, datos: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Reporte - Detector de neumonia", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    lineas = [
        f"Fecha: {datos['fecha']:%Y-%m-%d %H:%M}",
        f"Paciente: {datos['paciente']}",
        f"RUT: {db.formatear_rut(datos['rut'])}",
        f"Edad: {datos['edad']}",
        f"Archivo: {datos['archivo']}",
        f"Resultado del modelo: {datos['resultado']}",
        f"Probabilidad de neumonia: {datos['probabilidad']:.1%}  (umbral {datos['umbral']:.0%})",
        f"Procesado por: {datos['profesional']} ({db.ROL_NOMBRE.get(datos['rol'], datos['rol'])})",
    ]
    for l in lineas:
        pdf.cell(0, 8, _l1(l), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    if datos["informe"]:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Informe / observaciones:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, _l1(datos["informe"]))
        pdf.ln(2)

    pdf.image(image.convert("RGB").resize((300, 300)), w=90)
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(
        0, 6,
        "Este resultado es generado por un modelo automatico con fines educativos "
        "y no reemplaza el diagnostico de un profesional de la salud.",
    )
    return bytes(pdf.output())


# ---------------------------------------------------------------
# Predicción en caché de sesión (no se repite al guardar o cambiar de pestaña)
# ---------------------------------------------------------------
def get_prediccion(image: Image.Image, uid: str, show_cam: bool) -> dict:
    cache = st.session_state.setdefault("pred", {})
    p = cache.get(uid)
    if p is None:
        with st.spinner("Analizando..."):
            prob = float(model.predict(preprocess(image), verbose=0)[0][0])
        p = {"prob": prob, "heat": None, "heat_error": None}
        cache[uid] = p
    if show_cam and p["heat"] is None and p["heat_error"] is None:
        try:
            p["heat"] = make_gradcam(model, preprocess(image))
        except Exception as e:
            p["heat_error"] = str(e)
    return p


# ---------------------------------------------------------------
# Bloque de resultados + formulario de registro de UNA imagen
# ---------------------------------------------------------------
def process_and_display(image: Image.Image, source_name: str, uid: str,
                        threshold: float, show_cam: bool, es_ejemplo: bool = False):
    p = get_prediccion(image, uid, show_cam)
    prob = p["prob"]
    is_pneumonia = prob >= threshold
    label = "NEUMONÍA" if is_pneumonia else "NORMAL"
    confidence = prob if is_pneumonia else 1 - prob

    st.markdown(f"#### {source_name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Radiografía", width="stretch")
    if show_cam:
        if p["heat"] is not None:
            col2.image(overlay_heatmap(image, p["heat"]),
                       caption="Grad-CAM (zonas relevantes)", width="stretch")
        elif p["heat_error"]:
            col2.warning(f"No se pudo generar el mapa de calor: {p['heat_error']}")
    with col3:
        st.plotly_chart(make_gauge(prob, threshold), width="stretch", key=f"gauge_{uid}")

    css_class = "result-pneumonia" if is_pneumonia else "result-normal"
    st.markdown(
        f'<div class="result-card {css_class}"><b>Resultado: {label}</b> — Confianza: {confidence:.1%}</div>',
        unsafe_allow_html=True,
    )

    # ---- Registro del procesamiento ----
    guardados = st.session_state.setdefault("guardados", {})

    if es_ejemplo:
        st.caption("Imagen de ejemplo: no se registra en la base de datos.")
    elif uid in guardados:
        d = guardados[uid]
        st.success(f"✅ Registro guardado — {db.formatear_rut(d['rut'])} · {d['paciente']}")
        st.download_button(
            "📄 Descargar reporte PDF",
            data=generate_pdf(image, d),
            file_name=f"reporte_{d['rut']}_{d['fecha']:%Y%m%d_%H%M}.pdf",
            mime="application/pdf",
            key=f"pdf_{uid}",
        )
    else:
        with st.form(f"form_{uid}"):
            st.markdown("**Registrar procesamiento**")
            c1, c2, c3 = st.columns([2, 3, 1])
            rut = c1.text_input("RUT del paciente", placeholder="12.345.678-5")
            paciente = c2.text_input("Nombre del paciente")
            edad = c3.number_input("Edad", min_value=0, max_value=120, value=0)
            informe = st.text_area("Informe / observaciones del profesional", height=100)
            guardar = st.form_submit_button("💾 Guardar registro", type="primary")

        if guardar:
            if not db.rut_valido(rut):
                st.error("RUT inválido. Revisa el número y el dígito verificador.")
            elif not paciente.strip():
                st.error("Ingresa el nombre del paciente.")
            else:
                datos = dict(
                    fecha=db.ahora(), rut=db.normalizar_rut(rut), paciente=paciente.strip(),
                    edad=int(edad), archivo=source_name, resultado=label, probabilidad=prob,
                    umbral=threshold, usuario=sesion["usuario"], profesional=sesion["nombre"],
                    rol=rol, informe=informe.strip(),
                )
                db.guardar_proceso(
                    rut=datos["rut"], paciente=datos["paciente"], edad=datos["edad"],
                    archivo=source_name, resultado=label, probabilidad=prob,
                    umbral=threshold, sesion=sesion, informe=datos["informe"],
                )
                db.registrar_auditoria(sesion["usuario"], "REGISTRO", datos["rut"])
                guardados[uid] = datos
                st.rerun()
    st.divider()


# ---------------------------------------------------------------
# Utilidad para mostrar tablas legibles
# ---------------------------------------------------------------
def tabla_legible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["rut"] = out["rut"].map(db.formatear_rut)
    out["probabilidad"] = (out["probabilidad"] * 100).round(1).astype(str) + "%"
    out["rol"] = out["rol"].map(lambda r: db.ROL_NOMBRE.get(r, r))
    cols = ["fecha", "rut", "paciente", "edad", "archivo", "resultado", "probabilidad",
            "profesional", "rol", "informe"]
    return out[cols]


# ---------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown(f"👤 **{sesion['nombre']}**")
    st.caption(db.ROL_NOMBRE.get(rol, rol))
    if st.button("Cerrar sesión", width="stretch"):
        auth.cerrar_sesion()

    st.divider()
    st.header("⚙️ Configuración")
    threshold = st.slider(
        "Umbral de decisión", min_value=0.10, max_value=0.90, value=0.50, step=0.05,
        help="Si la probabilidad de neumonía supera este valor, se clasifica como NEUMONÍA.",
    )
    show_cam = st.checkbox("Mostrar mapa de calor (Grad-CAM)", value=True)

    st.divider()
    st.header("🖼️ Imágenes de ejemplo")
    sample_files = {
        "Ejemplo normal": os.path.join(SAMPLES_DIR, "normal.jpg"),
        "Ejemplo con neumonía": os.path.join(SAMPLES_DIR, "pneumonia.jpg"),
    }
    if any(os.path.exists(p) for p in sample_files.values()):
        for nombre_btn, path in sample_files.items():
            if os.path.exists(path) and st.button(nombre_btn, width="stretch"):
                st.session_state["sample"] = path
        if st.session_state.get("sample") and st.button("Quitar ejemplo", width="stretch"):
            st.session_state.pop("sample")
            st.rerun()
    else:
        st.caption("Agrega imágenes en `samples/` (`normal.jpg`, `pneumonia.jpg`) para habilitar esta opción.")

# ---------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(
        f"No se encontró el modelo en `{MODEL_PATH}`. Copia `modelo_neumonia.keras` junto a "
        "`app.py` o define la variable de entorno MODEL_PATH."
    )
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------
# Pestañas según perfil
# ---------------------------------------------------------------
nombres_tabs = auth.PERMISOS[rol]
T = dict(zip(nombres_tabs, st.tabs(nombres_tabs)))

# ================= PROCESAR =================
with T["Procesar"]:
    uploaded_files = st.file_uploader(
        "Sube una o varias radiografías de tórax",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    sample_path = st.session_state.get("sample")
    if sample_path and os.path.exists(sample_path):
        process_and_display(Image.open(sample_path), os.path.basename(sample_path),
                            f"sample_{os.path.basename(sample_path)}", threshold, show_cam,
                            es_ejemplo=True)

    for f in uploaded_files or []:
        uid = f"{f.name}_{f.size}"
        process_and_display(Image.open(f), f.name, uid, threshold, show_cam)

    if not uploaded_files and not sample_path:
        st.info("Sube una imagen para comenzar. Cada radiografía se registra con el RUT del paciente.")

# ================= BUSCAR PACIENTE =================
with T["Buscar paciente"]:
    st.subheader("🔎 Buscar paciente por RUT")
    rut_b = st.text_input("RUT", placeholder="12.345.678-5", key="rut_busqueda")
    if rut_b:
        if not db.rut_valido(rut_b):
            st.error("RUT inválido.")
        else:
            res = db.buscar_por_rut(rut_b)
            db.registrar_auditoria(sesion["usuario"], "CONSULTA_RUT", db.normalizar_rut(rut_b))
            if res.empty:
                st.info("Este RUT no tiene procesamientos registrados.")
            else:
                st.markdown(f"**{res.iloc[0]['paciente']}** — {db.formatear_rut(rut_b)} · "
                            f"{len(res)} procesamiento(s)")
                st.dataframe(tabla_legible(res), width="stretch", hide_index=True)

# ================= INFORMES =================
with T["Informes"]:
    st.subheader("📋 Informes y registros")
    c1, c2, c3 = st.columns(3)
    hoy = db.ahora().date()
    desde = c1.date_input("Desde", hoy - timedelta(days=30), key="inf_desde")
    hasta = c2.date_input("Hasta", hoy, key="inf_hasta")

    if rol == "tecnologo":
        usuario_filtro = sesion["usuario"]
        c3.caption("Se muestran solo tus registros.")
    else:
        lista = db.listar_usuarios()
        opciones = ["(todos)"] + lista["usuario"].tolist()
        elegido = c3.selectbox("Profesional", opciones)
        usuario_filtro = None if elegido == "(todos)" else elegido

    df = db.procesos_df(desde, hasta, usuario_filtro)
    if df.empty:
        st.info("Sin registros para el filtro seleccionado.")
    else:
        vista = tabla_legible(df)
        st.dataframe(vista, width="stretch", hide_index=True)
        st.download_button("⬇️ Descargar CSV", vista.to_csv(index=False).encode("utf-8-sig"),
                           "informes.csv", "text/csv")

# ================= ESTADÍSTICA =================
if "Estadística" in T:
    with T["Estadística"]:
        st.subheader("📊 Estadística")
        c1, c2 = st.columns(2)
        hoy = db.ahora().date()
        desde = c1.date_input("Desde", hoy - timedelta(days=30), key="est_desde")
        hasta = c2.date_input("Hasta", hoy, key="est_hasta")
        df = db.procesos_df(desde, hasta)

        if df.empty:
            st.info("Aún no hay datos en el período seleccionado.")
        else:
            total = len(df)
            neum = int((df["resultado"] == "NEUMONÍA").sum())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total procesados", total)
            m2.metric("Neumonía", neum)
            m3.metric("Normal", total - neum)
            m4.metric("% neumonía", f"{neum / total:.1%}")

            g1, g2 = st.columns(2)
            por_prof = df.groupby("profesional").size().reset_index(name="procesados")
            g1.plotly_chart(px.bar(por_prof, x="profesional", y="procesados",
                                   title="Procesamientos por profesional"), width="stretch")
            por_res = df.groupby("resultado").size().reset_index(name="casos")
            g2.plotly_chart(px.pie(por_res, names="resultado", values="casos",
                                   title="Distribución de resultados"), width="stretch")

            df["dia"] = pd.to_datetime(df["fecha"]).dt.date
            por_dia = df.groupby(["dia", "resultado"]).size().reset_index(name="casos")
            st.plotly_chart(px.bar(por_dia, x="dia", y="casos", color="resultado",
                                   title="Procesamientos por día"), width="stretch")

            cruce = pd.crosstab(df["profesional"], df["resultado"]).reset_index()
            st.markdown("**Resultados por profesional**")
            st.dataframe(cruce, width="stretch", hide_index=True)

# ================= ADMINISTRACIÓN =================
if "Administración" in T:
    with T["Administración"]:
        st.subheader("👥 Usuarios")
        usuarios_df = db.listar_usuarios()
        st.dataframe(usuarios_df, width="stretch", hide_index=True)

        a1, a2 = st.columns(2)
        with a1:
            with st.form("nuevo_usuario", clear_on_submit=True):
                st.markdown("**Crear usuario**")
                nu = st.text_input("Usuario (para ingresar)")
                nn = st.text_input("Nombre completo")
                nr = st.selectbox("Perfil", db.ROLES, format_func=lambda r: db.ROL_NOMBRE[r])
                nc = st.text_input("Clave inicial (mín. 8 caracteres)", type="password")
                if st.form_submit_button("Crear"):
                    if not nu.strip() or not nn.strip() or len(nc) < 8:
                        st.error("Completa todos los campos (clave de al menos 8 caracteres).")
                    else:
                        try:
                            db.crear_usuario(nu.strip(), nc, nn.strip(), nr)
                            db.registrar_auditoria(sesion["usuario"], "CREAR_USUARIO", nu.strip())
                            st.success("Usuario creado.")
                            st.rerun()
                        except Exception:
                            st.error("Ese nombre de usuario ya existe.")
        with a2:
            with st.form("gestion_usuario"):
                st.markdown("**Activar / desactivar / cambiar clave**")
                sel = st.selectbox("Usuario", usuarios_df["usuario"].tolist())
                nueva = st.text_input("Nueva clave (opcional)", type="password")
                accion = st.radio("Acción", ["Solo cambiar clave", "Activar", "Desactivar"], horizontal=True)
                if st.form_submit_button("Aplicar"):
                    if accion == "Desactivar" and sel == sesion["usuario"]:
                        st.error("No puedes desactivar tu propia cuenta.")
                    else:
                        if nueva:
                            if len(nueva) < 8:
                                st.error("La clave debe tener al menos 8 caracteres.")
                                st.stop()
                            db.reset_clave(sel, nueva)
                        if accion == "Activar":
                            db.set_activo(sel, True)
                        elif accion == "Desactivar":
                            db.set_activo(sel, False)
                        db.registrar_auditoria(sesion["usuario"], "GESTION_USUARIO", f"{sel}: {accion}")
                        st.success("Cambios aplicados.")
                        st.rerun()

        st.divider()
        st.subheader("🧾 Auditoría (últimos 500 eventos)")
        st.dataframe(db.ver_auditoria(), width="stretch", hide_index=True)
