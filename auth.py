import streamlit as st
import db

# Qué pestañas ve cada perfil
PERMISOS = {
    "medico":    ["Procesar", "Buscar paciente", "Informes", "Estadística"],
    "tecnologo": ["Procesar", "Buscar paciente", "Informes"],
    "admin":     ["Procesar", "Buscar paciente", "Informes", "Estadística", "Administración"],
}

MAX_INTENTOS = 5


def pantalla_login():
    st.markdown("## 🫁 Detector de neumonía")
    st.caption("Acceso restringido a personal autorizado.")

    if st.session_state.get("intentos", 0) >= MAX_INTENTOS:
        st.error("Demasiados intentos fallidos. Recarga la página e inténtalo de nuevo.")
        st.stop()

    with st.form("login"):
        usuario = st.text_input("Usuario")
        clave = st.text_input("Clave", type="password")
        entrar = st.form_submit_button("Ingresar", type="primary")

    if entrar:
        sesion = db.login(usuario.strip(), clave)
        if sesion:
            st.session_state["sesion"] = sesion
            st.session_state["intentos"] = 0
            db.registrar_auditoria(sesion["usuario"], "LOGIN")
            st.rerun()
        else:
            st.session_state["intentos"] = st.session_state.get("intentos", 0) + 1
            db.registrar_auditoria(usuario.strip() or "?", "LOGIN_FALLIDO")
            st.error("Usuario o clave incorrectos.")


def requerir_login() -> dict:
    """Si no hay sesión muestra el login y detiene la app."""
    if "sesion" not in st.session_state:
        pantalla_login()
        st.stop()
    return st.session_state["sesion"]


def cerrar_sesion():
    db.registrar_auditoria(st.session_state["sesion"]["usuario"], "LOGOUT")
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()
