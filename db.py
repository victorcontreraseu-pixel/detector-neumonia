"""
Base de datos del detector de neumonía.

- Por defecto usa SQLite local (archivo neumonia.db) -> solo para pruebas.
- Si defines DATABASE_URL en st.secrets (o variable de entorno), usa esa base
  (por ejemplo PostgreSQL de Supabase) y los datos NO se pierden al reiniciar la app.
"""
import os
import re
import hmac
import hashlib
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Float, Text,
    Boolean, DateTime, select, insert, update, func,
)

TZ = ZoneInfo("America/Santiago")
ROLES = ("medico", "tecnologo", "admin")
ROL_NOMBRE = {"medico": "Médico", "tecnologo": "Tecnólogo médico", "admin": "Enfermero (administrador)"}


def ahora():
    """Hora de Chile sin zona (Streamlit Cloud corre en UTC)."""
    return datetime.now(TZ).replace(tzinfo=None)


def _secret(nombre, defecto=None):
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:
        pass
    return os.environ.get(nombre, defecto)


# ---------------------------------------------------------------
# Motor y tablas
# ---------------------------------------------------------------
metadata = MetaData()

usuarios = Table(
    "usuarios", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("usuario", String(60), unique=True, nullable=False),
    Column("nombre", String(120), nullable=False),
    Column("rol", String(20), nullable=False),
    Column("clave", String(300), nullable=False),
    Column("activo", Boolean, nullable=False, default=True),
)

procesamientos = Table(
    "procesamientos", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("fecha", DateTime, nullable=False),
    Column("rut", String(12), nullable=False, index=True),
    Column("paciente", String(150)),
    Column("edad", Integer),
    Column("archivo", String(200)),
    Column("resultado", String(20)),
    Column("probabilidad", Float),
    Column("umbral", Float),
    Column("usuario", String(60), nullable=False),
    Column("profesional", String(120)),
    Column("rol", String(20)),
    Column("informe", Text),
)

auditoria = Table(
    "auditoria", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("fecha", DateTime, nullable=False),
    Column("usuario", String(60)),
    Column("accion", String(60)),
    Column("detalle", Text),
)


@st.cache_resource
def get_engine():
    url = _secret("DATABASE_URL", "sqlite:///neumonia.db")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, pool_pre_ping=True, connect_args=args)


# ---------------------------------------------------------------
# Claves
# ---------------------------------------------------------------
def hash_clave(clave: str, sal: str | None = None) -> str:
    sal = sal or os.urandom(16).hex()
    h = hashlib.pbkdf2_hmac("sha256", clave.encode(), sal.encode(), 200_000).hex()
    return f"{sal}${h}"


def verificar_clave(clave: str, guardado: str) -> bool:
    try:
        sal, _ = guardado.split("$")
    except ValueError:
        return False
    return hmac.compare_digest(hash_clave(clave, sal), guardado)


# ---------------------------------------------------------------
# RUT
# ---------------------------------------------------------------
def normalizar_rut(rut: str) -> str:
    return re.sub(r"[^0-9kK]", "", rut or "").upper()


def rut_valido(rut: str) -> bool:
    r = normalizar_rut(rut)
    if len(r) < 2 or not r[:-1].isdigit():
        return False
    cuerpo, dv = r[:-1], r[-1]
    suma, mult = 0, 2
    for d in reversed(cuerpo):
        suma += int(d) * mult
        mult = 2 if mult == 7 else mult + 1
    esperado = 11 - (suma % 11)
    esperado = "0" if esperado == 11 else "K" if esperado == 10 else str(esperado)
    return dv == esperado


def formatear_rut(rut: str) -> str:
    r = normalizar_rut(rut)
    if len(r) < 2:
        return r
    cuerpo, dv = r[:-1], r[-1]
    return f"{int(cuerpo):,}".replace(",", ".") + f"-{dv}"


# ---------------------------------------------------------------
# Inicialización
# ---------------------------------------------------------------
@st.cache_resource
def iniciar_db():
    """Crea tablas y el primer administrador (desde secrets) si no existe ninguno."""
    eng = get_engine()
    metadata.create_all(eng)
    with eng.begin() as con:
        hay_admin = con.execute(
            select(usuarios.c.id).where(usuarios.c.rol == "admin")
        ).first()
        if not hay_admin:
            u = _secret("ADMIN_USER")
            c = _secret("ADMIN_CLAVE")
            if u and c:
                con.execute(insert(usuarios).values(
                    usuario=u, nombre=_secret("ADMIN_NOMBRE", "Administrador"),
                    rol="admin", clave=hash_clave(c), activo=True))
    return True


# ---------------------------------------------------------------
# Usuarios
# ---------------------------------------------------------------
def login(usuario: str, clave: str):
    with get_engine().connect() as con:
        f = con.execute(select(usuarios).where(usuarios.c.usuario == usuario)).mappings().first()
    if f and f["activo"] and verificar_clave(clave, f["clave"]):
        return {"usuario": f["usuario"], "nombre": f["nombre"], "rol": f["rol"]}
    return None


def crear_usuario(usuario, clave, nombre, rol):
    if rol not in ROLES:
        raise ValueError("Rol inválido")
    with get_engine().begin() as con:
        con.execute(insert(usuarios).values(
            usuario=usuario, nombre=nombre, rol=rol, clave=hash_clave(clave), activo=True))


def listar_usuarios() -> pd.DataFrame:
    return pd.read_sql(
        select(usuarios.c.usuario, usuarios.c.nombre, usuarios.c.rol, usuarios.c.activo)
        .order_by(usuarios.c.rol, usuarios.c.usuario),
        get_engine())


def set_activo(usuario: str, activo: bool):
    with get_engine().begin() as con:
        con.execute(update(usuarios).where(usuarios.c.usuario == usuario).values(activo=activo))


def reset_clave(usuario: str, nueva: str):
    with get_engine().begin() as con:
        con.execute(update(usuarios).where(usuarios.c.usuario == usuario)
                    .values(clave=hash_clave(nueva)))


# ---------------------------------------------------------------
# Procesamientos
# ---------------------------------------------------------------
def guardar_proceso(*, rut, paciente, edad, archivo, resultado, probabilidad,
                    umbral, sesion, informe):
    with get_engine().begin() as con:
        con.execute(insert(procesamientos).values(
            fecha=ahora(), rut=normalizar_rut(rut), paciente=paciente, edad=edad,
            archivo=archivo, resultado=resultado, probabilidad=probabilidad,
            umbral=umbral, usuario=sesion["usuario"], profesional=sesion["nombre"],
            rol=sesion["rol"], informe=informe))


def buscar_por_rut(rut: str) -> pd.DataFrame:
    return pd.read_sql(
        select(procesamientos).where(procesamientos.c.rut == normalizar_rut(rut))
        .order_by(procesamientos.c.fecha.desc()),
        get_engine())


def procesos_df(desde: date | None = None, hasta: date | None = None,
                usuario: str | None = None) -> pd.DataFrame:
    q = select(procesamientos)
    if desde:
        q = q.where(procesamientos.c.fecha >= datetime.combine(desde, datetime.min.time()))
    if hasta:
        q = q.where(procesamientos.c.fecha < datetime.combine(hasta + timedelta(days=1), datetime.min.time()))
    if usuario:
        q = q.where(procesamientos.c.usuario == usuario)
    return pd.read_sql(q.order_by(procesamientos.c.fecha.desc()), get_engine())


# ---------------------------------------------------------------
# Auditoría (quién hizo qué)
# ---------------------------------------------------------------
def registrar_auditoria(usuario: str, accion: str, detalle: str = ""):
    with get_engine().begin() as con:
        con.execute(insert(auditoria).values(
            fecha=ahora(), usuario=usuario, accion=accion, detalle=detalle))


def ver_auditoria(limite: int = 500) -> pd.DataFrame:
    return pd.read_sql(
        select(auditoria).order_by(auditoria.c.fecha.desc()).limit(limite),
        get_engine())
