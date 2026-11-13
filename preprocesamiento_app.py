import cv2
import numpy as np

IMG_SIZE = 384
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocesar(gris_uint8):
    # Recibe una imagen 2D uint8 (escala de grises). Devuelve un lienzo cuadrado IMG_SIZE x IMG_SIZE uint8.
    h, w = gris_uint8.shape
    escala = IMG_SIZE / max(h, w)
    nh, nw = max(1, int(round(h * escala))), max(1, int(round(w * escala)))
    r = cv2.resize(gris_uint8, (nw, nh), interpolation=cv2.INTER_AREA)
    r = _clahe.apply(r)
    lienzo = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    y0, x0 = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = r
    return lienzo

def a_tres_canales(lienzo):
    # Devuelve (IMG_SIZE, IMG_SIZE, 3) float32 con valores 0-255. NO dividir por 255: el modelo ya normaliza por dentro.
    return np.repeat(lienzo[:, :, None], 3, axis=-1).astype("float32")

def calibrar(p, a, b):
    # Convierte la salida cruda del modelo en probabilidad calibrada (a y b vienen de config_app.json)
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))
