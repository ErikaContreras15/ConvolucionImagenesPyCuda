
import csv
import os
import statistics

import numpy as np
from PIL import Image


# ---------- Imágenes ----------
def leer_pgm(ruta):
    """Lee una imagen en escala de grises y la devuelve como arreglo float32 2D.

    Pillow solo se usa para leer el archivo (la guía lo permite).
    El arreglo queda contiguo en memoria (C-contiguous), fila por fila,
    igual que std::vector<float> en la práctica 01.
    """
    img = Image.open(ruta).convert("L")
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32))


def guardar_raw(ruta, salida):
    """Valores float32 crudos (sin recortar), para la validación numérica.
    Mismo formato que los .f32 de la práctica 01: solo los valores, fila por fila."""
    np.ascontiguousarray(salida, dtype=np.float32).tofile(ruta)


def guardar_pgm(ruta, salida):
    """Conversión al rango de visualización: redondeo + recorte a [0, 255],
    igual que guardarPGM en common.h (los negativos quedan en 0)."""
    vis = np.clip(np.rint(salida), 0, 255).astype(np.uint8)
    Image.fromarray(vis, mode="L").save(ruta)


# ---------- Máscaras ----------
def crear_mascara(filtro, k):
    """Devuelve (máscara float32 de k*k elementos, k).
    media: k libre e impar; nitidez y laplaciano: k = 3 forzado."""
    if filtro == "media":
        if k < 1 or k % 2 == 0:
            raise ValueError("k debe ser impar")
        return np.full(k * k, 1.0 / (k * k), dtype=np.float32), k
    if filtro == "nitidez":
        return np.array([0, -1, 0, -1, 5, -1, 0, -1, 0], dtype=np.float32), 3
    if filtro == "laplaciano":
        return np.array([0, 1, 0, 1, -4, 1, 0, 1, 0], dtype=np.float32), 3
    raise ValueError("Filtro desconocido: " + filtro)


# Matriz 5x5 (laplaciano)
MATRIZ_PRUEBA = np.array([[35, 40, 41, 45, 50],
                          [40, 40, 42, 46, 52],
                          [42, 46, 50, 55, 55],
                          [48, 52, 56, 58, 60],
                          [56, 60, 65, 70, 75]], dtype=np.float32)
ESPERADOS = [((0, 0), -60.0), ((1, 1), 8.0), ((2, 2), -1.0)]


def verificar_prueba(salida, nombre):
    print(f"Salida laplaciano 5x5 ({nombre}):")
    for fila in salida:
        print("".join(f"{v:6.0f}" for v in fila))
    ok = True
    for (f, c), esperado in ESPERADOS:
        v = float(salida[f, c])
        bien = abs(v - esperado) < 1e-4
        print(f"({f},{c}): obtenido {v:.1f}, esperado {esperado:.1f} -> {'OK' if bien else 'ERROR'}")
        ok &= bien
    return ok


# ---------- CSV ----------
COLUMNAS = ["version", "filtro", "ancho", "alto", "k", "hilos_bloque", "bloques",
            "repeticion", "fase", "tiempo", "unidad", "estado"]


def escribir_csv(ruta, version, filtro, ancho, alto, k, hilos, bloques,
                 rep, fase, tiempo_ms, estado="completada"):
    """Una fila por repetición y fase. tiempo en milisegundos.
    estado: 'completada' o 'no_completada' (guía: viabilidad de Python secuencial)."""
    nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
    with open(ruta, "a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLUMNAS)
        w.writerow([version, filtro, ancho, alto, k, hilos, bloques,
                    rep, fase, f"{tiempo_ms:.4f}", "ms", estado])


def mediana(valores):
    return statistics.median(valores)
