"""motor_gpu.py — convolución A COLOR con PyCUDA
"""
import os
import sys
import time

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
P02 = os.path.join(os.path.dirname(AQUI), "Practica02")
sys.path.insert(0, P02)
sys.path.insert(0, os.path.join(P02, "PyCuda"))
import comun               # noqa: E402  (Practica02/comun.py)
import conv_pycuda as p02  # noqa: E402  (Practica02/PyCuda/conv_pycuda.py)

# Filtros y tamaños de máscara permitidos.
#   Obligatorios (guía, experimentos y validación): media, nitidez, laplaciano.
#   Adicionales (solo demostración web): gaussiano, relieve.
FILTROS = {"media": (3, 9, 21), "nitidez": (3,), "laplaciano": (3,),
           "gaussiano": (5,), "relieve": (3,)}
OBLIGATORIOS = ("media", "nitidez", "laplaciano")

_G = np.array([1, 4, 6, 4, 1], dtype=np.float32)
MASCARAS_EXTRA = {
    # Gaussiano 5x5 (aproximación binomial), suma 1: suaviza sin cambiar el brillo
    "gaussiano": (np.outer(_G, _G) / 256.0).astype(np.float32).ravel(),
    # Relieve (emboss) 3x3, suma 1: conserva el brillo medio
    "relieve": np.array([-2, -1, 0,
                         -1, 1, 1,
                         0, 1, 2], dtype=np.float32),
}


class GPUNoDisponible(RuntimeError):
    """No hay GPU NVIDIA visible (driver, --gpus all, NVIDIA Container Toolkit)."""


class SinMemoriaGPU(RuntimeError):
    """La imagen no cabe en la memoria de la GPU."""


def validar_parametros(filtro, k):
    if filtro not in FILTROS:
        raise ValueError(f"Filtro desconocido: '{filtro}'. Permitidos: {', '.join(FILTROS)}")
    if k not in FILTROS[filtro]:
        raise ValueError(f"Tamaño de máscara {k} no permitido para '{filtro}'. "
                         f"Permitidos: {FILTROS[filtro]}")


def crear_mascara(filtro, k):
    validar_parametros(filtro, k)
    if filtro in OBLIGATORIOS:
        return comun.crear_mascara(filtro, k)        # mismos coeficientes que P01 y P02
    return MASCARAS_EXTRA[filtro], k


# ---------- Imágenes a color: planos float32 (canales, alto, ancho) ----------
def a_planos(img):
    """-> float32 contiguo con forma (canales, alto, ancho).
    Acepta: imagen Pillow (RGB o gris), arreglo 2D (gris) o arreglo 3D YA en planos."""
    if isinstance(img, Image.Image):
        gris = img.mode in ("L", "1", "I", "F", "I;16")
        arr = np.asarray(img.convert("L") if gris else img.convert("RGB"), dtype=np.float32)
        if arr.ndim == 3:                   # Pillow da (alto, ancho, canales) -> planos
            arr = arr.transpose(2, 0, 1)
    else:
        arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:                       # gris: 1 plano
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("La imagen debe ser 2D (gris) o 3D en planos (canales, alto, ancho)")
    return np.ascontiguousarray(arr)


def leer_imagen(ruta):
    with Image.open(ruta) as img:
        return a_planos(img)


def a_visual(planos):
    """Redondeo + recorte a [0, 255] (igual que guardar_pgm de la P02) -> imagen Pillow."""
    vis = np.clip(np.rint(planos), 0, 255).astype(np.uint8)
    if vis.shape[0] == 1:
        return Image.fromarray(vis[0], mode="L")
    return Image.fromarray(vis.transpose(1, 2, 0), mode="RGB")


def guardar_raw(ruta, planos):
    """float32 crudo, plano por plano (R completo, luego G, luego B), SIN recortar."""
    np.ascontiguousarray(planos, dtype=np.float32).tofile(ruta)


class MotorConvolucion:
    def __init__(self, hilos=256):
        # 1) Contexto CUDA (medido aparte). Si no hay GPU, error claro: NO se usa la CPU.
        try:
            self.drv, self.ctx, self.t_init = p02.iniciar_cuda()
        except Exception as e:
            raise GPUNoDisponible(
                f"No se pudo iniciar CUDA ({e}). Revise el driver NVIDIA y, en Docker, "
                "que el contenedor se ejecute con --gpus all.") from e
        try:
            dev = self.ctx.get_device()
            max_hilos = dev.get_attribute(self.drv.device_attribute.MAX_THREADS_PER_BLOCK)
            if not 1 <= hilos <= max_hilos:
                raise ValueError(f"hilos por bloque fuera de rango (1..{max_hilos})")
            self.hilos = hilos
            self.gpu = dev.name()
            _, total = self.drv.mem_get_info()
            self.memoria_total_mb = total / 2**20
            # 2) Compilación del kernel de la P02 con nvcc (medida aparte, una sola vez)
            self.func, self.t_comp = p02.compilar_kernel()
        except Exception:
            self.cerrar()
            raise

    def procesar(self, img, filtro, k):
        """img: planos (canales, alto, ancho), arreglo 2D gris o imagen Pillow.
        Devuelve (salida float32 (canales, alto, ancho), info con tiempos en ms)."""
        drv = self.drv
        t0 = time.perf_counter()

        # 1. Preparación en la CPU
        mascara, k = crear_mascara(filtro, k)
        planos = a_planos(img)
        canales, alto, ancho = planos.shape
        N = ancho * alto
        bloques = (N + self.hilos - 1) // self.hilos
        h_in = planos.ravel()
        h_out = np.empty_like(h_in)
        t1 = time.perf_counter()

        # 2. Reserva en la GPU
        reservas = []
        try:
            for nbytes in (h_in.nbytes, h_out.nbytes, mascara.nbytes):
                reservas.append(drv.mem_alloc(nbytes))
        except drv.MemoryError as e:
            for d in reservas:
                d.free()
            raise SinMemoriaGPU(f"La imagen {ancho}x{alto}x{canales} no cabe en la memoria de la GPU") from e
        d_in, d_out, d_m = reservas
        t2 = time.perf_counter()

        # 3-5. Copias y kernel, medidos con eventos CUDA (mismo esquema que ejecutar() de la P02)
        try:
            e0, e1, e2, e3 = drv.Event(), drv.Event(), drv.Event(), drv.Event()
            e0.record()
            drv.memcpy_htod(d_in, h_in)                  # todos los planos CPU -> GPU
            drv.memcpy_htod(d_m, mascara)                # máscara CPU -> GPU
            e1.record()
            bytes_plano = N * 4                          # float32
            for c in range(canales):                     # mismo kernel, un lanzamiento por plano
                off = c * bytes_plano
                self.func(np.uintp(int(d_in) + off), np.uintp(int(d_out) + off),
                          np.int32(ancho), np.int32(alto), d_m, np.int32(k),
                          block=(self.hilos, 1, 1), grid=(bloques, 1))
            e2.record()
            drv.memcpy_dtoh(h_out, d_out)                # todos los planos GPU -> CPU
            e3.record()
            e3.synchronize()                             # asegurar que todo terminó
            t = {"h2d": e0.time_till(e1), "kernel": e1.time_till(e2),
                 "d2h": e2.time_till(e3), "total_gpu": e0.time_till(e3)}
        finally:
            t3 = time.perf_counter()
            for d in reservas:                           # 6. Liberación (siempre)
                d.free()
        t4 = time.perf_counter()

        t.update({
            "preparacion": (t1 - t0) * 1000.0,
            "reserva": (t2 - t1) * 1000.0,
            "liberacion": (t4 - t3) * 1000.0,
            "procesamiento": (t4 - t0) * 1000.0,         # todo procesar() (reloj monotónico)
        })
        return h_out.reshape(canales, alto, ancho), {
            "tiempos": t, "ancho": ancho, "alto": alto, "canales": canales,
            "k": k, "hilos": self.hilos, "bloques": bloques}

    def calentar(self):
        """Una ejecución pequeña para que el primer resultado medido no incluya el arranque en frío."""
        self.procesar(np.zeros((3, 64, 64), dtype=np.float32), "media", 3)

    def cerrar(self):
        if getattr(self, "ctx", None) is not None:
            self.ctx.pop()
            self.ctx.detach()
            self.ctx = None