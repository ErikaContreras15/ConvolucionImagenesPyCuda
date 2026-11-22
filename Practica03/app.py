
import os
import queue
import re
import shutil
import sys
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, UnidentifiedImageError

import motor_gpu
import registro
from motor_gpu import FILTROS, MotorConvolucion, SinMemoriaGPU, a_visual, guardar_raw

# ---------- Configuración (variables de entorno) ----------
HILOS = int(os.environ.get("HILOS_BLOQUE", "256"))
COLA_MAX = int(os.environ.get("COLA_MAX", "4"))        # solicitudes esperando (además de la que se procesa)
MAX_MB = int(os.environ.get("MAX_MB", "120"))          # tamaño máximo del archivo subido
MAX_LADO = int(os.environ.get("MAX_LADO", "6000"))     # ancho y alto máximos en píxeles
TTL_S = int(os.environ.get("TTL_S", "900"))            # segundos que se guardan los resultados
DIR_TMP = os.environ.get("DIR_TMP", "/tmp/conv_servicio")
FORMATOS = {"PNG", "JPEG", "PPM", "BMP", "TIFF"}       # PPM incluye .pgm (gris) y .ppm (color)
VISTA_NAVEGADOR = {"PNG", "JPEG", "BMP"}               # formatos que el navegador muestra solo
ARCHIVOS = {"resultado.png", "resultado.f32", "original.png"}
RE_ID = re.compile(r"^[0-9a-f]{32}$")

Image.MAX_IMAGE_PIXELS = MAX_LADO * MAX_LADO           # protección extra de Pillow

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 2**20     # Flask responde 413 si se supera
app.json.ensure_ascii = False                          # tildes legibles en los mensajes JSON
os.makedirs(DIR_TMP, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------- Hilo GPU: único dueño del contexto CUDA ----------
class TrabajadorGPU(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True, name="hilo-gpu")
        self.cola = queue.Queue(maxsize=COLA_MAX)
        self.listo = threading.Event()
        self.error = None
        self.info = {}
        self.ocupado = False

    def run(self):
        try:
            motor = MotorConvolucion(HILOS)
            t0 = time.perf_counter()
            motor.calentar()
            self.info = {"gpu": motor.gpu, "memoria_gpu_mb": round(motor.memoria_total_mb),
                         "hilos_bloque": HILOS, "init_ms": round(motor.t_init, 1),
                         "compilacion_ms": round(motor.t_comp, 1),
                         "calentamiento_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
            log(f"GPU lista: {self.info}")
        except Exception as e:                       # sin GPU: diagnóstico claro, NO se usa la CPU
            self.error = f"{type(e).__name__}: {e}"
            log(f"ERROR al iniciar la GPU: {self.error}")
            self.listo.set()
            return
        self.listo.set()

        while True:                                  # atiende en orden de llegada (FIFO)
            trabajo = self.cola.get()
            self.ocupado = True
            trabajo["t_inicio_gpu"] = time.perf_counter()
            try:
                trabajo["resultado"] = motor.procesar(trabajo["planos"], trabajo["filtro"], trabajo["k"])
            except Exception as e:
                trabajo["error"] = e
            finally:
                trabajo["planos"] = None             # liberar la entrada en la CPU
                self.ocupado = False
                trabajo["hecho"].set()
                self.cola.task_done()


trabajador = TrabajadorGPU()
trabajador.start()


# ---------- Utilidades ----------
def error(estado, codigo, mensaje, sid=None):
    log(f"{estado} {codigo} {sid or ''} {mensaje}")
    r = jsonify(error=codigo, mensaje=mensaje, id=sid)
    r.status_code = estado
    if estado == 503:
        r.headers["Retry-After"] = "5"
    return r


_bloqueo_limpieza = threading.Lock()


def limpiar_viejos():
    """Borra las carpetas de resultados con más de TTL_S segundos."""
    if not _bloqueo_limpieza.acquire(blocking=False):
        return
    try:
        ahora = time.time()
        for nombre in os.listdir(DIR_TMP):
            ruta = os.path.join(DIR_TMP, nombre)
            try:
                if ahora - os.path.getmtime(ruta) > TTL_S:
                    shutil.rmtree(ruta, ignore_errors=True)
            except OSError:
                pass
    finally:
        _bloqueo_limpieza.release()


# ---------- Rutas ----------
@app.get("/")
def inicio():
    return render_template("index.html", filtros=FILTROS, max_mb=MAX_MB, max_lado=MAX_LADO)


@app.get("/api/salud")
def salud():
    """Estado del servicio y de la GPU (sirve para comprobar el acceso a GPU)."""
    trabajador.listo.wait(timeout=1)
    estado = "error" if trabajador.error else ("listo" if trabajador.listo.is_set() else "iniciando")
    return jsonify(estado=estado, error=trabajador.error, entorno=registro.entorno(),
                   cola_esperando=trabajador.cola.qsize(), cola_max=COLA_MAX,
                   procesando=trabajador.ocupado, filtros={f: list(k) for f, k in FILTROS.items()},
                   limites={"max_mb": MAX_MB, "max_lado": MAX_LADO, "formatos": sorted(FORMATOS)},
                   **trabajador.info), (503 if trabajador.error else 200)


@app.post("/api/procesar")
def procesar():
    """Campos (multipart/form-data): imagen (archivo), filtro (texto), k (entero).
    Respuesta 200: JSON con id, dimensiones, tiempos (ms) y rutas de descarga."""
    t0 = time.perf_counter()
    sid = uuid.uuid4().hex
    limpiar_viejos()

    if not trabajador.listo.wait(timeout=60):
        return error(503, "gpu_iniciando", "La GPU todavía se está iniciando. Intente de nuevo.", sid)
    if trabajador.error:
        return error(503, "gpu_no_disponible", "No hay GPU NVIDIA disponible para el servicio: "
                     + trabajador.error, sid)

    # Recepción del archivo (Flask lee el cuerpo de la solicitud aquí)
    archivo = request.files.get("imagen")
    t_rec = time.perf_counter()
    if archivo is None or not archivo.filename:
        return error(400, "falta_imagen", "Debe enviar un archivo en el campo 'imagen'.", sid)

    # Parámetros
    filtro = request.form.get("filtro", "").strip().lower()
    try:
        k = int(request.form.get("k", "3"))
    except ValueError:
        return error(400, "parametro_invalido", "El tamaño de máscara 'k' debe ser un número entero.", sid)
    try:
        motor_gpu.validar_parametros(filtro, k)
    except ValueError as e:
        return error(400, "parametro_invalido", str(e), sid)

    # Validación del formato REAL (contenido, no la extensión) y decodificación
    try:
        with Image.open(archivo.stream) as img:
            formato = img.format
            if formato not in FORMATOS:
                return error(415, "formato_no_permitido",
                             f"Formato {formato} no permitido. Use: {', '.join(sorted(FORMATOS))}.", sid)
            ancho, alto = img.size
            if not (1 <= ancho <= MAX_LADO and 1 <= alto <= MAX_LADO):
                return error(413, "dimensiones_excedidas",
                             f"La imagen mide {ancho}x{alto}; el máximo es {MAX_LADO}x{MAX_LADO}.", sid)
            img.load()
            planos = motor_gpu.a_planos(img)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, SyntaxError) as e:
        return error(400, "imagen_invalida", f"El archivo no es una imagen válida ({type(e).__name__}).", sid)
    t_dec = time.perf_counter()

    # Encolar (o rechazar si la cola está llena)
    trabajo = {"planos": planos, "filtro": filtro, "k": k, "hecho": threading.Event(),
               "t_encolado": time.perf_counter()}
    try:
        trabajador.cola.put_nowait(trabajo)
    except queue.Full:
        return error(503, "cola_llena", f"El servicio está ocupado ({COLA_MAX} solicitudes en espera). "
                     "Intente de nuevo en unos segundos.", sid)
    del planos
    trabajo["hecho"].wait()

    if "error" in trabajo:
        e = trabajo["error"]
        if isinstance(e, SinMemoriaGPU):
            return error(507, "sin_memoria_gpu", str(e), sid)
        return error(500, "error_procesamiento", f"Falló el procesamiento en la GPU: {type(e).__name__}: {e}", sid)
    salida, info = trabajo["resultado"]
    t_fin_gpu = time.perf_counter()

    # Preparación de la salida: PNG visual (recortado a 0-255) y .f32 crudo, en su carpeta propia
    carpeta = os.path.join(DIR_TMP, sid)
    os.makedirs(carpeta)
    a_visual(salida).save(os.path.join(carpeta, "resultado.png"), compress_level=1)
    guardar_raw(os.path.join(carpeta, "resultado.f32"), salida)
    t_sal = time.perf_counter()
    del salida

    tiempos = dict(info["tiempos"])
    tiempos["procesamiento_gpu"] = tiempos.pop("procesamiento")     # todo motor.procesar()
    tiempos.update({
        "recepcion": (t_rec - t0) * 1000.0,
        "decodificacion": (t_dec - t_rec) * 1000.0,
        "espera_cola": (trabajo["t_inicio_gpu"] - trabajo["t_encolado"]) * 1000.0,
        "salida": (t_sal - t_fin_gpu) * 1000.0,                       # codificar PNG + escribir .f32
        "servidor_total": (t_sal - t0) * 1000.0,
    })

    archivos = {"png": f"/api/resultado/{sid}/resultado.png", "f32": f"/api/resultado/{sid}/resultado.f32"}
    if formato not in VISTA_NAVEGADOR:            # p. ej. PPM: el navegador no lo muestra solo
        archivo.stream.seek(0)                    # fuera de servidor_total (solo para la interfaz)
        with Image.open(archivo.stream) as vista:
            vista.save(os.path.join(carpeta, "original.png"), compress_level=1)
        archivos["original"] = f"/api/resultado/{sid}/original.png"

    log(f"200 {sid} {filtro} k={k} {ancho}x{alto}x{info['canales']} "
        f"kernel {tiempos['kernel']:.3f} ms servidor {tiempos['servidor_total']:.1f} ms "
        f"cola {tiempos['espera_cola']:.1f} ms")
    return jsonify(id=sid, filtro=filtro, k=k, ancho=ancho, alto=alto, canales=info["canales"],
                   formato_entrada=formato, hilos_bloque=info["hilos"], bloques=info["bloques"],
                   gpu=trabajador.info.get("gpu"), entorno=registro.entorno(),
                   unidad="ms", tiempos={c: round(v, 4) for c, v in tiempos.items()},
                   archivos=archivos,
                   f32="float32 crudo sin recortar, plano por plano (canales, alto, ancho)")


@app.get("/api/resultado/<sid>/<nombre>")
def resultado(sid, nombre):
    if not RE_ID.match(sid) or nombre not in ARCHIVOS:
        return error(404, "no_encontrado", "Resultado no encontrado.")
    carpeta = os.path.join(DIR_TMP, sid)
    if not os.path.exists(os.path.join(carpeta, nombre)):
        return error(404, "no_encontrado", "Resultado no encontrado o ya expiró.", sid)
    return send_from_directory(carpeta, nombre, as_attachment=request.args.get("descargar") == "1")


@app.errorhandler(413)
def demasiado_grande(_e):
    return error(413, "archivo_muy_grande", f"El archivo supera el límite de {MAX_MB} MB.")


@app.errorhandler(404)
def no_existe(_e):
    return error(404, "ruta_no_encontrada", "Ruta no encontrada.")


@app.errorhandler(405)
def metodo(_e):
    return error(405, "metodo_no_permitido", "Método HTTP no permitido en esta ruta.")
