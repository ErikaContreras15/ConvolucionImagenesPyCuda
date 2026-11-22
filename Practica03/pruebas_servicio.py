"""pruebas_servicio.py — validación funcional y numérica del servicio

Se ejecuta en el ANFITRIÓN (como cliente), con el servicio corriendo (Docker o gunicorn).

  1. Matriz pequeña y límites : imagen 1003x997 (999 991 píxeles, no múltiplo de 256)
                                -> servicio vs PyCUDA directo.
  2. Equivalencia directo–servicio : media 3/9/21, nitidez, laplaciano en 4000x4000
                                -> error absoluto máximo y medio ANTES del recorte (.f32).
  3. Resultado visual        : PNG de los 3 filtros + el MISMO recorte de región.
  4. Entradas inválidas      : archivo no válido, archivo > límite, dimensiones > límite,
                                formato no permitido, falta imagen, filtro desconocido,
                                máscara no permitida, k no numérico -> respuesta + recuperación.
  5. Aislamiento y cola      : 2 solicitudes simultáneas identificables reciben SU resultado;
                                ráfaga de solicitudes simultáneas -> atendidas / rechazadas (503).
"""
import argparse
import io
import os
import subprocess
import sys
import threading
import time

import numpy as np
import requests
from PIL import Image

TOL = 1e-2
AQUI = os.path.dirname(os.path.abspath(__file__))
LOG = None


def log(msg=""):
    print(msg, flush=True)
    LOG.write(msg + "\n")
    LOG.flush()


def enviar(url, datos, nombre, filtro, k, tipo="application/octet-stream", timeout=600):
    archivos = {"imagen": (nombre, datos, tipo)} if datos is not None else None
    return requests.post(f"{url}/api/procesar", files=archivos,
                         data={"filtro": filtro, "k": str(k)}, timeout=timeout)


def bajar_f32(url, j):
    r = requests.get(url + j["archivos"]["f32"], timeout=120)
    r.raise_for_status()
    return np.frombuffer(r.content, dtype=np.float32).reshape(j["canales"], j["alto"], j["ancho"])


def comparar(ref, otra):
    err = np.abs(ref.astype(np.float64) - otra.astype(np.float64))
    fuera = int((err > TOL).sum())
    return err.max(), err.mean(), fuera


def directo(img, filtro, k, prefijo):
    """PyCUDA directo validado (directo.py) -> .f32 de referencia."""
    if not os.path.exists(prefijo + ".f32"):
        subprocess.run([sys.executable, os.path.join(AQUI, "directo.py"), img, filtro, str(k), "256", "1",
                        prefijo, os.path.join("resultados", "pruebas_directo.csv")],
                       check=True, capture_output=True)
    return np.fromfile(prefijo + ".f32", dtype=np.float32)


def leer_bytes(ruta):
    with open(ruta, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
def prueba_limites(url):
    log("== 1. Matriz pequeña y límites ==")
    r = subprocess.run([sys.executable, os.path.join(AQUI, "directo.py"), "--test"], capture_output=True, text=True)
    oks = r.stdout.count("-> OK")
    log(f"Prueba manual 5x5 (directo.py --test, 4 hilos/bloque, 3 hilos sobrantes): {oks}/12 OK")

    os.makedirs("salidas", exist_ok=True)
    ruta = "salidas/prueba_1003x997.ppm"
    with Image.open("imagenes_color/img_2000.ppm") as im:
        im.crop((0, 0, 1003, 997)).save(ruta)
    ref = directo(ruta, "media", 9, "salidas/directo_media9_1003x997").reshape(3, 997, 1003)
    resp = enviar(url, leer_bytes(ruta), "prueba_1003x997.ppm", "media", 9)
    j = resp.json()
    srv = bajar_f32(url, j)
    emax, emed, fuera = comparar(ref, srv)
    N = 1003 * 997
    sobrantes = j["bloques"] * j["hilos_bloque"] - N
    log(f"1003x997 media 9x9 | {N} píxeles/canal | {j['hilos_bloque']} hilos x {j['bloques']} bloques "
        f"= {sobrantes} hilos sobrantes por canal | servicio vs directo: error máx {emax:.3e}, "
        f"medio {emed:.3e}, > tol {fuera} -> {'OK' if fuera == 0 else 'ERROR'}")
    # Bordes: la extensión con ceros oscurece el borde en la media (filas/columnas extremas)
    log(f"   borde superior izq. (canal R) servicio={srv[0, 0, 0]:.3f} directo={ref[0, 0, 0]:.3f} | "
        f"esquina inf. der. servicio={srv[0, -1, -1]:.3f} directo={ref[0, -1, -1]:.3f}")
    return fuera == 0


def prueba_equivalencia(url):
    log("\n== 2. Equivalencia directo–servicio (4000x4000 color, antes del recorte) ==")
    img = "imagenes_color/img_4000.ppm"
    datos = leer_bytes(img)
    ok_total = True
    for filtro, k in (("media", 3), ("media", 9), ("media", 21), ("nitidez", 3), ("laplaciano", 3)):
        ref = directo(img, filtro, k, f"salidas/color_{filtro}{k}_4000")
        j = enviar(url, datos, "img_4000.ppm", filtro, k).json()
        srv = bajar_f32(url, j)
        ref = ref.reshape(srv.shape)
        emax, emed, fuera = comparar(ref, srv)
        ok = fuera == 0
        ok_total &= ok
        log(f"{filtro:10s} k={k:<2d} | error máx {emax:.3e} | error medio {emed:.3e} | > tol {fuera} | "
            f"{'OK' if ok else 'ERROR'} | kernel {j['tiempos']['kernel']:.3f} ms | id {j['id']}")
    return ok_total


def prueba_visual(url):
    log("\n== 3. Resultado visual (2000x2000, mismo recorte de región) ==")
    os.makedirs("evidencia", exist_ok=True)
    img = "imagenes_color/img_2000.ppm"
    region = (700, 500, 1100, 900)                          # mismo recorte 400x400 en todas
    with Image.open(img) as im:
        im.convert("RGB").save("evidencia/original_2000.png")
        im.convert("RGB").crop(region).save("evidencia/recorte_original.png")
    datos = leer_bytes(img)
    for filtro, k in (("media", 9), ("nitidez", 3), ("laplaciano", 3), ("gaussiano", 5), ("relieve", 3)):
        j = enviar(url, datos, "img_2000.ppm", filtro, k).json()
        png = requests.get(url + j["archivos"]["png"], timeout=120).content
        with Image.open(io.BytesIO(png)) as res:
            res.save(f"evidencia/{filtro}{k}_2000.png")
            res.crop(region).save(f"evidencia/recorte_{filtro}{k}.png")
        log(f"{filtro:10s} k={k:<2d} -> evidencia/{filtro}{k}_2000.png y evidencia/recorte_{filtro}{k}.png")
    log(f"Región del recorte (x0, y0, x1, y1): {region}")


def prueba_invalidas(url, max_mb):
    log("\n== 4. Entradas inválidas y recuperación ==")
    png_ok = io.BytesIO()
    Image.new("RGB", (64, 48), (120, 60, 200)).save(png_ok, format="PNG")
    png_ok = png_ok.getvalue()
    grande = io.BytesIO()
    Image.new("L", (6001, 8)).save(grande, format="PNG")
    gif = io.BytesIO()
    Image.new("RGB", (32, 32)).save(gif, format="GIF")

    casos = [
        ("Archivo no válido (texto con extensión .png)", b"esto no es una imagen", "falsa.png", "media", 3),
        (f"Archivo que supera el límite ({max_mb + 5} MB)", b"\0" * ((max_mb + 5) * 2**20), "enorme.ppm", "media", 3),
        ("Dimensiones que superan el límite (6001x8)", grande.getvalue(), "ancha.png", "media", 3),
        ("Formato no permitido (GIF)", gif.getvalue(), "animacion.gif", "media", 3),
        ("Falta la imagen", None, None, "media", 3),
        ("Filtro desconocido ('sobel')", png_ok, "ok.png", "sobel", 3),
        ("Tamaño de máscara no permitido (media 5x5)", png_ok, "ok.png", "media", 5),
        ("Máscara no permitida para el filtro (nitidez 9x9)", png_ok, "ok.png", "nitidez", 9),
        ("k no numérico ('abc')", png_ok, "ok.png", "media", "abc"),
    ]
    ok_total = True
    for desc, datos, nombre, filtro, k in casos:
        r = enviar(url, datos, nombre, filtro, k)
        try:
            j = r.json()
        except ValueError:
            j = {"error": "(sin JSON)", "mensaje": r.text[:80]}
        rechazada = 400 <= r.status_code < 500
        ok_total &= rechazada
        log(f"{desc:52s} -> HTTP {r.status_code} {j.get('error')}: {j.get('mensaje')}")

    s = requests.get(f"{url}/api/salud", timeout=30)
    r = enviar(url, png_ok, "ok.png", "media", 3)
    recupera = s.status_code == 200 and r.status_code == 200
    log(f"Recuperación: /api/salud -> HTTP {s.status_code} ({s.json().get('estado')}); "
        f"solicitud válida posterior -> HTTP {r.status_code} -> {'OK' if recupera else 'ERROR'}")
    return ok_total and recupera


def prueba_aislamiento(url):
    log("\n== 5a. Aislamiento: 2 solicitudes simultáneas identificables ==")
    imgs = {}
    for valor in (50, 200):                                 # imágenes de un solo color, fáciles de identificar
        b = io.BytesIO()
        Image.new("RGB", (2000, 2000), (valor, valor, valor)).save(b, format="PNG")
        imgs[valor] = b.getvalue()
    resultados = {}

    def hilo(valor):
        resultados[valor] = enviar(url, imgs[valor], f"constante_{valor}.png", "media", 3).json()

    hs = [threading.Thread(target=hilo, args=(v,)) for v in imgs]
    for h in hs:
        h.start()
    for h in hs:
        h.join()
    ok_total = True
    for valor, j in resultados.items():
        a = bajar_f32(url, j)
        centro = float(a[0, 1000, 1000])
        ok = abs(centro - valor) < 1e-3
        ok_total &= ok
        log(f"Solicitud 'constante_{valor}' id {j['id']} | píxel central {centro:.4f} (esperado {valor}) | "
            f"espera en cola {j['tiempos']['espera_cola']:.1f} ms | {'OK' if ok else 'ERROR'}")
    ids = [j["id"] for j in resultados.values()]
    log(f"Identificadores distintos: {'OK' if len(set(ids)) == len(ids) else 'ERROR'}")
    return ok_total


def prueba_cola(url, n):
    log(f"\n== 5b. Cola: {n} solicitudes simultáneas (4000x4000, media 21x21) ==")
    s = requests.get(f"{url}/api/salud", timeout=30).json()
    log(f"Configuración: cola_max = {s['cola_max']} en espera + 1 en proceso; atención FIFO de una en una")
    datos = leer_bytes("imagenes_color/img_4000.ppm")
    res = [None] * n

    def hilo(i):
        t0 = time.perf_counter()
        r = enviar(url, datos, f"rafaga_{i}.ppm", "media", 21)
        res[i] = (r.status_code, r.json(), (time.perf_counter() - t0) * 1000.0)

    hs = [threading.Thread(target=hilo, args=(i,)) for i in range(n)]
    for h in hs:
        h.start()
        time.sleep(0.02)
    for h in hs:
        h.join()
    atendidas = [r for r in res if r[0] == 200]
    rechazadas = [r for r in res if r[0] == 503]
    for i, (codigo, j, t) in enumerate(res):
        extra = (f"espera en cola {j['tiempos']['espera_cola']:.1f} ms" if codigo == 200
                 else f"{j.get('error')}: {j.get('mensaje')}")
        log(f"  solicitud {i}: HTTP {codigo} | cliente {t:.0f} ms | {extra}")
    log(f"Atendidas {len(atendidas)} | rechazadas por cola llena {len(rechazadas)} | "
        f"otras {n - len(atendidas) - len(rechazadas)}")
    s = requests.get(f"{url}/api/salud", timeout=30)
    log(f"Estado posterior: HTTP {s.status_code}, cola en espera {s.json()['cola_esperando']} -> "
        f"{'OK' if s.status_code == 200 else 'ERROR'}")
    return len(atendidas) > 0 and len(atendidas) + len(rechazadas) == n


def main():
    global LOG
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:5000")
    p.add_argument("--rafaga", type=int, default=10, help="solicitudes simultáneas en la prueba de cola")
    a = p.parse_args()
    os.makedirs("resultados", exist_ok=True)
    LOG = open("resultados/pruebas_servicio.txt", "w")
    s = requests.get(f"{a.url}/api/salud", timeout=30)
    info = s.json()
    log(f"Servicio {a.url} | {time.strftime('%Y-%m-%d %H:%M:%S')} | estado {info.get('estado')} | "
        f"{info.get('gpu')} | entorno {info.get('entorno')} | límites {info.get('limites')}")
    if s.status_code != 200:
        log("El servicio no está listo; se detienen las pruebas.")
        return 1
    resumen = {
        "1. Matriz pequeña y límites": prueba_limites(a.url),
        "2. Equivalencia directo–servicio": prueba_equivalencia(a.url),
    }
    prueba_visual(a.url)
    resumen["3. Resultado visual"] = True
    resumen["4. Entradas inválidas"] = prueba_invalidas(a.url, info["limites"]["max_mb"])
    resumen["5a. Aislamiento"] = prueba_aislamiento(a.url)
    resumen["5b. Cola"] = prueba_cola(a.url, a.rafaga)
    log("\n== Resumen ==")
    for nombre, ok in resumen.items():
        log(f"{nombre:36s} {'OK' if ok else 'REVISAR'}")
    LOG.close()
    return 0 if all(resumen.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
