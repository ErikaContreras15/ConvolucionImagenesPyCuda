"""
Modos:
  serie        una configuración: 1 calentamiento (repetición 0, registrado aparte) + N repeticiones
  concurrencia C clientes simultáneos, 1 solicitud por cliente, R rondas
               -> latencia de cada solicitud y rendimiento = exitosas / intervalo total de la ronda

Desde OTRO equipo (cliente remoto), anteponer MODO=servicio_red para marcar los registros:
  MODO=servicio_red python3 cliente.py serie http://192.168.1.12:5000 ...

"""
import os
import socket
import sys
import threading
import time

import requests

import registro

FASES_SERVIDOR = ("recepcion", "decodificacion", "espera_cola", "preparacion", "reserva", "h2d", "kernel",
                  "d2h", "total_gpu", "liberacion", "procesamiento_gpu", "salida", "servidor_total")
# MODO=servicio      -> cliente en el MISMO equipo que el servidor (localhost)
# MODO=servicio_red  -> cliente en OTRO equipo, a través de la red local
MODO = os.environ.get("MODO", "servicio")
CLIENTE = socket.gethostname() + (" (otro equipo, red local)" if MODO == "servicio_red"
                                  else " (mismo equipo que el servidor)")


def una_solicitud(url, datos, nombre, filtro, k):
    """Devuelve (código HTTP, JSON, tiempo del cliente en ms). El tiempo incluye la descarga del PNG."""
    t0 = time.perf_counter()
    r = requests.post(f"{url}/api/procesar", files={"imagen": (nombre, datos, "application/octet-stream")},
                      data={"filtro": filtro, "k": str(k)}, timeout=600)
    try:
        j = r.json()
    except ValueError:
        j = {"error": f"HTTP {r.status_code} sin JSON"}
    if r.status_code == 200:
        png = requests.get(url + j["archivos"]["png"], timeout=600)
        png.raise_for_status()
        _ = png.content
    return r.status_code, j, (time.perf_counter() - t0) * 1000.0


def registrar(csv, j, codigo, t_cliente, rep, concurrencia, base):
    sid = j.get("id") or "sin_id"
    if codigo != 200:
        registro.escribir(csv, id_solicitud=sid, repeticion=rep, concurrencia=concurrencia, fase="cliente_total",
                          tiempo=t_cliente, estado=f"rechazada_{codigo}_{j.get('error')}", **base)
        return
    comun = dict(id_solicitud=sid, repeticion=rep, concurrencia=concurrencia, bloques=j["bloques"],
                 entorno=j["entorno"], **base)
    for fase in FASES_SERVIDOR:
        registro.escribir(csv, fase=fase, tiempo=float(j["tiempos"][fase]), **comun)
    registro.escribir(csv, fase="cliente_total", tiempo=t_cliente, **comun)


def base_config(salud, imagen, filtro, k):
    from PIL import Image
    with Image.open(imagen) as im:
        ancho, alto = im.size
        canales = len(im.getbands())
    return dict(modo=MODO, filtro=filtro, ancho=ancho, alto=alto, canales=canales, k=k,
                hilos_bloque=salud["hilos_bloque"])


def registrar_arranque(csv, salud, base):
    """Arranque en frío del servicio (contexto, compilación y calentamiento al iniciar), una vez por CSV."""
    if os.path.exists(csv):
        return
    for fase in ("init_ms", "compilacion_ms", "calentamiento_ms"):
        registro.escribir(csv, id_solicitud="arranque_servicio", repeticion=0, concurrencia=0,
                          fase=fase.replace("_ms", ""), tiempo=float(salud[fase]),
                          entorno=salud["entorno"], **base)


def serie(url, imagen, filtro, k, reps, csv):
    salud = requests.get(f"{url}/api/salud", timeout=30).json()
    base = base_config(salud, imagen, filtro, k)
    registrar_arranque(csv, salud, base)
    with open(imagen, "rb") as f:
        datos = f.read()
    nombre = os.path.basename(imagen)
    # Repetición 0: primera solicitud de esta configuración (calentamiento), se registra aparte
    codigo, j, t = una_solicitud(url, datos, nombre, filtro, k)
    registrar(csv, j, codigo, t, 0, 1, base)
    kern, srv, cli, fallos = [], [], [], 0
    for rep in range(1, reps + 1):
        codigo, j, t = una_solicitud(url, datos, nombre, filtro, k)
        registrar(csv, j, codigo, t, rep, 1, base)
        if codigo == 200:
            kern.append(j["tiempos"]["kernel"]); srv.append(j["tiempos"]["servidor_total"]); cli.append(t)
        else:
            fallos += 1
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    print(f"Servicio ({CLIENTE}) {filtro} k={k} {base['ancho']}x{base['alto']}x{base['canales']} | "
          f"kernel {med(kern):.3f} ms | servidor {med(srv):.1f} ms | cliente {med(cli):.1f} ms "
          f"(mediana de {len(cli)}) | fallos {fallos}")
    return 0 if fallos == 0 else 2


def concurrencia(url, imagen, filtro, k, clientes, rondas, csv):
    salud = requests.get(f"{url}/api/salud", timeout=30).json()
    base = base_config(salud, imagen, filtro, k)
    registrar_arranque(csv, salud, base)
    with open(imagen, "rb") as f:
        datos = f.read()
    nombre = os.path.basename(imagen)
    una_solicitud(url, datos, nombre, filtro, k)            # calentamiento, no se registra
    for ronda in range(1, rondas + 1):
        res = [None] * clientes
        barrera = threading.Barrier(clientes)

        def hilo(i):
            barrera.wait()                                    # todos salen a la vez
            res[i] = una_solicitud(url, datos, f"c{i}_{nombre}", filtro, k)

        t0 = time.perf_counter()
        hs = [threading.Thread(target=hilo, args=(i,)) for i in range(clientes)]
        for h in hs:
            h.start()
        for h in hs:
            h.join()
        intervalo = (time.perf_counter() - t0) * 1000.0
        exitosas = sum(1 for r in res if r[0] == 200)
        for codigo, j, t in res:
            registrar(csv, j, codigo, t, ronda, clientes, base)
        rend = exitosas / (intervalo / 1000.0)
        registro.escribir(csv, id_solicitud=f"ronda_{clientes}c_{ronda}", repeticion=ronda, concurrencia=clientes,
                          fase="intervalo_ronda", tiempo=intervalo, **base)
        registro.escribir(csv, id_solicitud=f"ronda_{clientes}c_{ronda}", repeticion=ronda, concurrencia=clientes,
                          fase="rendimiento", tiempo=rend, unidad="solicitudes/s", **base)
        lat = ", ".join(f"{r[2]:.0f}" for r in res)
        esp = ", ".join(f"{r[1]['tiempos']['espera_cola']:.0f}" if r[0] == 200 else str(r[0]) for r in res)
        print(f"{clientes} cliente(s) ronda {ronda}: latencias [{lat}] ms | espera en cola [{esp}] ms | "
              f"exitosas {exitosas}/{clientes} | ronda {intervalo:.0f} ms | {rend:.2f} solicitudes/s")
    red = "red local hacia " + url if MODO == "servicio_red" else "localhost (sin red física)"
    print(f"Cliente: {CLIENTE}; red: {red}. Política: FIFO, 1 solicitud a la vez en la GPU.")
    return 0


def main(argv):
    if len(argv) == 8 and argv[1] == "serie":
        return serie(argv[2], argv[3], argv[4], int(argv[5]), int(argv[6]), argv[7])
    if len(argv) == 9 and argv[1] == "concurrencia":
        return concurrencia(argv[2], argv[3], argv[4], int(argv[5]), int(argv[6]), int(argv[7]), argv[8])
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))