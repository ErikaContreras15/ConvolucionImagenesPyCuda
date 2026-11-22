
import csv
import os

COLUMNAS = ["id_solicitud", "modo", "entorno", "filtro", "ancho", "alto", "canales", "k",
            "hilos_bloque", "bloques", "repeticion", "concurrencia",
            "fase", "tiempo", "unidad", "estado"]


def entorno():
    """'contenedor' si se ejecuta dentro de Docker, 'anfitrion' si no."""
    return "contenedor" if os.path.exists("/.dockerenv") else "anfitrion"


def escribir(ruta, **fila):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    nuevo = not os.path.exists(ruta) or os.path.getsize(ruta) == 0
    fila.setdefault("unidad", "ms")
    fila.setdefault("estado", "completada")
    fila.setdefault("entorno", entorno())
    if isinstance(fila.get("tiempo"), float):
        fila["tiempo"] = f"{fila['tiempo']:.4f}"
    with open(ruta, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if nuevo:
            w.writeheader()
        w.writerow({c: fila.get(c, "") for c in COLUMNAS})