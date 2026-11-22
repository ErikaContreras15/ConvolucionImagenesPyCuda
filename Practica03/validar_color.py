"""Valida la salida A COLOR de la P03 contra la P02, canal por canal.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

AQUI = os.path.dirname(os.path.abspath(__file__))
CONV_P02 = os.path.join(os.path.dirname(AQUI), "Practica02", "PyCuda", "conv_pycuda.py")
TOL = 1e-2   # misma tolerancia que en las prácticas 01 y 02


def main(argv):
    if len(argv) not in (6, 7):
        print(__doc__)
        return 1
    ruta, filtro, k, f32_p03, archivo = argv[1], argv[2], argv[3], argv[4], argv[5]
    hilos = argv[6] if len(argv) == 7 else "256"

    with Image.open(ruta) as img:
        rgb = img.convert("RGB")
        canales = rgb.split()
        ancho, alto = rgb.size
    p03 = np.fromfile(f32_p03, dtype=np.float32)
    if p03.size != 3 * ancho * alto:
        print(f"ERROR: {f32_p03} tiene {p03.size} valores; se esperaban {3 * ancho * alto}")
        return 1
    p03 = p03.reshape(3, alto, ancho)

    ok_total = True
    lineas = []
    with tempfile.TemporaryDirectory() as tmp:
        for c, (nombre, canal) in enumerate(zip("RGB", canales)):
            pgm = os.path.join(tmp, f"canal_{nombre}.pgm")
            canal.save(pgm)
            prefijo = os.path.join(tmp, f"p02_{nombre}")
            r = subprocess.run([sys.executable, CONV_P02, pgm, filtro, k, hilos, "1",
                                prefijo, os.path.join(tmp, "p02.csv")],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(r.stdout, r.stderr)
                return 1
            ref = np.fromfile(prefijo + ".f32", dtype=np.float32).reshape(alto, ancho)
            err = np.abs(ref.astype(np.float64) - p03[c].astype(np.float64))
            fuera = int((err > TOL).sum())
            ok = fuera == 0
            ok_total &= ok
            lineas.append(f"Color vs P02 | {filtro} k={k} {ancho}x{alto} canal {nombre} | "
                          f"píxeles {err.size} | error máx {err.max():.3e} | "
                          f"error medio {err.mean():.3e} | > tol {fuera} | {'OK' if ok else 'ERROR'}")

    os.makedirs(os.path.dirname(os.path.abspath(archivo)), exist_ok=True)
    with open(archivo, "a") as f:
        for l in lineas:
            print(l)
            f.write(l + "\n")
    return 0 if ok_total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
