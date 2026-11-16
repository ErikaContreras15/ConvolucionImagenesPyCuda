
import sys

import numpy as np


def main(argv):
    if len(argv) not in (5, 6):
        print(__doc__)
        return 1
    ref = np.fromfile(argv[1], dtype=np.float32)
    otra = np.fromfile(argv[2], dtype=np.float32)
    etiqueta, archivo = argv[3], argv[4]
    tol = float(argv[5]) if len(argv) == 6 else 1e-2

    if ref.size != otra.size:
        linea = f"{etiqueta} | ERROR: tamaños distintos ({ref.size} vs {otra.size})"
        ok = False
    else:
        err = np.abs(ref.astype(np.float64) - otra.astype(np.float64))
        fuera = int((err > tol).sum())
        ok = fuera == 0
        linea = (f"{etiqueta} | píxeles {ref.size} | error máx {err.max():.3e} | "
                 f"error medio {err.mean():.3e} | > tol {fuera} | {'OK' if ok else 'ERROR'}")
    print(linea)
    with open(archivo, "a") as f:
        f.write(linea + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
