
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

version = sys.argv[1] if len(sys.argv) > 1 else "pycuda"
x0, y0, lado = (int(v) for v in sys.argv[2:5]) if len(sys.argv) == 5 else (1800, 1800, 300)
sufijo = "_h256" if version in ("pycuda", "cuda") else ""

archivos = [("Original", "../CPU/imgCPU/img_4000.pgm"),
            ("Media 3×3", f"salidas/{version}_media3_4000{sufijo}.pgm"),
            ("Nitidez 3×3", f"salidas/{version}_nitidez3_4000{sufijo}.pgm"),
            ("Laplaciano 3×3", f"salidas/{version}_laplaciano3_4000{sufijo}.pgm")]
nombres = {"pycuda": "PyCUDA", "cuda": "CUDA C", "py": "Python secuencial", "cpp": "C/C++ secuencial"}

fig, ejes = plt.subplots(2, 4, figsize=(16, 8.6))
for c, (titulo, ruta) in enumerate(archivos):
    if not os.path.exists(ruta):
        sys.exit(f"No existe {ruta}: ¿terminó ejecutar_p02.sh?")
    img = np.asarray(Image.open(ruta).convert("L"))
    ejes[0, c].imshow(img, cmap="gray", vmin=0, vmax=255)
    ejes[0, c].add_patch(patches.Rectangle((x0, y0), lado, lado, fill=False, edgecolor="red", linewidth=1.5))
    ejes[0, c].set_title(f"{titulo} · 4000×4000", fontsize=12)
    ejes[1, c].imshow(img[y0:y0 + lado, x0:x0 + lado], cmap="gray", vmin=0, vmax=255,
                      interpolation="nearest")
    ejes[1, c].set_title(f"Recorte {lado}×{lado} (x={x0}, y={y0})", fontsize=11)
    for f in (0, 1):
        ejes[f, c].axis("off")

fig.suptitle(f"Evidencia visual — salidas de {nombres.get(version, version)} "
             f"(recuadro rojo = región del recorte)", fontsize=14)
fig.tight_layout()
os.makedirs("capturas", exist_ok=True)
salida = f"capturas/05_evidencia_visual_{version}.png"
fig.savefig(salida, dpi=150)
print("Guardada", salida)
