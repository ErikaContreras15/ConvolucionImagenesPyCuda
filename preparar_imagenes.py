"""Prepara las tres resoluciones a partir de UNA imagen original.

Pasos (documentar en el informe):
  1. Convertir a escala de grises (8 bits).
  2. Recorte cuadrado centrado del lado menor.
  3. Redimensionar a 2000, 4000 y 6000 px con filtro LANCZOS.
Se recomienda una imagen original de al menos 6000 px de lado menor
para no ampliar (si se amplía, indicarlo en el informe).

Uso:  python preparar_imagenes.py original.jpg img/
Requiere: pip install pillow
"""
import sys
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # permitir imágenes muy grandes

def main(origen: str, carpeta: str) -> None:
    salida = Path(carpeta)
    salida.mkdir(parents=True, exist_ok=True)

    img = Image.open(origen).convert("L")
    w, h = img.size
    lado = min(w, h)
    izq, arr = (w - lado) // 2, (h - lado) // 2
    img = img.crop((izq, arr, izq + lado, arr + lado))
    print(f"Original {w}x{h} -> recorte {lado}x{lado}")

    for n in (2000, 4000, 6000):
        if n > lado:
            print(f"  Aviso: {n} px es mayor que el recorte ({lado}); se amplía")
        img.resize((n, n), Image.LANCZOS).save(salida / f"img_{n}.pgm")
        print(f"  Guardada img_{n}.pgm")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python preparar_imagenes.py <original> <carpeta_salida>")
    main(sys.argv[1], sys.argv[2])
