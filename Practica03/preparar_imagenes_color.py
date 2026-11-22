
import sys
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main(origen: str, carpeta: str) -> None:
    salida = Path(carpeta)
    salida.mkdir(parents=True, exist_ok=True)

    img = Image.open(origen).convert("RGB")
    w, h = img.size
    lado = min(w, h)
    izq, arr = (w - lado) // 2, (h - lado) // 2
    img = img.crop((izq, arr, izq + lado, arr + lado))
    print(f"Original {w}x{h} -> recorte {lado}x{lado} (RGB)")

    for n in (2000, 4000, 6000):
        if n > lado:
            print(f"  Aviso: {n} px es mayor que el recorte ({lado}); se amplía")
        img.resize((n, n), Image.LANCZOS).save(salida / f"img_{n}.ppm")
        print(f"  Guardada img_{n}.ppm")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Uso: python3 preparar_imagenes_color.py <original> <carpeta_salida>")
    main(sys.argv[1], sys.argv[2])
