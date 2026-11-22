"""Crea una imagen de prueba de 1003x997 a partir de img_2000.pgm.

1003 y 997 no son múltiplos de 128, 256 ni 512, así que el último bloque
tiene hilos sobrantes: sirve para comprobar el control de límites.

Uso: python3 crear_prueba.py img/img_2000.pgm img/prueba_1003x997.pgm
"""
import sys
from PIL import Image

if len(sys.argv) != 3:
    sys.exit("Uso: python3 crear_prueba.py <entrada.pgm> <salida.pgm>")
Image.open(sys.argv[1]).convert("L").crop((0, 0, 1003, 997)).save(sys.argv[2])
print("Guardada", sys.argv[2], "(1003x997)")
