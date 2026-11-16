
import sys
import time

import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comun


class LimiteSuperado(Exception):
    pass


def convolucion_python(entrada, ancho, alto, mascara, k, limite_s=None):
    """entrada y mascara: listas de np.float32 (1D, fila por fila).
    Devuelve la salida como lista de np.float32 (arreglo separado de la entrada).
    Cada píxel de salida se calcula recorriendo explícitamente la máscara."""
    r = k // 2
    cero = np.float32(0.0)
    salida = [cero] * (ancho * alto)
    inicio = time.perf_counter()
    for fila in range(alto):
        # Control de tiempo una vez por fila (costo despreciable)
        if limite_s is not None and time.perf_counter() - inicio > limite_s:
            raise LimiteSuperado(fila)
        for col in range(ancho):
            suma = cero
            for i in range(-r, r + 1):
                y = fila + i
                if y < 0 or y >= alto:
                    continue                      # fuera de la imagen: aporta 0
                for j in range(-r, r + 1):
                    x = col + j
                    if x < 0 or x >= ancho:
                        continue                  # fuera de la imagen: aporta 0
                    # producto y suma en float32, como 'float' en C
                    suma = suma + entrada[y * ancho + x] * mascara[(i + r) * k + (j + r)]
            salida[fila * ancho + col] = suma     # arreglo de salida separado
    return salida


def preparar(arr2d, mascara):
    """Convierte los arreglos NumPy a listas de np.float32 (fuera de la medición):
    indexar una lista de Python es mucho más rápido que indexar un arreglo NumPy
    elemento por elemento, y conserva el tipo float32."""
    return list(arr2d.ravel()), list(mascara)


def prueba_manual():
    img = comun.MATRIZ_PRUEBA
    m, k = comun.crear_mascara("laplaciano", 3)
    ent, msk = preparar(img, m)
    sal = convolucion_python(ent, 5, 5, msk, k)
    salida = np.array(sal, dtype=np.float32).reshape(5, 5)
    return 0 if comun.verificar_prueba(salida, "Python secuencial") else 1


def main(argv):
    if len(argv) == 2 and argv[1] == "--test":
        return prueba_manual()
    if len(argv) not in (7, 8):
        print(__doc__)
        return 1

    ruta, filtro, k, reps, prefijo, csv_ruta = argv[1], argv[2], int(argv[3]), int(argv[4]), argv[5], argv[6]
    limite_s = float(argv[7]) if len(argv) == 8 else 3600.0

    # --- Preparación: lectura, máscara y conversión a listas ---
    img = comun.leer_pgm(ruta)
    alto, ancho = img.shape
    mascara, k = comun.crear_mascara(filtro, k)
    entrada, msk = preparar(img, mascara)

    # --- Calentamiento ---
    rec = np.ascontiguousarray(img[:64, :64])
    ent_c, _ = preparar(rec, mascara)
    convolucion_python(ent_c, rec.shape[1], rec.shape[0], msk, k)

    # --- Repeticiones medidas ---
    tiempos = []
    salida = None
    for rep in range(1, reps + 1):
        t0 = time.perf_counter()
        try:
            salida = convolucion_python(entrada, ancho, alto, msk, k, limite_s)
        except LimiteSuperado as e:
            transcurrido = (time.perf_counter() - t0) * 1000.0
            comun.escribir_csv(csv_ruta, "python", filtro, ancho, alto, k, 0, 0,
                               rep, "total", transcurrido, "no_completada")
            print(f"Python {filtro} k={k} {ancho}x{alto} | NO COMPLETADA: superó {limite_s:.0f} s "
                  f"(se detuvo en la fila {e.args[0]} de {alto}; {transcurrido/1000:.1f} s transcurridos)")
            return 2
        ms = (time.perf_counter() - t0) * 1000.0
        tiempos.append(ms)
        comun.escribir_csv(csv_ruta, "python", filtro, ancho, alto, k, 0, 0, rep, "total", ms)
        print(f"  repetición {rep}: {ms/1000:.2f} s", flush=True)

    # --- Salidas (NO se miden) ---
    sal = np.array(salida, dtype=np.float32).reshape(alto, ancho)
    comun.guardar_raw(prefijo + ".f32", sal)
    comun.guardar_pgm(prefijo + ".pgm", sal)
    print(f"Python {filtro} k={k} {ancho}x{alto} | mediana {comun.mediana(tiempos):.2f} ms ({reps} reps)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
