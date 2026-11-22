"""
Sirve como referencia para comparar con el servicio:
  - en el anfitrión  : "PyCUDA directo en el anfitrión"
  - en el contenedor : "PyCUDA directo dentro del contenedor"
(el CSV lo distingue solo, en la columna 'entorno').
 - Filtros: media (3, 9, 21), nitidez (3), laplaciano (3), gaussiano (5), relieve (3)
"""
import os
import sys
import uuid

import numpy as np

import registro
from motor_gpu import MotorConvolucion, a_visual, comun, guardar_raw, leer_imagen

FASES = ("preparacion", "reserva", "h2d", "kernel", "d2h", "total_gpu", "liberacion", "procesamiento")


def prueba_manual():
    """Matriz 5x5 con 4 hilos/bloque: 25 píxeles -> 7 bloques = 28 hilos (3 sobrantes).
    Se prueba en gris (1 plano) y a color (la misma matriz en los 3 planos)."""
    motor = MotorConvolucion(hilos=4)
    try:
        m = comun.MATRIZ_PRUEBA
        sal, info = motor.procesar(m, "laplaciano", 3)
        print(f"GPU: {motor.gpu} | {info['hilos']} hilos x {info['bloques']} bloques")
        ok = comun.verificar_prueba(sal[0], "motor P03, gris, 4 hilos/bloque")
        rgb = np.stack([m, m, m])
        sal3, _ = motor.procesar(rgb, "laplaciano", 3)
        for c, nombre in enumerate("RGB"):
            ok &= comun.verificar_prueba(sal3[c], f"motor P03, color canal {nombre}")
        return 0 if ok else 1
    finally:
        motor.cerrar()


def main(argv):
    if len(argv) == 2 and argv[1] == "--test":
        return prueba_manual()
    if len(argv) != 8:
        print(__doc__)
        return 1
    ruta, filtro = argv[1], argv[2]
    k, hilos, reps = int(argv[3]), int(argv[4]), int(argv[5])
    prefijo, csv_ruta = argv[6], argv[7]

    motor = MotorConvolucion(hilos=hilos)
    try:
        img = leer_imagen(ruta)
        canales, alto, ancho = img.shape
        base = dict(modo="directo", filtro=filtro, ancho=ancho, alto=alto, canales=canales,
                    k=k, hilos_bloque=hilos, concurrencia=1)

        # Arranque en frío: registrado aparte (repetición 0), fuera de las medianas
        registro.escribir(csv_ruta, id_solicitud="arranque", repeticion=0, fase="init",
                          tiempo=motor.t_init, **base)
        registro.escribir(csv_ruta, id_solicitud="arranque", repeticion=0, fase="compilacion",
                          tiempo=motor.t_comp, **base)

        # Repetición 0: primera ejecución (calentamiento), registrada aparte; no entra en las medianas
        _, info0 = motor.procesar(img, filtro, k)
        for fase in FASES:
            registro.escribir(csv_ruta, id_solicitud="calentamiento", repeticion=0, fase=fase,
                              tiempo=info0["tiempos"][fase], bloques=info0["bloques"], **base)

        kern, proc = [], []
        for rep in range(1, reps + 1):
            sal, info = motor.procesar(img, filtro, k)
            sid = uuid.uuid4().hex[:12]
            for fase in FASES:
                registro.escribir(csv_ruta, id_solicitud=sid, repeticion=rep, fase=fase,
                                  tiempo=info["tiempos"][fase], bloques=info["bloques"], **base)
            kern.append(info["tiempos"]["kernel"])
            proc.append(info["tiempos"]["procesamiento"])

        os.makedirs(os.path.dirname(os.path.abspath(prefijo)), exist_ok=True)
        guardar_raw(prefijo + ".f32", sal)        # salidas: fuera de la medición
        a_visual(sal).save(prefijo + ".png")
        print(f"Directo ({registro.entorno()}) {filtro} k={k} {ancho}x{alto}x{canales} | {motor.gpu} | "
              f"{hilos} hilos x {info['bloques']} bloques x {canales} planos | "
              f"init {motor.t_init:.1f} ms | compilación {motor.t_comp:.1f} ms | "
              f"kernel {np.median(kern):.3f} ms | procesamiento {np.median(proc):.3f} ms "
              f"(mediana, {reps} reps)")
        return 0
    finally:
        motor.cerrar()


if __name__ == "__main__":
    sys.exit(main(sys.argv))