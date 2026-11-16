"""
  - init        : inicialización de CUDA y creación del contexto (una vez, time.perf_counter)
  - compilacion : SourceModule compila el kernel con nvcc (una vez, time.perf_counter).
                  Se usa cache_dir=False para que compile de verdad y no lea una copia guardada.
  - h2d, kernel, d2h, total : por repetición, con eventos CUDA (igual que en CUDA C).
  Init y compilación NO se suman a h2d/kernel/d2h/total: se reportan aparte.

"""
import os
import sys
import time

import numpy as np

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import comun

RUTA_KERNEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_convolucion.cu")


# ---------- Inicialización y compilación (medidas por separado) ----------
def iniciar_cuda():
    """Equivale a 'import pycuda.autoinit', pero hecho a mano para poder medirlo."""
    t0 = time.perf_counter()
    import pycuda.driver as drv
    drv.init()
    ctx = drv.Device(0).make_context()
    t_init = (time.perf_counter() - t0) * 1000.0
    return drv, ctx, t_init


def compilar_kernel():
    from pycuda.compiler import SourceModule
    with open(RUTA_KERNEL) as f:
        codigo = f.read()
    t0 = time.perf_counter()
    mod = SourceModule(codigo, options=["-O2"], cache_dir=False)
    func = mod.get_function("convolucionGPU")
    t_comp = (time.perf_counter() - t0) * 1000.0
    return func, t_comp


# ---------- Un ciclo completo: CPU->GPU, kernel, GPU->CPU ----------
def ejecutar(drv, func, h_in, h_out, h_m, d_in, d_out, d_m, ancho, alto, k, hilos):
    """Igual que ejecutar() de conv_cuda.cu. La memoria de GPU ya está reservada (no se mide)."""
    N = ancho * alto
    bloques = (N + hilos - 1) // hilos
    e0, e1, e2, e3 = drv.Event(), drv.Event(), drv.Event(), drv.Event()

    e0.record()
    drv.memcpy_htod(d_in, h_in)                      # imagen CPU -> GPU
    drv.memcpy_htod(d_m, h_m)                        # máscara CPU -> GPU
    e1.record()
    func(d_in, d_out, np.int32(ancho), np.int32(alto), d_m, np.int32(k),
         block=(hilos, 1, 1), grid=(bloques, 1))     # misma configuración 1D que CUDA C
    e2.record()
    drv.memcpy_dtoh(h_out, d_out)                    # resultado GPU -> CPU
    e3.record()
    e3.synchronize()                                 # asegurar que todo terminó

    return {"h2d": e0.time_till(e1), "kernel": e1.time_till(e2),
            "d2h": e2.time_till(e3), "total": e0.time_till(e3)}


# ---------- Prueba manual (estilo hello_gpu.py de clase: drv.In / drv.Out) ----------
def prueba_manual():
    drv, ctx, _ = iniciar_cuda()
    try:
        func, _ = compilar_kernel()
        img = comun.MATRIZ_PRUEBA.ravel().copy()
        m, k = comun.crear_mascara("laplaciano", 3)
        out = np.zeros_like(img)
        hilos = 4                                    # 25 píxeles -> 7 bloques = 28 hilos (3 sobrantes)
        bloques = (25 + hilos - 1) // hilos
        func(drv.In(img), drv.Out(out), np.int32(5), np.int32(5), drv.In(m), np.int32(k),
             block=(hilos, 1, 1), grid=(bloques, 1))
        ok = comun.verificar_prueba(out.reshape(5, 5), f"PyCUDA, {hilos} hilos/bloque")
        return 0 if ok else 1
    finally:
        ctx.pop()


def main(argv):
    if len(argv) == 2 and argv[1] == "--test":
        return prueba_manual()
    if len(argv) != 8:
        print(__doc__)
        return 1

    ruta, filtro = argv[1], argv[2]
    k, hilos, reps = int(argv[3]), int(argv[4]), int(argv[5])
    prefijo, csv_ruta = argv[6], argv[7]

    drv, ctx, t_init = iniciar_cuda()
    try:
        max_hilos = drv.Context.get_device().get_attribute(drv.device_attribute.MAX_THREADS_PER_BLOCK)
        if hilos < 1 or hilos > max_hilos:
            raise ValueError("hilos_bloque fuera del rango del dispositivo")
        func, t_comp = compilar_kernel()

        # --- Preparación ---
        img = comun.leer_pgm(ruta)
        alto, ancho = img.shape
        mascara, k = comun.crear_mascara(filtro, k)
        N = ancho * alto
        bloques = (N + hilos - 1) // hilos
        h_in = img.ravel()                           # float32 contiguo
        h_out = np.empty(N, dtype=np.float32)
        d_in = drv.mem_alloc(h_in.nbytes)            # reserva previa, fuera de la medición
        d_out = drv.mem_alloc(h_out.nbytes)
        d_m = drv.mem_alloc(mascara.nbytes)

        comun.escribir_csv(csv_ruta, "pycuda", filtro, ancho, alto, k, hilos, bloques, 0, "init", t_init)
        comun.escribir_csv(csv_ruta, "pycuda", filtro, ancho, alto, k, hilos, bloques, 0, "compilacion", t_comp)

        # --- Calentamiento (no se registra) ---
        ejecutar(drv, func, h_in, h_out, mascara, d_in, d_out, d_m, ancho, alto, k, hilos)

        # --- Repeticiones medidas ---
        tk, tt = [], []
        for rep in range(1, reps + 1):
            t = ejecutar(drv, func, h_in, h_out, mascara, d_in, d_out, d_m, ancho, alto, k, hilos)
            for fase in ("h2d", "kernel", "d2h", "total"):
                comun.escribir_csv(csv_ruta, "pycuda", filtro, ancho, alto, k, hilos, bloques, rep, fase, t[fase])
            tk.append(t["kernel"]); tt.append(t["total"])

        d_in.free(); d_out.free(); d_m.free()

        # --- Salidas (NO se miden) ---
        sal = h_out.reshape(alto, ancho)
        comun.guardar_raw(prefijo + ".f32", sal)
        comun.guardar_pgm(prefijo + ".pgm", sal)
        print(f"PyCUDA {filtro} k={k} {ancho}x{alto} | {hilos} hilos x {bloques} bloques | "
              f"init {t_init:.1f} ms | compilación {t_comp:.1f} ms | "
              f"kernel {comun.mediana(tk):.3f} ms | total {comun.mediana(tt):.3f} ms (mediana, {reps} reps)")
        return 0
    finally:
        ctx.pop()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
