"""Genera tablas y gráficas de la Práctica 01 a partir de los CSV originales.

Uso (desde la carpeta Practica01):
    python3 graficas.py CPU/resultados.csv CUDA/resultados.csv graficas/

Salidas en la carpeta indicada:
    resumen.csv        mediana, Q1, Q3, IQR, mínimo y máximo por configuración y fase
    aceleraciones.csv  aceleración de cómputo y con transferencias, % de transferencias
    g1_filtros.png, g2_resolucion.png, g3_mascara.png, g4_hilos.png, g5_transferencias.png

Variabilidad: rango intercuartílico (IQR = Q3 - Q1) y rango mínimo-máximo.
Las barras de error de las gráficas muestran el rango mínimo-máximo de las 5 repeticiones.
Requiere: sudo apt install python3-matplotlib
"""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HILOS_FIJO = 256


def leer(rutas):
    """Agrupa los tiempos por (impl, filtro, n, k, hilos, fase)."""
    datos = defaultdict(list)
    for ruta in rutas:
        with open(ruta, newline="") as f:
            for r in csv.DictReader(f):
                clave = (r["implementacion"], r["filtro"], int(r["ancho"]),
                         int(r["k"]), int(r["hilos_bloque"]), r["fase"])
                datos[clave].append(float(r["tiempo_ms"]))
    return datos


def estad(v):
    q1, _, q3 = st.quantiles(v, n=4, method="inclusive")
    return {"n": len(v), "mediana": st.median(v), "q1": q1, "q3": q3,
            "iqr": q3 - q1, "min": min(v), "max": max(v)}


def main(cpu_csv, cuda_csv, carpeta):
    out = Path(carpeta)
    out.mkdir(parents=True, exist_ok=True)
    datos = leer([cpu_csv, cuda_csv])
    E = {c: estad(v) for c, v in datos.items()}

    # ---------- resumen.csv ----------
    with open(out / "resumen.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["implementacion", "filtro", "resolucion", "k", "hilos_bloque", "fase",
                    "repeticiones", "mediana_ms", "q1_ms", "q3_ms", "iqr_ms",
                    "min_ms", "max_ms", "iqr_pct_mediana"])
        for c in sorted(E):
            e = E[c]
            w.writerow([*c, e["n"], f"{e['mediana']:.4f}", f"{e['q1']:.4f}",
                        f"{e['q3']:.4f}", f"{e['iqr']:.4f}", f"{e['min']:.4f}",
                        f"{e['max']:.4f}", f"{100 * e['iqr'] / e['mediana']:.2f}"])

    def cpu(f, n, k):
        return E.get(("cpu", f, n, k, 0, "total"))

    def gpu(f, n, k, h, fase):
        return E.get(("cuda", f, n, k, h, fase))

    # ---------- aceleraciones.csv ----------
    filas = []
    for (impl, f, n, k, h, fase) in sorted(E):
        if impl != "cuda" or fase != "total":
            continue
        c = cpu(f, n, k)
        if c is None:
            continue
        ker, tot = gpu(f, n, k, h, "kernel"), gpu(f, n, k, h, "total")
        transf = gpu(f, n, k, h, "h2d")["mediana"] + gpu(f, n, k, h, "d2h")["mediana"]
        filas.append([f, n, k, h, c["mediana"], ker["mediana"], tot["mediana"],
                      c["mediana"] / ker["mediana"], c["mediana"] / tot["mediana"],
                      100 * transf / tot["mediana"]])
    with open(out / "aceleraciones.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filtro", "resolucion", "k", "hilos_bloque", "cpu_ms", "kernel_ms",
                    "total_gpu_ms", "acel_computo", "acel_con_transferencias",
                    "pct_transferencias"])
        for r in filas:
            w.writerow(r[:4] + [f"{x:.4f}" for x in r[4:7]] + [f"{x:.2f}" for x in r[7:]])

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

    def barra_err(e):
        return [[e["mediana"] - e["min"]], [e["max"] - e["mediana"]]]

    def barras(ax, xs, series, ancho=0.27):
        for i, (nombre, valores) in enumerate(series):
            pos = [x + (i - (len(series) - 1) / 2) * ancho for x in range(len(xs))]
            meds = [v["mediana"] for v in valores]
            errs = [[v["mediana"] - v["min"] for v in valores],
                    [v["max"] - v["mediana"] for v in valores]]
            b = ax.bar(pos, meds, ancho, yerr=errs, capsize=3, label=nombre)
            ax.bar_label(b, fmt="%.2f", fontsize=7, padding=2)
        ax.set_xticks(range(len(xs)), xs)

    # ---------- G1: filtros ----------
    fil = ["media", "nitidez", "laplaciano"]
    fig, ax = plt.subplots(figsize=(7, 4))
    barras(ax, ["Media 3×3", "Nitidez 3×3", "Laplaciano 3×3"], [
        ("CPU", [cpu(f, 4000, 3) for f in fil]),
        ("CUDA kernel", [gpu(f, 4000, 3, HILOS_FIJO, "kernel") for f in fil]),
        ("CUDA total", [gpu(f, 4000, 3, HILOS_FIJO, "total") for f in fil])])
    ax.set_yscale("log"); ax.set_ylabel("Tiempo (ms, escala log)")
    ax.set_ylim(top=ax.get_ylim()[1] * 4)
    ax.set_title(f"Experimento 1: filtros 3×3, imagen 4000×4000, {HILOS_FIJO} hilos/bloque")
    ax.legend(loc="upper center", ncols=3); fig.tight_layout(); fig.savefig(out / "g1_filtros.png"); plt.close(fig)

    # ---------- G2: resolución ----------
    res = [2000, 4000, 6000]
    fig, ax = plt.subplots(figsize=(7, 4))
    for nombre, getter in [("CPU", lambda n: cpu("media", n, 9)),
                           ("CUDA kernel", lambda n: gpu("media", n, 9, HILOS_FIJO, "kernel")),
                           ("CUDA total", lambda n: gpu("media", n, 9, HILOS_FIJO, "total"))]:
        vals = [getter(n) for n in res]
        ax.errorbar([n * n / 1e6 for n in res], [v["mediana"] for v in vals],
                    yerr=[[v["mediana"] - v["min"] for v in vals],
                          [v["max"] - v["mediana"] for v in vals]],
                    marker="o", capsize=3, label=nombre)
    ax.set_yscale("log"); ax.set_xlabel("Píxeles (millones): 2000², 4000², 6000²")
    ax.set_ylabel("Tiempo (ms, escala log)")
    ax.set_title(f"Experimento 2: resolución, media 9×9, {HILOS_FIJO} hilos/bloque")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(out / "g2_resolucion.png"); plt.close(fig)

    # ---------- G3: máscara ----------
    ks = [3, 9, 21]
    fig, ax = plt.subplots(figsize=(7, 4))
    for nombre, getter in [("CPU", lambda k: cpu("media", 4000, k)),
                           ("CUDA kernel", lambda k: gpu("media", 4000, k, HILOS_FIJO, "kernel")),
                           ("CUDA total", lambda k: gpu("media", 4000, k, HILOS_FIJO, "total"))]:
        vals = [getter(k) for k in ks]
        ax.errorbar([k * k for k in ks], [v["mediana"] for v in vals],
                    yerr=[[v["mediana"] - v["min"] for v in vals],
                          [v["max"] - v["mediana"] for v in vals]],
                    marker="o", capsize=3, label=nombre)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([9, 81, 441], ["3×3 (9)", "9×9 (81)", "21×21 (441)"])
    ax.set_xlabel("Tamaño de máscara (coeficientes por píxel)")
    ax.set_ylabel("Tiempo (ms, escala log)")
    ax.set_title(f"Experimento 3: máscara, imagen 4000×4000, {HILOS_FIJO} hilos/bloque")
    ax.grid(alpha=.3, which="both"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "g3_mascara.png"); plt.close(fig)

    # ---------- G4: hilos por bloque ----------
    hs = [128, 256, 512]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, fase, titulo in [(axs[0], "kernel", "Kernel"), (axs[1], "total", "Total (con transferencias)")]:
        vals = [gpu("media", 4000, 9, h, fase) for h in hs]
        b = ax.bar([str(h) for h in hs], [v["mediana"] for v in vals],
                   yerr=[[v["mediana"] - v["min"] for v in vals],
                         [v["max"] - v["mediana"] for v in vals]],
                   capsize=4, color="#4C72B0" if fase == "kernel" else "#DD8452")
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xlabel("Hilos por bloque"); ax.set_ylabel("Tiempo (ms)"); ax.set_title(titulo)
    fig.suptitle("Experimento 4: hilos por bloque, media 9×9, imagen 4000×4000")
    fig.tight_layout(); fig.savefig(out / "g4_hilos.png"); plt.close(fig)

    # ---------- G5: desglose de tiempo en GPU ----------
    confs = [("Media 3×3\n4000²", "media", 4000, 3), ("Nitidez 3×3\n4000²", "nitidez", 4000, 3),
             ("Lapl. 3×3\n4000²", "laplaciano", 4000, 3), ("Media 9×9\n2000²", "media", 2000, 9),
             ("Media 9×9\n4000²", "media", 4000, 9), ("Media 9×9\n6000²", "media", 6000, 9),
             ("Media 21×21\n4000²", "media", 4000, 21)]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    base = [0.0] * len(confs)
    for fase, nombre, color in [("h2d", "CPU→GPU", "#55A868"), ("kernel", "Kernel", "#4C72B0"),
                                ("d2h", "GPU→CPU", "#C44E52")]:
        tot = [gpu(f, n, k, HILOS_FIJO, "total")["mediana"] for _, f, n, k in confs]
        pct = [100 * gpu(f, n, k, HILOS_FIJO, fase)["mediana"] / t
               for (_, f, n, k), t in zip(confs, tot)]
        ax.bar([c[0] for c in confs], pct, bottom=base, label=nombre, color=color)
        base = [b + p for b, p in zip(base, pct)]
    ax.set_ylabel("% del tiempo total en GPU (medianas)"); ax.set_ylim(0, 105)
    ax.set_title(f"Proporción de transferencias vs kernel ({HILOS_FIJO} hilos/bloque)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1)); fig.tight_layout()
    fig.savefig(out / "g5_transferencias.png"); plt.close(fig)

    # ---------- resumen en consola ----------
    print(f"{'filtro':<11}{'res':>6}{'k':>4}{'hilos':>6}{'CPU ms':>10}{'kernel':>9}"
          f"{'total':>9}{'ac.comp':>9}{'ac.transf':>10}{'%transf':>8}")
    for r in filas:
        print(f"{r[0]:<11}{r[1]:>6}{r[2]:>4}{r[3]:>6}{r[4]:>10.2f}{r[5]:>9.3f}"
              f"{r[6]:>9.3f}{r[7]:>9.1f}{r[8]:>10.1f}{r[9]:>8.1f}")
    print(f"\nArchivos generados en {out}/")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Uso: python3 graficas.py <cpu.csv> <cuda.csv> <carpeta_salida>")
    main(*sys.argv[1:])
