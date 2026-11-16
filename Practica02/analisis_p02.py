
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, G = "resultados", "graficas"
os.makedirs(G, exist_ok=True)

NOMBRE = {"cpp": "C/C++ secuencial", "cuda": "CUDA C", "python": "Python secuencial", "pycuda": "PyCUDA"}
COLOR = {"cpp": "#2a78d6", "cuda": "#eb6834", "python": "#1baf7a", "pycuda": "#6250d6"}


# ---------- 1. Unir CSV en un solo formato ----------
def leer(ruta, version_forzada=None):
    filas = []
    if not os.path.exists(ruta):
        print("Aviso: no existe", ruta)
        return filas
    with open(ruta) as f:
        for r in csv.DictReader(f):
            filas.append({
                "version": version_forzada or r["version"],
                "filtro": r["filtro"], "ancho": int(r["ancho"]), "alto": int(r["alto"]),
                "k": int(r["k"]), "hilos_bloque": int(r["hilos_bloque"]), "bloques": int(r["bloques"]),
                "repeticion": int(r["repeticion"]), "fase": r["fase"],
                "tiempo": float(r.get("tiempo", r.get("tiempo_ms"))), "unidad": "ms",
                "estado": r.get("estado", "completada")})
    return filas


datos = (leer(f"{R}/resultados_cpp.csv", "cpp") + leer(f"{R}/resultados_cuda.csv", "cuda")
         + leer(f"{R}/resultados_pycuda.csv") + leer(f"{R}/resultados_python.csv"))
COLS = ["version", "filtro", "ancho", "alto", "k", "hilos_bloque", "bloques",
        "repeticion", "fase", "tiempo", "unidad", "estado"]
with open(f"{R}/resultados_p02.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for d in datos:
        w.writerow({**d, "tiempo": f"{d['tiempo']:.4f}"})
print(f"resultados_p02.csv: {len(datos)} registros")

# ---------- 2. Mediana y variabilidad (solo ejecuciones completas) ----------
grupos = defaultdict(list)
incompletas = set()
for d in datos:
    clave = (d["version"], d["filtro"], d["ancho"], d["k"], d["hilos_bloque"], d["fase"])
    if d["estado"] == "completada":
        grupos[clave].append(d["tiempo"])
    else:
        incompletas.add(clave[:5])


def cuartiles(v):
    if len(v) < 2:
        return v[0], v[0], v[0]
    q1, q2, q3 = statistics.quantiles(v, n=4, method="inclusive")
    return q1, statistics.median(v), q3


resumen = {}
with open(f"{R}/resumen_p02.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version", "filtro", "resolucion", "k", "hilos_bloque", "fase",
                "n", "mediana_ms", "q1_ms", "q3_ms", "iqr_ms"])
    for clave in sorted(grupos):
        v = grupos[clave]
        q1, med, q3 = cuartiles(v)
        resumen[clave] = med
        ver, fil, n, k, h, fase = clave
        w.writerow([ver, fil, f"{n}x{n}", k, h, fase, len(v),
                    f"{med:.4f}", f"{q1:.4f}", f"{q3:.4f}", f"{q3 - q1:.4f}"])


def med(ver, fil, n, k, fase, h=None):
    if h is None:
        h = 256 if ver in ("cuda", "pycuda") else 0
    if (ver, fil, n, k, h) in incompletas:
        return None
    return resumen.get((ver, fil, n, k, h, fase))


# ---------- 3. Aceleraciones (nunca con ejecuciones incompletas) ----------
CONFIGS = [("media", 4000, 3), ("nitidez", 4000, 3), ("laplaciano", 4000, 3),
           ("media", 2000, 9), ("media", 4000, 9), ("media", 6000, 9), ("media", 4000, 21)]
ETQ = {c: f"{c[0].capitalize()} {c[2]}×{c[2]} · {c[1]}²" for c in CONFIGS}


def div(a, b):
    return a / b if a is not None and b else None


acel = []
with open(f"{R}/aceleraciones_p02.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filtro", "resolucion", "k",
                "cpp_ms", "cuda_kernel_ms", "cuda_total_ms", "python_ms", "pycuda_kernel_ms", "pycuda_total_ms",
                "cuda_vs_cpp_kernel", "cuda_vs_cpp_con_transferencias",
                "pycuda_vs_python_kernel", "pycuda_vs_python_con_transferencias",
                "pycuda_entre_cuda_kernel", "pycuda_entre_cuda_total", "python_entre_cpp"])
    fmt = lambda x: "" if x is None else f"{x:.4f}"
    for fil, n, k in CONFIGS:
        cpp = med("cpp", fil, n, k, "total")
        ck, ct = med("cuda", fil, n, k, "kernel"), med("cuda", fil, n, k, "total")
        py = med("python", fil, n, k, "total")
        pk, pt = med("pycuda", fil, n, k, "kernel"), med("pycuda", fil, n, k, "total")
        fila = [cpp, ck, ct, py, pk, pt, div(cpp, ck), div(cpp, ct), div(py, pk), div(py, pt),
                div(pk, ck), div(pt, ct), div(py, cpp)]
        acel.append(((fil, n, k), fila))
        w.writerow([fil, f"{n}x{n}", k] + [fmt(x) for x in fila])

# Tabla en pantalla
print("\nAceleraciones (medianas):")
print(f"{'Configuración':24s} {'CUDA/C++ kern':>14s} {'CUDA/C++ total':>15s} {'PyC/Py kern':>13s} {'PyC/Py total':>13s}")
for c, fila in acel:
    s = lambda x: "no completada" if x is None else f"{x:,.1f}×"
    print(f"{ETQ[c]:24s} {s(fila[6]):>14s} {s(fila[7]):>15s} {s(fila[8]):>13s} {s(fila[9]):>13s}")


# ---------- 4. Gráficas ----------
def estilo(ax, titulo, xlabel, ylabel, log=True):
    ax.set_title(titulo, fontsize=13, loc="left")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


SERIES = [("cpp", "total", "-", "C/C++ secuencial"), ("python", "total", "-", "Python secuencial"),
          ("cuda", "kernel", "--", "CUDA C · kernel"), ("cuda", "total", "-", "CUDA C · con transferencias"),
          ("pycuda", "kernel", "--", "PyCUDA · kernel"), ("pycuda", "total", "-", "PyCUDA · con transferencias")]


def grafica_lineas(nombre, titulo, xs, xlabel, xticks, puntos):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for ver, fase, ls, etq in SERIES:
        ys = [med(ver, *p, fase) for p in puntos]
        xv = [x for x, y in zip(xs, ys) if y is not None]
        yv = [y for y in ys if y is not None]
        if yv:
            ax.plot(xv, yv, ls, color=COLOR[ver], linewidth=2, marker="s" if ver == "pycuda" else "o",
                    markersize=9 if ver == "pycuda" else 6, markerfacecolor="none" if ver == "pycuda" else None, label=etq)
    estilo(ax, titulo, xlabel, "Tiempo (ms, escala logarítmica)")
    ax.set_xticks(xs); ax.set_xticklabels(xticks)
    ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout(); fig.savefig(f"{G}/{nombre}", dpi=160); plt.close(fig)


# Exp 1: filtros (barras agrupadas)
fig, ax = plt.subplots(figsize=(11, 5.2))
filtros = ["media", "nitidez", "laplaciano"]
ancho_b = 0.13
for i, (ver, fase, ls, etq) in enumerate(SERIES):
    ys = [med(ver, f, 4000, 3, fase) or 0 for f in filtros]
    xs = [j + (i - 2.5) * (ancho_b + 0.01) for j in range(3)]
    ax.bar(xs, ys, ancho_b, color=COLOR[ver], alpha=1.0 if fase == "total" else 0.55,
           hatch="//" if fase == "kernel" else None, edgecolor="white", linewidth=0.8, label=etq)
estilo(ax, "Experimento 1 · Filtros 3×3 en 4000×4000", "Filtro", "Tiempo (ms, escala logarítmica)")
ax.set_xticks(range(3)); ax.set_xticklabels(["Media", "Nitidez", "Laplaciano"])
ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(f"{G}/g1_filtros.png", dpi=160); plt.close(fig)

# Exp 2 y 3
grafica_lineas("g2_resolucion.png", "Experimento 2 · Resolución (media 9×9)", [4, 16, 36],
               "Píxeles (millones)", ["2000² (4 M)", "4000² (16 M)", "6000² (36 M)"],
               [("media", 2000, 9), ("media", 4000, 9), ("media", 6000, 9)])
grafica_lineas("g3_mascara.png", "Experimento 3 · Tamaño de máscara (media, 4000×4000)", [9, 81, 441],
               "Coeficientes por píxel (k×k)", ["3×3 (9)", "9×9 (81)", "21×21 (441)"],
               [("media", 4000, 3), ("media", 4000, 9), ("media", 4000, 21)])

# Exp 4: hilos (lineal: diferencias pequeñas)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6))
hilos = [128, 256, 512]
for ax, fase, tit in ((axs[0], "kernel", "Kernel"), (axs[1], "total", "Con transferencias")):
    for i, ver in enumerate(("cuda", "pycuda")):
        ys = [med(ver, "media", 4000, 9, fase, h) or 0 for h in hilos]
        xs = [j + (i - 0.5) * 0.36 for j in range(3)]
        ax.bar(xs, ys, 0.34, color=COLOR[ver], edgecolor="white", linewidth=0.8, label=NOMBRE[ver])
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    estilo(ax, tit, "Hilos por bloque", "Tiempo (ms)", log=False)
    ax.set_xticks(range(3)); ax.set_xticklabels([str(h) for h in hilos])
axs[0].legend(fontsize=9, frameon=False)
fig.suptitle("Experimento 4 · Hilos por bloque (media 9×9, 4000×4000)", fontsize=13, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{G}/g4_hilos.png", dpi=160); plt.close(fig)

# Desglose del tiempo GPU (CPU→GPU, kernel, GPU→CPU) por configuración
fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
etqs = [ETQ[c] for c in CONFIGS]
for ax, ver in zip(axs, ("cuda", "pycuda")):
    izq = [0] * len(CONFIGS)
    for fase, col, nom in (("h2d", "#9ec5f4", "CPU→GPU"), ("kernel", COLOR[ver], "Kernel"), ("d2h", "#c3c2b7", "GPU→CPU")):
        vals = [med(ver, f, n, k, fase) or 0 for f, n, k in CONFIGS]
        ax.barh(etqs, vals, left=izq, color=col, edgecolor="white", linewidth=1, label=nom)
        izq = [a + b for a, b in zip(izq, vals)]
    ax.set_title(NOMBRE[ver], fontsize=12, loc="left")
    ax.set_xlabel("Tiempo (ms)")
    ax.grid(axis="x", color="#e5e4df"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.invert_yaxis() if ver == "cuda" else None
axs[0].legend(fontsize=9, frameon=False, loc="lower right")
fig.suptitle("Desglose del tiempo en GPU: transferencias frente a kernel", fontsize=13, x=0.02, ha="left")
fig.tight_layout(); fig.savefig(f"{G}/g5_transferencias.png", dpi=160); plt.close(fig)

# Aceleraciones
fig, ax = plt.subplots(figsize=(12.5, 5.6))
nombres_a = [("CUDA C vs C/C++ · kernel", 6, "cuda", "//"), ("CUDA C vs C/C++ · con transf.", 7, "cuda", None),
             ("PyCUDA vs Python · kernel", 8, "pycuda", "//"), ("PyCUDA vs Python · con transf.", 9, "pycuda", None)]
for i, (etq, idx, ver, hatch) in enumerate(nombres_a):
    ys = [fila[idx] or 0 for _, fila in acel]
    xs = [j + (i - 1.5) * 0.2 for j in range(len(acel))]
    ax.bar(xs, ys, 0.19, color=COLOR[ver], alpha=0.55 if hatch else 1.0, hatch=hatch,
           edgecolor="white", linewidth=0.8, label=etq)
estilo(ax, "Aceleraciones (veces más rápido, escala logarítmica)", "", "Aceleración (×)")
ax.set_xticks(range(len(acel))); ax.set_xticklabels([ETQ[c] for c, _ in acel], rotation=20, ha="right", fontsize=9)
ax.legend(fontsize=9, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(f"{G}/g6_aceleraciones.png", dpi=160); plt.close(fig)

# PyCUDA: inicialización y compilación frente al trabajo en GPU
ini = [d["tiempo"] for d in datos if d["version"] == "pycuda" and d["fase"] == "init"]
com = [d["tiempo"] for d in datos if d["version"] == "pycuda" and d["fase"] == "compilacion"]
if ini and com:
    k9 = med("pycuda", "media", 4000, 9, "kernel"); t9 = med("pycuda", "media", 4000, 9, "total")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    etq = ["Inicialización\ndel contexto", "Compilación\n(SourceModule)", "Total GPU\n(media 9×9 · 4000²)", "Kernel\n(media 9×9 · 4000²)"]
    vals = [statistics.median(ini), statistics.median(com), t9 or 0, k9 or 0]
    ax.bar(etq, vals, 0.55, color=["#c3c2b7", "#c3c2b7", COLOR["pycuda"], COLOR["pycuda"]], edgecolor="white")
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:.1f} ms" if v >= 10 else f"{v:.2f} ms", ha="center", va="bottom", fontsize=9)
    estilo(ax, f"PyCUDA · costos fijos por ejecución (mediana de {len(ini)} ejecuciones)", "", "Tiempo (ms, escala logarítmica)")
    fig.tight_layout(); fig.savefig(f"{G}/g7_init_compilacion.png", dpi=160); plt.close(fig)
    print(f"\nPyCUDA init: mediana {statistics.median(ini):.1f} ms (mín {min(ini):.1f}, máx {max(ini):.1f}, n={len(ini)})")
    print(f"PyCUDA compilación: mediana {statistics.median(com):.1f} ms (mín {min(com):.1f}, máx {max(com):.1f}, n={len(com)})")

if incompletas:
    print("\nNo completadas (excluidas de medianas y aceleraciones):", sorted(incompletas))
print(f"\nListo: {R}/resultados_p02.csv, resumen_p02.csv, aceleraciones_p02.csv y gráficas en {G}/")
