"""analisis_p03.py — resumen, tablas y gráficas de los experimentos de la Práctica 03.

Lee resultados/exp1_*.csv, exp2_resolucion.csv, exp3_filtros.csv y exp4_concurrencia.csv.
- Solo usa solicitudes completadas y repeticiones >= 1 (la repetición 0 es el calentamiento).
- Reporta MEDIANA y variabilidad (Q1, Q3, rango intercuartílico) por configuración y fase.

"""
import csv
import glob
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, G = "resultados", "graficas"
os.makedirs(G, exist_ok=True)

# Paleta categórica (orden fijo) y tinta de texto
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TINTA, TINTA2, REJILLA, FONDO = "#0b0b0b", "#52514e", "#e5e4df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA2,
                     "xtick.color": TINTA2, "ytick.color": TINTA2, "figure.facecolor": FONDO,
                     "axes.facecolor": FONDO, "savefig.facecolor": FONDO})


# ---------- 1. Leer y unir ----------
datos = []
for ruta in sorted(glob.glob(f"{R}/exp*.csv")):
    exp = os.path.basename(ruta).split("_")[0]                     # exp1, exp2, ...
    with open(ruta) as f:
        for r in csv.DictReader(f):
            r["experimento"] = exp
            r["tiempo"] = float(r["tiempo"])
            for c in ("ancho", "alto", "k", "repeticion", "concurrencia"):
                r[c] = int(r[c]) if r[c] not in ("", None) else 0
            datos.append(r)
if not datos:
    raise SystemExit("No hay CSV de experimentos en resultados/. Ejecute primero ejecutar_p03.sh")

cols = ["experimento"] + [c for c in datos[0] if c != "experimento"]
with open(f"{R}/datos_p03.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(datos)
print(f"datos_p03.csv: {len(datos)} registros")

incompletas = [d for d in datos if d["estado"] != "completada"]
if incompletas:
    print(f"Aviso: {len(incompletas)} registros no completados (excluidos de las medianas):")
    for d in incompletas[:10]:
        print("  ", d["experimento"], d["filtro"], d["ancho"], d["k"], d["repeticion"], d["estado"])


# ---------- 2. Mediana y variabilidad ----------
def clave(d):
    return (d["experimento"], d["modo"], d["entorno"], d["filtro"], d["ancho"], d["k"], d["concurrencia"], d["fase"])


grupos = defaultdict(list)
unidades = {}
for d in datos:
    if d["estado"] == "completada" and d["repeticion"] >= 1:
        grupos[clave(d)].append(d["tiempo"])
        unidades[clave(d)] = d["unidad"]


def cuartiles(v):
    if len(v) < 2:
        return v[0], v[0], v[0]
    q1, q2, q3 = statistics.quantiles(v, n=4, method="inclusive")
    return q1, statistics.median(v), q3


RES = {}
with open(f"{R}/resumen_p03.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experimento", "modo", "entorno", "filtro", "resolucion", "k", "concurrencia", "fase",
                "n", "mediana", "q1", "q3", "iqr", "unidad"])
    for c in sorted(grupos):
        q1, med, q3 = cuartiles(grupos[c])
        RES[c] = (med, q1, q3, len(grupos[c]))
        exp, modo, ent, fil, n, k, conc, fase = c
        w.writerow([exp, modo, ent, fil, f"{n}x{n}", k, conc, fase, len(grupos[c]),
                    f"{med:.4f}", f"{q1:.4f}", f"{q3:.4f}", f"{q3 - q1:.4f}", unidades[c]])
print(f"resumen_p03.csv: {len(RES)} grupos (mediana, Q1, Q3, IQR)")


def m(exp, fase, modo="servicio", entorno=None, filtro="media", n=4000, k=9, conc=1):
    for c, v in RES.items():
        if c[0] == exp and c[1] == modo and (entorno is None or c[2] == entorno) and c[3] == filtro \
                and c[4] == n and c[5] == k and c[6] == conc and c[7] == fase:
            return v
    return None


def med(*a, **kw):
    v = m(*a, **kw)
    return v[0] if v else None


# ---------- 3. Arranque en frío (aparte) ----------
with open(f"{R}/arranque_p03.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experimento", "modo", "entorno", "filtro", "resolucion", "k", "registro", "fase", "tiempo_ms"])
    for d in datos:
        if d["repeticion"] == 0 and d["concurrencia"] in (0, 1) and d["fase"] in (
                "init", "compilacion", "calentamiento", "kernel", "procesamiento", "procesamiento_gpu",
                "servidor_total", "cliente_total"):
            w.writerow([d["experimento"], d["modo"], d["entorno"], d["filtro"], f"{d['ancho']}x{d['alto']}",
                        d["k"], d["id_solicitud"], d["fase"], f"{d['tiempo']:.4f}"])
print("arranque_p03.csv: contexto CUDA, compilación y primera ejecución de cada configuración (repetición 0)")


# ---------- 4. Tablas en pantalla ----------
def f(x, dec=3):
    return "—" if x is None else f"{x:,.{dec}f}"


print("\nExp. 1 · Integración y contenedor (media 9x9, 4000x4000x3; mediana en ms)")
print(f"{'Modo':34s} {'kernel':>9s} {'CPU↔GPU+kernel':>15s} {'motor GPU':>10s} {'servidor':>10s} {'cliente':>10s}")
EXP1 = [("PyCUDA directo · anfitrión", "directo", "anfitrion"),
        ("PyCUDA directo · contenedor", "directo", "contenedor"),
        ("Servicio contenerizado", "servicio", "contenedor")]
for nom, modo, ent in EXP1:
    proc = "procesamiento" if modo == "directo" else "procesamiento_gpu"
    print(f"{nom:34s} {f(med('exp1', 'kernel', modo, ent)):>9s} {f(med('exp1', 'total_gpu', modo, ent)):>15s} "
          f"{f(med('exp1', proc, modo, ent), 1):>10s} {f(med('exp1', 'servidor_total', modo, ent), 1):>10s} "
          f"{f(med('exp1', 'cliente_total', modo, ent), 1):>10s}")

print("\nExp. 2 · Resolución (media 9x9; mediana en ms)")
RESOL = [2000, 4000, 6000]
for n in RESOL:
    print(f"{n}x{n}x3  kernel {f(med('exp2', 'kernel', n=n)):>7s} | servidor {f(med('exp2', 'servidor_total', n=n), 1):>8s}"
          f" | cliente {f(med('exp2', 'cliente_total', n=n), 1):>8s}")

print("\nExp. 3 · Filtros y máscara (4000x4000x3; mediana en ms)")
FILT = [("media", 3, "Media 3×3"), ("media", 9, "Media 9×9"), ("media", 21, "Media 21×21"),
        ("nitidez", 3, "Nitidez 3×3"), ("laplaciano", 3, "Laplaciano 3×3")]
for fil, k, nom in FILT:
    print(f"{nom:15s} kernel {f(med('exp3', 'kernel', filtro=fil, k=k)):>7s} | motor GPU "
          f"{f(med('exp3', 'procesamiento_gpu', filtro=fil, k=k), 1):>6s} | salida PNG+.f32 "
          f"{f(med('exp3', 'salida', filtro=fil, k=k), 1):>7s} | servidor {f(med('exp3', 'servidor_total', filtro=fil, k=k), 1):>7s}")

print("\nExp. 4 · Concurrencia (media 9x9, 4000x4000x3; 5 rondas)")
for c in (1, 2):
    lat = [d["tiempo"] for d in datos if d["experimento"] == "exp4" and d["concurrencia"] == c and d["modo"] == "servicio"
           and d["fase"] == "cliente_total" and d["estado"] == "completada" and d["repeticion"] >= 1]
    esp = [d["tiempo"] for d in datos if d["experimento"] == "exp4" and d["concurrencia"] == c and d["modo"] == "servicio"
           and d["fase"] == "espera_cola" and d["estado"] == "completada" and d["repeticion"] >= 1]
    fallos = sum(1 for d in datos if d["experimento"] == "exp4" and d["concurrencia"] == c and d["modo"] == "servicio"
                 and d["fase"] == "cliente_total" and d["estado"] != "completada")
    rend = med("exp4", "rendimiento", conc=c)
    if lat:
        q1, md, q3 = cuartiles(lat)
        print(f"{c} cliente(s): latencia mediana {md:.0f} ms (Q1 {q1:.0f}, Q3 {q3:.0f}, n={len(lat)}) | "
              f"espera en cola máx {max(esp):.1f} ms | rendimiento {f(rend, 2)} solicitudes/s | fallos {fallos}")

# Exp. 4b: cliente en OTRO equipo (red local), si existe
REMOTO = any(d["modo"] == "servicio_red" for d in datos)
if REMOTO:
    print("\nExp. 1c/4 desde OTRO equipo (red local) frente a localhost (media 9x9, 4000x4000x3)")
    for exp, nom in (("exp1", "Solicitud individual (exp1)"), ("exp4", "Concurrencia (exp4)")):
        for c in ((1,) if exp == "exp1" else (1, 2)):
            for modo, lugar in (("servicio", "localhost  "), ("servicio_red", "otro equipo")):
                cli, srv = med(exp, "cliente_total", modo, conc=c), med(exp, "servidor_total", modo, conc=c)
                rend = med(exp, "rendimiento", modo, conc=c) if exp == "exp4" else None
                if cli is not None:
                    print(f"  {nom:28s} {c} cliente(s) {lugar} | cliente {cli:8.1f} ms | servidor {f(srv, 1):>7s} ms | "
                          f"comunicación y resto {cli - (srv or 0):8.1f} ms"
                          + (f" | {rend:.2f} solicitudes/s" if rend else ""))

# Diferencia entre el motor completo y la suma de sus fases medidas
print("\nMotor GPU: tiempo no atribuido = procesamiento − (preparación + reserva + CPU↔GPU+kernel + liberación)")
for exp, nom, kw in [("exp1", "directo anfitrión", dict(modo="directo", entorno="anfitrion")),
                     ("exp1", "directo contenedor", dict(modo="directo", entorno="contenedor")),
                     ("exp1", "servicio", dict(modo="servicio")),
                     ("exp2", "servicio 2000", dict(n=2000)), ("exp2", "servicio 6000", dict(n=6000))]:
    proc = "procesamiento" if kw.get("modo") == "directo" else "procesamiento_gpu"
    partes = [med(exp, ph, **kw) for ph in ("preparacion", "reserva", "total_gpu", "liberacion")]
    total = med(exp, proc, **kw)
    if total is not None and None not in partes:
        print(f"  {nom:20s} motor {total:8.2f} ms | suma de fases {sum(partes):8.2f} ms | "
              f"sin atribuir {total - sum(partes):6.2f} ms")


# ---------- 5. Gráficas ----------
def estilo(ax, titulo, xlabel="", ylabel="", log=False):
    ax.set_title(titulo, fontsize=12, loc="left", color=TINTA)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.grid(axis="y", color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def barras(ax, x, vals, ancho, color, etiqueta, rango=None, fmt="{:.1f}"):
    xs, ys = [a for a, v in zip(x, vals) if v is not None], [v for v in vals if v is not None]
    ax.bar(xs, ys, ancho, color=color, edgecolor=FONDO, linewidth=2, label=etiqueta)
    if rango:
        for a, (q1, q3) in zip(x, rango):
            if q1 is not None:
                ax.plot([a, a], [q1, q3], color=TINTA, linewidth=1)
    for a, v in zip(xs, ys):
        ax.text(a, v * 1.08, fmt.format(v), ha="center", va="bottom", fontsize=8, color=TINTA2)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(f"{G}/{nombre}", dpi=160)
    plt.close(fig)
    print(f"  {G}/{nombre}")


print("\nGráficas:")

# g1: integración y contenedor
fig, ax = plt.subplots(figsize=(11, 5))
etapas = [("kernel", "Kernel"), ("total_gpu", "CPU↔GPU + kernel"), ("proc", "Motor GPU completo"),
          ("servidor_total", "Servidor total"), ("cliente_total", "Cliente total")]
for i, ((nom, modo, ent), col) in enumerate(zip(EXP1, (AZUL, NARANJA, AQUA))):
    vals, rng = [], []
    for fase, _ in etapas:
        fase = ("procesamiento" if modo == "directo" else "procesamiento_gpu") if fase == "proc" else fase
        v = m("exp1", fase, modo, ent)
        vals.append(v[0] if v else None)
        rng.append((v[1], v[2]) if v else (None, None))
    x = [j + (i - 1) * 0.27 for j in range(len(etapas))]
    barras(ax, x, vals, 0.25, col, nom, rng)
estilo(ax, "Exp. 1 · Mismo cálculo directo y como servicio (media 9×9, 4000×4000 color)",
       ylabel="Tiempo (ms, mediana; escala log)", log=True)
ax.set_xticks(range(len(etapas)))
ax.set_xticklabels([e[1] for e in etapas])
ax.legend(frameon=False, fontsize=9, loc="upper left")
guardar(fig, "g1_integracion.png")

# g2: resolución
fig, ax = plt.subplots(figsize=(10, 5))
mp = [3 * n * n / 1e6 for n in RESOL]
for fase, nom, col, ls in (("kernel", "Kernel", AZUL, "-"), ("total_gpu", "CPU↔GPU + kernel", NARANJA, "-"),
                           ("servidor_total", "Servidor total", AQUA, "-"), ("cliente_total", "Cliente total", AMARILLO, "-")):
    ys = [med("exp2", fase, n=n) for n in RESOL]
    ax.plot(mp, ys, ls, color=col, linewidth=2, marker="o", markersize=8, label=nom)
    ax.text(mp[-1] * 1.03, ys[-1], f"{nom}  {ys[-1]:.1f} ms", va="center", fontsize=9, color=TINTA2)
estilo(ax, "Exp. 2 · Resolución (media 9×9, color)", "Valores procesados (millones = 3 canales × ancho × alto)",
       "Tiempo (ms, mediana; escala log)", log=True)
ax.set_xticks(mp)
ax.set_xticklabels([f"{n}²\n({v:.0f} M)" for n, v in zip(RESOL, mp)])
ax.set_xlim(0, mp[-1] * 1.45)
ax.legend(frameon=False, fontsize=9, loc="upper left")
guardar(fig, "g2_resolucion.png")

# g3: filtros y máscara
fig, ax = plt.subplots(figsize=(11, 5))
for i, (fase, nom, col) in enumerate((("kernel", "Kernel", AZUL), ("total_gpu", "CPU↔GPU + kernel", NARANJA),
                                      ("salida", "Salida (PNG + .f32)", MAGENTA),
                                      ("servidor_total", "Servidor total", AQUA))):
    vals = [med("exp3", fase, filtro=fil, k=k) for fil, k, _ in FILT]
    x = [j + (i - 1.5) * 0.2 for j in range(len(FILT))]
    barras(ax, x, vals, 0.18, col, nom)
estilo(ax, "Exp. 3 · Filtro y tamaño de máscara (4000×4000 color)", ylabel="Tiempo (ms, mediana; escala log)", log=True)
ax.set_xticks(range(len(FILT)))
ax.set_xticklabels([n for _, _, n in FILT])
ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=4)
ax.set_ylim(top=ax.get_ylim()[1] * 4)
guardar(fig, "g3_filtros.png")

# g4: desglose del tiempo del cliente (exp2)
fig, ax = plt.subplots(figsize=(11, 4.2))
SEG = [("recepcion", "Recepción", AZUL), ("decodificacion", "Decodificación", NARANJA),
       ("espera_cola", "Espera en cola", AMARILLO), ("procesamiento_gpu", "Motor GPU", AQUA),
       ("salida", "Salida PNG + .f32", MAGENTA)]
etq = [f"{n}×{n}" for n in RESOL]
izq = [0.0] * len(RESOL)
for fase, nom, col in SEG:
    vals = [med("exp2", fase, n=n) or 0 for n in RESOL]
    ax.barh(etq, vals, left=izq, color=col, edgecolor=FONDO, linewidth=2, label=nom)
    izq = [a + b for a, b in zip(izq, vals)]
com = [max((med("exp2", "cliente_total", n=n) or 0) - a, 0) for n, a in zip(RESOL, izq)]
ax.barh(etq, com, left=izq, color="#c3c2b7", edgecolor=FONDO, linewidth=2, label="Comunicación y resto (cliente − servidor)")
for y, n in enumerate(RESOL):
    tot = med("exp2", "cliente_total", n=n) or 0
    ax.text(tot * 1.01, y, f"{tot:.0f} ms", va="center", fontsize=9, color=TINTA2)
ax.set_title("Exp. 2 · ¿En qué se va el tiempo de una solicitud? (media 9×9, medianas)", fontsize=12, loc="left")
ax.set_xlabel("Tiempo (ms)")
ax.grid(axis="x", color=REJILLA)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.invert_yaxis()
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
guardar(fig, "g4_desglose.png")

# g5: concurrencia (dos medidas distintas -> dos paneles)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
SERIES4 = [("servicio", 1, AZUL, "1 cliente · localhost"), ("servicio", 2, NARANJA, "2 clientes · localhost")]
if REMOTO:
    SERIES4 += [("servicio_red", 1, AQUA, "1 cliente · otro equipo"), ("servicio_red", 2, AMARILLO, "2 clientes · otro equipo")]
for i, (modo, c, col, nom) in enumerate(SERIES4):
    pts = [(d["repeticion"], d["tiempo"]) for d in datos if d["experimento"] == "exp4" and d["concurrencia"] == c
           and d["modo"] == modo and d["fase"] == "cliente_total" and d["estado"] == "completada" and d["repeticion"] >= 1]
    off = (i - (len(SERIES4) - 1) / 2) * 0.12
    a1.scatter([p[0] + off for p in pts], [p[1] for p in pts], s=60, color=col, edgecolor=FONDO, linewidth=2,
               label=nom, zorder=3)
estilo(a1, "Latencia de cada solicitud", "Ronda", "Tiempo en el cliente (ms)")
a1.set_xticks(range(1, 6))
a1.legend(frameon=False, fontsize=9)
rend = [med("exp4", "rendimiento", modo, conc=c) or 0 for modo, c, _, _ in SERIES4]
a2.bar([n.replace(" · ", "\n") for _, _, _, n in SERIES4], rend, 0.5, color=[s4[2] for s4 in SERIES4],
       edgecolor=FONDO, linewidth=2)
for x, v in enumerate(rend):
    a2.text(x, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=TINTA2)
estilo(a2, "Rendimiento (mediana de 5 rondas)", ylabel="Solicitudes exitosas por segundo")
fig.suptitle("Exp. 4 · Concurrencia (media 9×9, 4000×4000 color)",
             fontsize=12, x=0.01, ha="left")
guardar(fig, "g5_concurrencia.png")

print(f"\nListo: {R}/datos_p03.csv, resumen_p03.csv, arranque_p03.csv y gráficas en {G}/")