# Práctica 02 · Convolución de imágenes con Python, PyCUDA y CUDA C

Universidad Politécnica Salesiana                      · Computación Paralela

Docente: Ing. Gabriel León

Integrantes: Erika Contreras, Jorge Pizarro


Se compara la misma convolución 2D (filtros de media k×k, nitidez 3×3 y laplaciano 3×3) en cuatro versiones:

| Versión | Archivo | Dónde corre |
|---|---|---|
| C/C++ secuencial (referencia) | `../CPU/conv_cpu.cpp` (práctica 01) | 1 núcleo de CPU |
| CUDA C | `../CUDA/conv_cuda.cu` (práctica 01) | GPU, compilado con nvcc |
| Python secuencial | `Python/conv_python.py` | 1 núcleo de CPU, Python puro |
| PyCUDA | `PyCuda/conv_pycuda.py` + `PyCuda/kernel_convolucion.cu` | GPU, mismo kernel compilado con `SourceModule` |


## Estructura

```
Practica02/
├── comun.py               # lectura/escala de grises, máscaras, guardado .f32/.pgm, CSV (equivale a common.h)
├── Python/conv_python.py  # convolución secuencial: 4 bucles explícitos, acumulación float32
├── PyCuda/
│   ├── kernel_convolucion.cu   # kernel de la práctica 01, sin cambios
│   └── conv_pycuda.py          # SourceModule, mem_alloc, memcpy, eventos CUDA
├── comparar.py            # validación numérica contra C/C++ (error máximo y medio)
├── ejecutar_p02.sh        # compila y corre las 4 versiones en una sola sesión
├── analisis_p02.py        # CSV unificado, mediana/IQR, aceleraciones y gráficas
├── evidencia_visual.py    # figura: original + 3 filtros + recorte
├── requirements.txt       # dependencias de Python con versiones
├── equipo_p02.txt         # hardware y versiones usadas
├── resultados/            # CSV, validacion.txt, pruebas.txt, log de la sesión
├── graficas/              # g1 a g7
└── capturas/              # evidencia visual
```

## Dependencias (versiones usadas)

| Herramienta | Versión |
|---|---|
| Ubuntu (Kubuntu) | 24.04.4 LTS |
| Controlador NVIDIA | 595.99.02 |
| CUDA Toolkit (nvcc) | 13.2 |
| g++ | 13.3.0 |
| Python | 3.12.3 |
| NumPy · Pillow · PyCUDA · matplotlib | ver `requirements.txt` (NumPy 2.5.4, Pillow 13.0.0, PyCUDA 2026.1) |

GPU usada: NVIDIA GeForce RTX 5080 Laptop GPU (capacidad de cómputo 12.0, 16 GB). Detalle completo en `equipo_p02.txt`.

## Instalación

```bash
python3 -m venv ~/venvs/cuda
source ~/venvs/cuda/bin/activate
pip install -r Practica02/requirements.txt
# comprobar que PyCUDA ve la GPU
python3 -c "import pycuda.driver as d; d.init(); print(d.Device(0).name())"
```

PyCUDA necesita el CUDA Toolkit instalado (`nvcc` en el PATH) para compilar el kernel.

## Datos de entrada

Se usan **las mismas imágenes de la práctica 01**: `CPU/imgCPU/img_2000.pgm`, `img_4000.pgm` e `img_6000.pgm`.
No se suben por su tamaño. Se generan con `CPU/preparar_imagenes.py` a partir de la imagen original, como explica el README principal (sección *Datos de entrada*).
La imagen de 1003×997 para la prueba de límites la crea `ejecutar_p02.sh` (recorte de `img_2000.pgm`).

## Ejecución

Todo desde `Practica02/`, con el entorno virtual activado.

**Pruebas pequeñas (matriz 5×5, laplaciano; esperado −60, 8, −1):**

```bash
python3 Python/conv_python.py --test
python3 PyCuda/conv_pycuda.py --test
```

**Una configuración suelta:**

```bash
# Python: <imagen> <filtro> <k> <repeticiones> <prefijo_salida> <csv> [limite_s]
python3 Python/conv_python.py ../CPU/imgCPU/img_2000.pgm media 3 5 salidas/py_media3_2000 resultados/prueba.csv 600
# PyCUDA: <imagen> <filtro> <k> <hilos_bloque> <repeticiones> <prefijo_salida> <csv>
python3 PyCuda/conv_pycuda.py ../CPU/imgCPU/img_4000.pgm media 9 256 5 salidas/pycuda_media9_4000 resultados/prueba.csv
```

**Todos los experimentos (≈ 1 h 45 min; Python secuencial va al final):**

```bash
powerprofilesctl set performance        # recomendado: cargador conectado y modo rendimiento
bash ejecutar_p02.sh 2>&1 | tee resultados/log_p02.txt
python3 analisis_p02.py                 # medianas, IQR, aceleraciones y gráficas
python3 evidencia_visual.py             # capturas/05_evidencia_visual_pycuda.png
```

`ejecutar_p02.sh` compila C/C++ y CUDA C (`g++ -O2 -std=c++17`, `nvcc -O2 -std=c++17 -arch=native`), corre las 4 pruebas 5×5, la prueba de límites con 128, 256 y 512 hilos, los 4 experimentos y la validación.

| Experimento | Configuración | Versiones |
|---|---|---|
| Filtros | Media, nitidez y laplaciano 3×3 · 4000² | Las cuatro |
| Resolución | Media 9×9 · 2000², 4000², 6000² | Las cuatro |
| Máscara | Media 3×3, 9×9, 21×21 · 4000² | Las cuatro |
| Hilos por bloque | Media 9×9 · 4000² · 128, 256, 512 | CUDA C y PyCUDA |

## Pruebas y validación

- `resultados/pruebas.txt`: prueba 5×5 de las cuatro versiones.
- `resultados/validacion.txt`: error absoluto máximo y medio de Python, CUDA C y PyCUDA contra C/C++ (valores float32 antes de recortar a 0–255), tolerancia 1·10⁻². Resultado: 33 comparaciones OK; Python error 0, CUDA C y PyCUDA como máximo 1,678·10⁻⁴.

## CSV

`resultados/resultados_p02.csv` reúne todas las mediciones (448 registros) con las columnas:

```
version, filtro, ancho, alto, k, hilos_bloque, bloques, repeticion, fase, tiempo, unidad, estado
```

- `version`: `cpp`, `cuda`, `python` o `pycuda`.
- `fase`: `total` (CPU); `h2d`, `kernel`, `d2h`, `total` (GPU); `init` y `compilacion` (PyCUDA, una vez por ejecución).
- `estado`: `completada` o `no_completada` (Python con límite de 1 h por repetición; en esta sesión todas se completaron).

Derivados: `resumen_p02.csv` (mediana, Q1, Q3, IQR), `aceleraciones_p02.csv`, `viabilidad_python.csv` (prueba previa de 2000²).

## Resultados principales (16/11/2026, medianas)

| Configuración | C/C++ | Python | Kernel CUDA C | Kernel PyCUDA | Total CUDA C | Total PyCUDA |
|---|---|---|---|---|---|---|
| Media 3×3 · 4000² | 59,50 ms | 16,16 s | 0,341 ms | 0,344 ms | 9,154 ms | 10,487 ms |
| Media 9×9 · 4000² | 552,37 ms | 119,48 s | 1,112 ms | 1,112 ms | 9,697 ms | 11,559 ms |
| Media 21×21 · 4000² | 2672,26 ms | 714,28 s | 6,990 ms | 7,220 ms | 15,669 ms | 18,856 ms |

- PyCUDA ejecuta el mismo kernel y tarda lo mismo que CUDA C en la GPU; con copias tarda 12–20 % más, y además paga ≈ 195 ms de inicialización y ≈ 186 ms de compilación por ejecución.
- Python secuencial es 214–272 veces más lento que C/C++ con el mismo cálculo.
