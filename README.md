

**Asignatura:** Computación Paralela · Universidad Politécnica Salesiana

**Integrantes:** Erika Contreras -  Jorge Pizarro

> **Práctica 02 (Python secuencial y PyCUDA):** ver [Practica02/README02.md](Practica02/README02.md).
> La práctica 01 (C/C++ y CUDA C) `CPU/` y `CUDA/`
>
> 
# Práctica 01 · Convolución de imágenes de alta resolución mediante CUDA C
Implementación de tres filtros de convolución (media k×k, nitidez 3×3 y laplaciano 3×3) en una versión secuencial C++ (referencia) y una versión paralela CUDA C, con validación numérica CPU–GPU y medición del efecto de la resolución, el tamaño de máscara y la configuración de hilos por bloque.

---

## Estructura del repositorio

```
Practica01/
├── README.md
├── .gitignore
├── common.h                 # lectura PGM, máscaras, CSV (compartido por CPU y CUDA)
├── equipo.txt               # registro del equipo y versiones
├── graficas.py              # tablas y gráficas a partir de los CSV
├── graficas/                # resumen.csv, aceleraciones.csv y gráficas PNG
├── CPU/
│   ├── conv_cpu.cpp         # versión secuencial (referencia)
│   ├── preparar_imagenes.py # gris + recorte cuadrado + redimensionamiento
│   ├── resultados.csv       # registros originales de CPU
│   ├── imgCPU/              # img_2000.pgm, img_4000.pgm, img_6000.pgm (no se suben)
│   └── FiltrosAplicados/    # salidas .pgm y .f32 (no se suben)
└── CUDA/
    ├── conv_cuda.cu         # versión CUDA C (un hilo por píxel)
    ├── comparar.cpp         # error absoluto máximo y medio CPU vs GPU
    ├── crear_prueba.py      # imagen 1003×997 (dimensiones no múltiplos)
    ├── resultados.csv       # registros originales de CUDA
    ├── validacion.txt       # resultados de la comparación CPU vs GPU
    └── FiltrosAplicados/    # salidas .pgm y .f32 (no se suben)
```

---

## Equipo y herramientas

| Componente | Detalle |
|---|---|
| Equipo | ASUS ROG Strix (laptop) |
| CPU | Intel Core Ultra 9 275HX · 24 núcleos (24 hilos, 1 hilo por núcleo) · 0,8–5,4 GHz |
| RAM | 30 GiB |
| GPU | NVIDIA GeForce RTX 5080 Laptop GPU · 16 303 MiB (límite de potencia mostrado por `nvidia-smi`: 80 W) |
| Driver NVIDIA | 595.99.02 |
| CUDA Toolkit | 13.2 (V13.2.86) |
| Sistema operativo | Kubuntu (Ubuntu 24.04.4 LTS) · kernel Linux 6.17.0-14-generic |
| Compilador C++ | g++ 13.3.0 |
| Python | Python 3.12.3 · Pillow 10.2.0 · matplotlib 3.6.3 |

Condiciones de medición: cargador conectado y perfil de energía `performance`. Detalle completo en `equipo.txt`. La versión CPU es secuencial: usa 1 de los 24 núcleos.

---

## Dependencias

```bash
sudo apt install build-essential python3-pil python3-matplotlib
```

Además: driver NVIDIA y CUDA Toolkit (probado con CUDA 13.2; la RTX 5080 requiere CUDA ≥ 12.8).

---

## 1. Datos de entrada

La imagen original no se incluye en el repositorio por su tamaño.

- **Imagen original:** _[https://unsplash.com/es/fotos/montanas-nevadas-sobre-green-valley-J71r_nZTv1U_ — 8192×5120 px, descargada en "tamaño original".
- Guardarla como `imagenOriginal/montana.jpg`.

Preparación (escala de grises → recorte cuadrado centrado de 5120×5120 → redimensionamiento Lanczos):

```bash
cd CPU
python3 preparar_imagenes.py ../imagenOriginal/montana.jpg imgCPU/
```

Genera `img_2000.pgm`, `img_4000.pgm` e `img_6000.pgm`. La versión de 6000×6000 se obtiene **ampliando** el recorte de 5120 px (factor ×1,17), porque la imagen original no alcanza 6000 px en su lado menor.

Todas las implementaciones usan exactamente estos mismos archivos.

---

## 2. Compilación

```bash
# CPU
cd CPU
g++ -O2 -std=c++17 -I.. conv_cpu.cpp -o conv_cpu

# CUDA
cd ../CUDA
nvcc -O2 -std=c++17 -arch=native -I.. conv_cuda.cu -o conv_cuda
g++  -O2 -std=c++17 comparar.cpp -o comparar
```

`-I..` indica al compilador que busque `common.h` en la carpeta raíz: CPU y CUDA usan un único archivo con las mismas máscaras y funciones.

---

## 3. Ejecución

```bash
./conv_cpu  <entrada.pgm> <media|nitidez|laplaciano> <k> <reps> <prefijo_salida> <resultados.csv>
./conv_cuda <entrada.pgm> <media|nitidez|laplaciano> <k> <hilos_bloque> <reps> <prefijo_salida> <resultados.csv>
./comparar  <cpu.f32> <gpu.f32> [tolerancia]
```

`k` solo aplica a la media; nitidez y laplaciano son siempre 3×3. Cada ejecución genera `<prefijo>.pgm` (imagen visualizable) y `<prefijo>.f32` (valores crudos en float para validación).

### Experimentos de CPU (desde `CPU/`)

```bash
rm -f resultados.csv
for f in media nitidez laplaciano; do
  ./conv_cpu imgCPU/img_4000.pgm $f 3 5 FiltrosAplicados/cpu_${f}3_4000 resultados.csv
done
for n in 2000 4000 6000; do
  ./conv_cpu imgCPU/img_$n.pgm media 9 5 FiltrosAplicados/cpu_media9_$n resultados.csv
done
./conv_cpu imgCPU/img_4000.pgm media 21 5 FiltrosAplicados/cpu_media21_4000 resultados.csv
```

### Experimentos de CUDA y validación (desde `CUDA/`)

Requiere haber ejecutado antes los experimentos de CPU (usa sus `.f32` como referencia).

```bash
rm -f resultados.csv validacion.txt
IMG=../CPU/imgCPU; REF=../CPU/FiltrosAplicados; OUT=FiltrosAplicados
mkdir -p $OUT

for f in media nitidez laplaciano; do
  ./conv_cuda $IMG/img_4000.pgm $f 3 256 5 $OUT/gpu_${f}3_4000 resultados.csv
  echo "--- $f 3x3 4000 256h ---" >> validacion.txt
  ./comparar $REF/cpu_${f}3_4000.f32 $OUT/gpu_${f}3_4000.f32 >> validacion.txt
done
for n in 2000 4000 6000; do
  ./conv_cuda $IMG/img_$n.pgm media 9 256 5 $OUT/gpu_media9_$n resultados.csv
  echo "--- media 9x9 $n 256h ---" >> validacion.txt
  ./comparar $REF/cpu_media9_$n.f32 $OUT/gpu_media9_$n.f32 >> validacion.txt
done
./conv_cuda $IMG/img_4000.pgm media 21 256 5 $OUT/gpu_media21_4000 resultados.csv
echo "--- media 21x21 4000 256h ---" >> validacion.txt
./comparar $REF/cpu_media21_4000.f32 $OUT/gpu_media21_4000.f32 >> validacion.txt
for h in 128 512; do
  ./conv_cuda $IMG/img_4000.pgm media 9 $h 5 $OUT/gpu_media9_4000_h$h resultados.csv
  echo "--- media 9x9 4000 ${h}h ---" >> validacion.txt
  ./comparar $REF/cpu_media9_4000.f32 $OUT/gpu_media9_4000_h$h.f32 >> validacion.txt
done
```

### Tablas y gráficas (desde `Practica01/`)

```bash
python3 graficas.py CPU/resultados.csv CUDA/resultados.csv graficas/
```

---

## 4. Pruebas de corrección

| Prueba | Comando | Resultado esperado |
|---|---|---|
| Prueba manual CPU (matriz 5×5 de la guía, laplaciano) | `CPU/conv_cpu --test` | (0,0) = −60, (1,1) = 8, (2,2) = −1 → OK |
| Prueba manual GPU (misma matriz, 4 hilos/bloque → 3 hilos sobrantes) | `CUDA/conv_cuda --test` | mismos valores → OK |
| Dimensiones no múltiplos (1003×997 = 999 991 píxeles): 128 hilos → 7813 bloques (73 sobrantes); 256 → 3907 (201 sobrantes); 512 → 1954 (457 sobrantes) | ver abajo | error dentro de tolerancia con los 3 tamaños de bloque |
| Comparación numérica CPU vs GPU (9 configuraciones) | `CUDA/validacion.txt` | todas OK |

```bash
cd CUDA
python3 crear_prueba.py ../CPU/imgCPU/img_2000.pgm prueba_1003x997.pgm
../CPU/conv_cpu prueba_1003x997.pgm media 9 1 FiltrosAplicados/cpu_prueba prueba.csv
for h in 128 256 512; do
  ./conv_cuda prueba_1003x997.pgm media 9 $h 1 FiltrosAplicados/gpu_prueba_h$h prueba.csv
  echo "--- prueba no multiplo 1003x997 media 9x9 ${h}h ---" >> validacion.txt
  ./comparar FiltrosAplicados/cpu_prueba.f32 FiltrosAplicados/gpu_prueba_h$h.f32 >> validacion.txt
done
```

**Tolerancia: 1·10⁻².** La comparación se hace sobre los valores crudos en float, antes de convertir a 0–255. La GPU puede fusionar multiplicación y suma (FMA), lo que cambia el redondeo; la cota de error de una suma de n términos en float es ≈ n·ε·Σ|términos|, que para la peor máscara (21×21, n = 441, píxeles ≤ 255, ε = 1,19·10⁻⁷) da ≈ 1,3·10⁻². Además 10⁻² es mucho menor que 0,5, por lo que la imagen visualizada no cambia.

**Resultado:** nitidez y laplaciano dan error 0 (coeficientes y píxeles enteros: sumas exactas en float); la media da error máximo 1,7·10⁻⁴ y medio ≈ 4·10⁻⁶.

---

## 5. Diseño de la solución

**Máscaras**

| Filtro | Máscara |
|---|---|
| Media k×k | todos los coeficientes = 1/(k·k) |
| Nitidez 3×3 | [0, −1, 0; −1, 5, −1; 0, −1, 0] |
| Laplaciano 3×3 | [0, 1, 0; 1, −4, 1; 0, 1, 0] |

Las tres máscaras son simétricas, por lo que convolución y correlación dan el mismo resultado.

**Distribución del trabajo en CUDA**
- La imagen se guarda como arreglo 1D por filas: `índice = fila·ancho + col`.
- Bloques e hilos en X: `id = blockIdx.x·blockDim.x + threadIdx.x`; `bloques = ⌈N / hilos⌉`.
- Índice global → fila y columna: `fila = id / ancho`, `col = id % ancho`.
- Control de límites: `if (id >= N) return;` descarta los hilos sobrantes del último bloque.
- Extensión con ceros: los vecinos fuera de la imagen no se suman.
- Arreglos separados de entrada y salida, para que ningún hilo lea un valor ya modificado por otro.
- La imagen y la máscara se leen desde memoria global.

**Conversión al rango de visualización:** redondeo y recorte a [0, 255]. Los valores negativos (laplaciano, bordes de la nitidez) quedan en 0.

---

## 6. Metodología de medición

- 1 ejecución de calentamiento no registrada + **5 repeticiones medidas** por configuración.
- Se reporta la **mediana** y, como variabilidad, el **rango intercuartílico (IQR)** y el rango mínimo–máximo (`graficas/resumen.csv`).
- **CPU:** `std::chrono::steady_clock` alrededor de la convolución.
- **CUDA:** eventos CUDA para H→D (imagen + máscara), kernel, D→H y total; `cudaEventSynchronize` asegura que la GPU terminó antes de leer los tiempos.
- **Se excluyen** lectura/escritura de archivos, visualización y preparación de imágenes.
- **La reserva de memoria se hace previamente** (`std::vector` en CPU, `cudaMalloc` en GPU) y no está incluida en los tiempos.
- Memoria del host paginable (`std::vector`, sin memoria fija).
- Un factor a la vez: 256 hilos/bloque fijos excepto en el experimento de hilos. Las configuraciones repetidas entre experimentos (media 3×3 y 9×9 en 4000², 256 hilos) se miden una sola vez.
- Aceleración de cómputo = tiempo CPU / tiempo kernel. Aceleración con transferencias = tiempo CPU / tiempo total GPU.

**Formato del CSV** (`resultados.csv`): `implementacion, filtro, ancho, alto, k, hilos_bloque, bloques, repeticion, fase, tiempo_ms`. Unidades: milisegundos. Fases: `total` (CPU); `h2d`, `kernel`, `d2h`, `total` (CUDA).

---

## 7. Resumen de resultados (medianas)

| Configuración | CPU (ms) | Kernel (ms) | Total GPU (ms) | Acel. cómputo | Acel. con transf. | % transf. |
|---|---|---|---|---|---|---|
| Media 3×3, 4000² | 59,40 | 0,438 | 11,38 | 136× | 5,2× | 98,7 % |
| Nitidez 3×3, 4000² | 59,58 | 0,434 | 11,79 | 137× | 5,1× | 96,3 % |
| Laplaciano 3×3, 4000² | 61,11 | 0,418 | 9,62 | 146× | 6,4× | 96,2 % |
| Media 9×9, 2000² | 139,51 | 0,352 | 2,46 | 397× | 56,8× | 85,6 % |
| Media 9×9, 4000² | 569,03 | 1,624 | 11,49 | 351× | 49,5× | 85,3 % |
| Media 9×9, 6000² | 1250,26 | 3,743 | 29,90 | 334× | 41,8× | 87,2 % |
| Media 21×21, 4000² | 2669,70 | 8,001 | 16,92 | 334× | 157,8× | 52,6 % |

**Hilos por bloque** (media 9×9, 4000², kernel): 128 → 1,646 ms · 256 → 1,624 ms · 512 → 1,636 ms (diferencia máxima 1,4 %).

El % de transferencias se calcula con la mediana de cada fase; como las medianas no son aditivas, puede diferir ligeramente de (total − kernel) / total.

CPU y CUDA se midieron en la misma sesión (4 de noviembre de 2026). Los resultados se limitan al equipo y las configuraciones evaluadas. El análisis completo está en el informe.

---


