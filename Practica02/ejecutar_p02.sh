#!/usr/bin/env bash
# Orden:
#   0. Equipo y versiones -> equipo_p02.txt
#   1. Compilación de C/C++ y CUDA C 
#   2. Prueba pequeña 5x5 en las 4 versiones                 -> resultados/pruebas.txt
#   3. Prueba de límites 1003x997 (128, 256 y 512 hilos)
#   4. Experimentos 1-4 en C/C++, CUDA C y PyCUDA
#   5. Experimentos 1-3 en Python secuencial (al final: es lo más lento, ~2 h)
#   6. Validación numérica de todo contra C/C++              -> resultados/validacion.txt
#
# Experimentos (un factor a la vez; 256 hilos/bloque salvo en el de hilos):
#   1. Filtro:     media 3x3, nitidez, laplaciano  | 4000x4000
#   2. Resolución: media 9x9                       | 2000, 4000, 6000
#   3. Máscara:    media 3x3, 9x9, 21x21           | 4000x4000
#   4. Hilos:      media 9x9 4000x4000 (solo CUDA C y PyCUDA) | 128, 256, 512
#
# Uso (desde Practica02, con el venv activado y la laptop al cargador):
#   bash ejecutar_p02.sh 2>&1 | tee resultados/log_p02.txt
set -uo pipefail

REPS=5
HILOS=256
LIMITE_PY=3600                 # segundos máximos por repetición de Python secuencial
IMG=../CPU/imgCPU
R=resultados
S=salidas
VAL=$R/validacion.txt

mkdir -p "$R" "$S" bin
rm -f "$R"/resultados_*.csv "$R"/prueba_limites_*.csv "$VAL" "$R/pruebas.txt"
rm -f "$S"/*.f32            # para que ninguna salida vieja se reutilice

echo "== 0. Equipo y versiones =="
{
    echo "### Fecha";          date
    echo; echo "### CPU";      lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
    echo; echo "### RAM";      free -h | head -2
    echo; echo "### GPU";      nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo; echo "### Sistema";  lsb_release -d 2>/dev/null; uname -r
    echo; echo "### CUDA Toolkit"; nvcc --version | tail -2
    echo; echo "### g++";      g++ --version | head -1
    echo; echo "### Python y librerías"
    python3 -c "import sys,numpy,PIL,pycuda; print('Python',sys.version.split()[0]); print('NumPy',numpy.__version__); print('Pillow',PIL.__version__); print('PyCUDA',pycuda.VERSION_TEXT)"
    echo; echo "### Entorno"
    echo "venv ~/venvs/cuda: python3 -m venv ~/venvs/cuda && pip install pycuda numpy pillow matplotlib"
    echo; echo "### Compilación"
    echo "g++  -O2 -std=c++17 -I.. ../CPU/conv_cpu.cpp  -o bin/conv_cpu"
    echo "nvcc -O2 -std=c++17 -arch=native -I.. ../CUDA/conv_cuda.cu -o bin/conv_cuda"
    echo "PyCUDA: SourceModule(kernel_convolucion.cu, options=['-O2'], cache_dir=False)"
} > equipo_p02.txt
cat equipo_p02.txt

echo; echo "== 1. Compilando C/C++ y CUDA C =="
g++  -O2 -std=c++17 -I.. ../CPU/conv_cpu.cpp  -o bin/conv_cpu  || exit 1
nvcc -O2 -std=c++17 -arch=native -I.. ../CUDA/conv_cuda.cu -o bin/conv_cuda || exit 1

echo; echo "== 2. Prueba pequeña 5x5 (las 4 versiones) =="
{
    bin/conv_cpu  --test; echo
    bin/conv_cuda --test; echo
    python3 PyCuda/conv_pycuda.py --test; echo
    python3 Python/conv_python.py --test
} 2>&1 | tee "$R/pruebas.txt"

# Compara una salida contra la referencia C/C++ y la anota en validacion.txt
validar() { python3 comparar.py "$1.f32" "$2.f32" "$3" "$VAL" || true; }

echo; echo "== 3. Prueba de límites 1003x997 (media 9x9) =="
python3 -c "from PIL import Image; Image.open('$IMG/img_2000.pgm').convert('L').crop((0,0,1003,997)).save('$S/prueba_1003x997.pgm')"
P=$S/prueba_1003x997.pgm
echo "--- Prueba de límites 1003x997, media 9x9 ---" >> "$VAL"
bin/conv_cpu "$P" media 9 1 "$S/cpp_prueba" "$R/prueba_limites_cpp.csv"
for h in 128 256 512; do
    bin/conv_cuda "$P" media 9 $h 1 "$S/cuda_prueba_h$h" "$R/prueba_limites_cuda.csv"
    python3 PyCuda/conv_pycuda.py "$P" media 9 $h 1 "$S/pycuda_prueba_h$h" "$R/prueba_limites_pycuda.csv"
    validar "$S/cpp_prueba" "$S/cuda_prueba_h$h"   "CUDA C  1003x997 h$h"
    validar "$S/cpp_prueba" "$S/pycuda_prueba_h$h" "PyCUDA  1003x997 h$h"
done
python3 Python/conv_python.py "$P" media 9 1 "$S/py_prueba" "$R/prueba_limites_python.csv" "$LIMITE_PY"
validar "$S/cpp_prueba" "$S/py_prueba" "Python  1003x997"

# Corre C/C++, CUDA C y PyCUDA para una configuración y valida contra C/C++.
#   $1 filtro  $2 k  $3 resolución  $4 hilos
correr_rapidas() {
    local f=$1 k=$2 n=$3 h=$4
    local ref="$S/cpp_${f}${k}_${n}"
    if [[ ! -f "${ref}.f32" ]]; then           # C/C++ no depende de los hilos: una vez
        bin/conv_cpu "$IMG/img_${n}.pgm" "$f" "$k" "$REPS" "$ref" "$R/resultados_cpp.csv"
    fi
    bin/conv_cuda "$IMG/img_${n}.pgm" "$f" "$k" "$h" "$REPS" "$S/cuda_${f}${k}_${n}_h$h" "$R/resultados_cuda.csv"
    python3 PyCuda/conv_pycuda.py "$IMG/img_${n}.pgm" "$f" "$k" "$h" "$REPS" "$S/pycuda_${f}${k}_${n}_h$h" "$R/resultados_pycuda.csv"
    validar "$ref" "$S/cuda_${f}${k}_${n}_h$h"   "CUDA C  ${f} k=$k ${n}² h$h"
    validar "$ref" "$S/pycuda_${f}${k}_${n}_h$h" "PyCUDA  ${f} k=$k ${n}² h$h"
}

echo; echo "== 4. Experimentos en C/C++, CUDA C y PyCUDA =="
echo "--- Experimentos ---" >> "$VAL"
correr_rapidas media      3  4000 $HILOS     # exp 1 y 3
correr_rapidas nitidez    3  4000 $HILOS     # exp 1
correr_rapidas laplaciano 3  4000 $HILOS     # exp 1
correr_rapidas media      9  2000 $HILOS     # exp 2
correr_rapidas media      9  4000 $HILOS     # exp 2, 3 y 4
correr_rapidas media      9  6000 $HILOS     # exp 2
correr_rapidas media      21 4000 $HILOS     # exp 3
correr_rapidas media      9  4000 128        # exp 4
correr_rapidas media      9  4000 512        # exp 4

echo; echo "== 5. Python secuencial (lo más lento, al final) =="
correr_python() {
    local f=$1 k=$2 n=$3
    echo "[$(date +%H:%M:%S)] Python ${f} k=$k ${n}x${n}"
    if python3 Python/conv_python.py "$IMG/img_${n}.pgm" "$f" "$k" "$REPS" "$S/py_${f}${k}_${n}" \
               "$R/resultados_python.csv" "$LIMITE_PY"; then
        validar "$S/cpp_${f}${k}_${n}" "$S/py_${f}${k}_${n}" "Python  ${f} k=$k ${n}²"
    else
        echo "Python  ${f} k=$k ${n}² | no completada (sin salida que validar)" >> "$VAL"
    fi
}
correr_python media      3  4000
correr_python nitidez    3  4000
correr_python laplaciano 3  4000
correr_python media      9  2000
correr_python media      9  4000
correr_python media      9  6000
correr_python media      21 4000

echo; echo "== 6. Validación (referencia: C/C++ secuencial, tolerancia 1e-2) =="
cat "$VAL"
echo; echo "Listo: [$(date +%H:%M:%S)] CSV en $R/, imágenes en $S/, equipo_p02.txt"
