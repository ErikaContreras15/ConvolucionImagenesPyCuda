#!/usr/bin/env bash
# ejecutar_p03.sh — experimentos de la Práctica 03 (un factor a la vez; 256 hilos/bloque; color).
#
#   0. Equipo, versiones, imagen Docker y commit           -> resultados/equipo_p03.txt
#   1. Integración y contenedor (media 9x9, 4000x4000):
#        a) PyCUDA directo en el ANFITRIÓN                   -> resultados/exp1_directo_anfitrion.csv
#        b) PyCUDA directo DENTRO del contenedor             -> resultados/exp1_directo_contenedor.csv
#        c) solicitud al SERVICIO contenerizado              -> resultados/exp1_servicio.csv
#   2. Resolución (media 9x9; 2000, 4000, 6000) - servicio   -> resultados/exp2_resolucion.csv
#   3. Filtros y máscara (4000; media 3/9/21, nitidez, laplaciano) - servicio -> resultados/exp3_filtros.csv
#   4. Concurrencia (media 9x9, 4000; 1 y 2 clientes; 5 rondas) - servicio    -> resultados/exp4_concurrencia.csv
#
# Cada configuración: 1 calentamiento (repetición 0, registrado aparte) + 5 repeticiones.
#
# Uso (desde Practica03, con el venv activado, laptop al cargador y en modo rendimiento,
#      SIN otro servicio usando el puerto 5000):
#   bash ejecutar_p03.sh 2>&1 | tee resultados/log_p03.txt
set -uo pipefail

REPS=5
HILOS=256
IMG=imagenes_color
R=resultados
S=salidas
IMAGEN_DOCKER=conv-pycuda:p03
URL=http://localhost:5000
RAIZ=$(cd .. && pwd)

mkdir -p "$R" "$S"
rm -f "$R"/exp*.csv

echo "== 0. Equipo y versiones =="
{
    echo "### Fecha (local)";   date
    echo; echo "### CPU";       lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core"
    echo; echo "### RAM";       free -h | head -2
    echo; echo "### GPU y driver (anfitrión)"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    echo; echo "### Sistema";   lsb_release -d 2>/dev/null; uname -r
    echo; echo "### Docker";    docker --version
    echo; echo "### NVIDIA Container Toolkit"; nvidia-ctk --version 2>/dev/null | head -1
    echo; echo "### Runtime GPU en Docker"; docker info 2>/dev/null | grep -iE "runtimes|default runtime"
    echo; echo "### Imagen Docker"
    docker images "$IMAGEN_DOCKER" --format "{{.Repository}}:{{.Tag}}  id {{.ID}}  creada {{.CreatedAt}}  {{.Size}}"
    echo "Base: nvidia/cuda:13.2.0-devel-ubuntu24.04"
    echo; echo "### Dentro del contenedor"
    docker run --rm --gpus all "$IMAGEN_DOCKER" bash -c \
      "nvcc --version | tail -2; python3 -c \"import sys,numpy,PIL,pycuda,flask,importlib.metadata as m; print('Python',sys.version.split()[0]); print('NumPy',numpy.__version__); print('Pillow',PIL.__version__); print('PyCUDA',pycuda.VERSION_TEXT); print('Flask',m.version('flask')); print('gunicorn',m.version('gunicorn'))\"" 2>/dev/null
    echo; echo "### Anfitrión (venv ~/venvs/cuda)"
    nvcc --version | tail -2
    python3 -c "import sys,numpy,PIL,pycuda; print('Python',sys.version.split()[0]); print('NumPy',numpy.__version__); print('Pillow',PIL.__version__); print('PyCUDA',pycuda.VERSION_TEXT)"
    echo; echo "### Código"
    # Código medido = el commit actual. Se ignoran resultados/ y graficas/ (este script los regenera)
    if git -C "$RAIZ" diff --quiet HEAD -- . ':(exclude)Practica03/resultados' ':(exclude)Practica03/graficas'; then
        echo "Commit: $(git -C "$RAIZ" rev-parse --short HEAD) (código sin cambios respecto al commit)"
    else
        echo "Commit: $(git -C "$RAIZ" rev-parse --short HEAD) (AVISO: hay cambios de código sin commit)"
        git -C "$RAIZ" diff --stat HEAD -- . ':(exclude)Practica03/resultados' ':(exclude)Practica03/graficas'
    fi
    echo "Commit de origen de la P02: 822f7b2"
    echo; echo "### Servicio"
    echo "gunicorn --workers 1 --threads 8 --timeout 300 --bind 0.0.0.0:5000 app:app (COLA_MAX=4)"
    echo "Copias CPU<->GPU con memoria PAGINABLE; kernel medido con eventos CUDA; resto con time.perf_counter"
} > "$R/equipo_p03.txt"
cat "$R/equipo_p03.txt"

docker rm -f conv-p03 >/dev/null 2>&1

echo; echo "== 1a. PyCUDA directo en el ANFITRIÓN (media 9x9, 4000x4000) =="
python3 directo.py "$IMG/img_4000.ppm" media 9 $HILOS $REPS "$S/exp1_anfitrion" "$R/exp1_directo_anfitrion.csv"

echo; echo "== 1b. PyCUDA directo DENTRO del contenedor =="
docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD/$IMG:/datos:ro" -v "$PWD/$R:/resultados" -v "$PWD/$S:/salidas" "$IMAGEN_DOCKER" \
    python3 directo.py /datos/img_4000.ppm media 9 $HILOS $REPS /salidas/exp1_contenedor /resultados/exp1_directo_contenedor.csv

echo; echo "== Iniciando el servicio contenerizado =="
docker run -d --rm --gpus all -p 5000:5000 --name conv-p03 "$IMAGEN_DOCKER" >/dev/null
for i in $(seq 1 60); do
    curl -sf "$URL/api/salud" >/dev/null && break
    sleep 1
done
curl -s "$URL/api/salud"; echo

echo; echo "== 1c. Solicitud al SERVICIO contenerizado =="
python3 cliente.py serie "$URL" "$IMG/img_4000.ppm" media 9 $REPS "$R/exp1_servicio.csv"

echo; echo "== 2. Resolución (media 9x9) =="
for n in 2000 4000 6000; do
    python3 cliente.py serie "$URL" "$IMG/img_${n}.ppm" media 9 $REPS "$R/exp2_resolucion.csv"
done

echo; echo "== 3. Filtros y máscara (4000x4000) =="
for cfg in "media 3" "media 9" "media 21" "nitidez 3" "laplaciano 3"; do
    set -- $cfg
    python3 cliente.py serie "$URL" "$IMG/img_4000.ppm" "$1" "$2" $REPS "$R/exp3_filtros.csv"
done

echo; echo "== 4. Concurrencia (media 9x9, 4000x4000; 5 rondas) =="
for c in 1 2; do
    python3 cliente.py concurrencia "$URL" "$IMG/img_4000.ppm" media 9 $c 5 "$R/exp4_concurrencia.csv"
done

echo; echo "== Registro del contenedor (evidencia) =="
docker logs conv-p03 > "$R/log_contenedor.txt" 2>&1
tail -n 5 "$R/log_contenedor.txt"
docker stop conv-p03 >/dev/null

echo; echo "Listo: [$(date +%H:%M:%S)] CSV en $R/ (exp1..exp4), equipo_p03.txt y log_contenedor.txt"
