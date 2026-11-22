# Práctica 03 · Servicio web de convolución de imágenes con Docker y PyCUDA

Asignatura: Computación Paralela · Universidad Politécnica Salesiana

Integrantes: Erika Contreras - Jorge Pizarro

La convolución de la práctica 02 (PyCUDA) se ofrece como **servicio web**: el navegador sube una imagen,
elige el filtro, el servicio la procesa en la GPU y devuelve el resultado. Todo se ejecuta **dentro de un
contenedor Docker** con acceso a la GPU NVIDIA del anfitrión.

```
Navegador → servicio web Python (Flask + gunicorn) → cola → hilo GPU (PyCUDA) → GPU → resultado → navegador
```

## Continuidad con la práctica 02

| | |
|---|---|
| Commit de origen de la P02 | `822f7b2` (repositorio `ConvolucionImagenesPyCuda`) |
| Reutilizado sin cambios | `Practica02/PyCuda/kernel_convolucion.cu` (un píxel por hilo, distribución en X), `iniciar_cuda()`, `compilar_kernel()` de `conv_pycuda.py`, máscaras de `comun.py` |
| Filtros obligatorios | media 3×3, 9×9 y 21×21 · nitidez 3×3 · laplaciano 3×3 (mismos coeficientes, float32, extensión con ceros) |
| Filtros adicionales (solo demostración) | gaussiano 5×5 · relieve 3×3 (el mismo kernel, máscaras nuevas) |
| Cambio justificado | **Color**: la imagen RGB se separa en 3 planos; suben juntos a la GPU en una copia, el **mismo kernel** se lanza una vez por plano y bajan juntos en una copia. Validado canal por canal contra la P02 con error 0. |

## Estructura

```
Practica01/                     ← raíz del repositorio (contexto de docker build)
├── .dockerignore               ← solo envía a Docker el código que el servicio necesita
├── Practica02/                 ← kernel y utilidades de la P02 (sin cambios)
└── Practica03/
    ├── Dockerfile
    ├── requirements.txt        ← versiones fijas
    ├── motor_gpu.py            ← convolución con PyCUDA, separada de la parte web
    ├── app.py                  ← servicio Flask (HTTP, validación, cola, resultados)
    ├── templates/index.html    ← interfaz web
    ├── directo.py              ← mismo motor sin web (referencia directa)
    ├── registro.py             ← formato del CSV
    ├── preparar_imagenes_color.py
    ├── validar_color.py        ← color vs P02, canal por canal
    ├── pruebas_servicio.py     ← validación funcional y numérica del servicio
    ├── cliente.py              ← cliente de medición
    ├── ejecutar_p03.sh         ← todos los experimentos en orden
    ├── analisis_p03.py         ← resúmenes y gráficas
    ├── resultados/             ← CSV, registros y equipo_p03.txt
    ├── graficas/               ← g1 … g5
    └── evidencia/              ← recortes de los filtros (misma región)
```

## Entorno usado

| Componente | Versión |
|---|---|
| Anfitrión | ASUS ROG Strix · Intel Core Ultra 9 275HX (24 núcleos) · 32 GB RAM · Ubuntu/Kubuntu 24.04.4 · kernel 7.0.0-43 |
| GPU y **controlador (del anfitrión)** | NVIDIA GeForce RTX 5080 Laptop GPU, 16 GB · driver **595.99.02** (soporta hasta CUDA 13.2) |
| Docker / runtime GPU | Docker 29.10.0 · NVIDIA Container Toolkit 1.20.2 (runtime `nvidia`) |
| Imagen base (**CUDA dentro de la imagen**) | `nvidia/cuda:13.2.0-devel-ubuntu24.04` (nvcc 13.2) |
| Python y librerías (en la imagen) | Python 3.12.3 · NumPy 2.5.4 · Pillow 13.0.0 · PyCUDA 2026.1 · Flask 3.1.3 · gunicorn 27.0.1 |

El controlador NVIDIA **no va dentro de la imagen**: lo aporta el anfitrión y el NVIDIA Container Toolkit lo
expone al contenedor con `--gpus all`. CUDA, nvcc, Python y PyCUDA sí van dentro de la imagen.
La etiqueta CUDA de la imagen (13.2) no debe superar la versión que soporta el driver (`nvidia-smi`).

## 1. Requisitos en el anfitrión

```bash
nvidia-smi                       # driver NVIDIA instalado; "CUDA Version" ≥ 13.2
docker --version                 # Docker Engine
nvidia-ctk --version             # NVIDIA Container Toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker   # una vez
```

## 2. Construir la imagen

Desde la **raíz del repositorio** (la carpeta que contiene `Practica02` y `Practica03`):

```bash
docker build -f Practica03/Dockerfile -t conv-pycuda:p03 .
docker images conv-pycuda:p03          # identificador de la imagen construida
```

## 3. Comprobar el acceso a la GPU

```bash
docker run --rm --gpus all conv-pycuda:p03 nvidia-smi
docker run --rm --gpus all conv-pycuda:p03 python3 directo.py --test     # 12 OK (gris + R, G, B)
```

## 4. Iniciar el servicio

```bash
docker run --rm --gpus all -p 5000:5000 --name conv-p03 conv-pycuda:p03
```

Arranca con **gunicorn** (sin depurador ni recarga automática): `--workers 1 --threads 8 --timeout 300`.
Al iniciar crea el contexto CUDA, compila el kernel una vez y escribe `GPU lista: {...}`.

- Interfaz: `http://localhost:5000` o, desde otro dispositivo de la misma red, `http://<IP del anfitrión>:5000`
  (la IP se obtiene con `hostname -I`).
- Detener: `Ctrl + C` o `docker stop conv-p03`.

Variables de entorno opcionales (`-e NOMBRE=valor`):

| Variable | Valor por defecto | Significado |
|---|---|---|
| `COLA_MAX` | 4 | solicitudes que pueden esperar mientras la GPU atiende una; más allá se rechaza con 503 |
| `MAX_MB` | 120 | tamaño máximo del archivo subido |
| `MAX_LADO` | 6000 | ancho y alto máximos en píxeles |
| `TTL_S` | 900 | segundos que se conservan los resultados antes de borrarse |
| `HILOS_BLOQUE` | 256 | hilos por bloque del kernel |

## 5. API

### `POST /api/procesar` (multipart/form-data)

| Campo | Tipo | Valores |
|---|---|---|
| `imagen` | archivo | PNG, JPEG, PPM/PGM, BMP o TIFF (se valida el contenido real, no la extensión) |
| `filtro` | texto | `media`, `nitidez`, `laplaciano`, `gaussiano`, `relieve` |
| `k` | entero | media: 3, 9 o 21 · nitidez y laplaciano: 3 · gaussiano: 5 · relieve: 3 |

```bash
curl -F imagen=@imagenes_color/img_2000.ppm -F filtro=media -F k=9 http://localhost:5000/api/procesar
```

Respuesta **200** (JSON):

```json
{
  "id": "814ea1ee72a24cf7a69d546e2ae4eb2f",
  "filtro": "media", "k": 9, "ancho": 4000, "alto": 4000, "canales": 3,
  "hilos_bloque": 256, "bloques": 62500, "gpu": "NVIDIA GeForce RTX 5080 Laptop GPU", "entorno": "contenedor",
  "unidad": "ms",
  "tiempos": { "recepcion": 53.4, "decodificacion": 155.6, "espera_cola": 0.2,
               "preparacion": 0.07, "reserva": 0.5, "h2d": 11.5, "kernel": 4.97, "d2h": 26.0,
               "total_gpu": 42.5, "liberacion": 1.3, "procesamiento_gpu": 66.0,
               "salida": 707.6, "servidor_total": 983.3 },
  "archivos": { "png": "/api/resultado/<id>/resultado.png", "f32": "/api/resultado/<id>/resultado.f32" }
}
```

- `resultado.png`: resultado visual (redondeo y recorte a 0–255, igual que la P02).
- `resultado.f32`: float32 crudo **sin recortar**, plano por plano `(canales, alto, ancho)`; sirve para validar.
- Con PPM o TIFF se agrega `original.png` para mostrar el original en el navegador.

Respuestas de error (JSON `{"error": código, "mensaje": texto, "id": ...}`):

| HTTP | `error` | Cuándo |
|---|---|---|
| 400 | `falta_imagen` · `imagen_invalida` · `parametro_invalido` | sin archivo, archivo que no es imagen, filtro o `k` no permitido |
| 413 | `archivo_muy_grande` · `dimensiones_excedidas` | más de `MAX_MB` o de `MAX_LADO`×`MAX_LADO` |
| 415 | `formato_no_permitido` | por ejemplo GIF |
| 503 | `cola_llena` · `gpu_no_disponible` · `gpu_iniciando` | servicio ocupado o sin GPU (**no** se usa la CPU en su lugar) |
| 507 | `sin_memoria_gpu` | la imagen no cabe en la memoria de la GPU |
| 500 | `error_procesamiento` | fallo inesperado en la GPU |

### `GET /api/salud`

Estado del servicio y de la GPU, tamaño de la cola, límites y tiempos de arranque (contexto, compilación).

### `GET /api/resultado/<id>/<archivo>`

Descarga `resultado.png`, `resultado.f32` u `original.png`. Con `?descargar=1` se descarga como archivo.

### Gestión de solicitudes

- Un **único hilo GPU** es dueño del contexto CUDA: lo crea al arrancar, compila el kernel una vez y atiende
  las solicitudes **en orden de llegada (FIFO), una a la vez**. Los 8 hilos HTTP reciben, validan y decodifican
  en paralelo y esperan en la cola.
- Cada solicitud tiene un **id único (uuid4)** y su **carpeta propia** en `/tmp/conv_servicio/<id>/`:
  ninguna sobrescribe ni recibe el resultado de otra. Las carpetas se borran tras `TTL_S` segundos.
- Por cada solicitud el motor reserva la memoria de la GPU, copia, ejecuta y la **libera siempre**
  (también si hay error). Las copias usan **memoria paginable**.

## 6. Imágenes de entrada

Se generan a color desde la **misma foto original de la P01** (`Practica01/imagenOriginal/montana.jpg`,
8192×5120, 7,2 MB, incluida en el repositorio), con el mismo procedimiento que la P01 pero sin convertir a
gris: recorte cuadrado centrado de 5120×5120 y redimensionado LANCZOS a 2000, 4000 y 6000 px (la de 6000 se
amplía ×1,17).

SHA-256 de la foto original:
`7f311620023a4158b1f34cac56a7241c3c84ceed9d61807797a3dc09660866b5`

```bash
cd Practica03
python3 preparar_imagenes_color.py ../Practica01/imagenOriginal/montana.jpg imagenes_color
sha256sum imagenes_color/*.ppm     # para comprobar que son las mismas
```

Las `.ppm` (12, 46 y 103 MB) no se suben al repositorio.

## 7. Ejecución directa (sin web) y validación

En el anfitrión, con un entorno virtual que tenga `numpy pillow pycuda` (mismas versiones que `requirements.txt`):

```bash
cd Practica03
python3 directo.py --test                                          # prueba manual 5x5, 12 OK
python3 directo.py imagenes_color/img_4000.ppm media 9 256 5 salidas/directo_media9_4000 resultados/directo.csv
python3 validar_color.py imagenes_color/img_4000.ppm media 9 salidas/directo_media9_4000.f32 resultados/validacion.txt
```

Validación del servicio (con el servicio corriendo; requiere `requests==2.34.2`):

```bash
python3 pruebas_servicio.py --url http://localhost:5000            # resultados/pruebas_servicio.txt y evidencia/
```

Cubre: prueba 5×5 y 1003×997 (hilos sobrantes), equivalencia directo–servicio de los 5 filtros obligatorios,
resultado visual con el mismo recorte, 9 entradas inválidas y recuperación, aislamiento de 2 solicitudes
simultáneas y comportamiento de la cola (con `-e COLA_MAX=1` aparece el rechazo 503).

## 8. Experimentos

```bash
cd Practica03
bash ejecutar_p03.sh 2>&1 | tee resultados/log_p03.txt    # ~3 min; arranca y detiene su propio contenedor
python3 analisis_p03.py                                   # resumen_p03.csv, arranque_p03.csv y graficas/
```

Un factor a la vez; 256 hilos por bloque; imágenes a color; 1 calentamiento (repetición 0, registrado aparte)
+ 5 repeticiones por configuración; concurrencia en 5 rondas. Se reporta la mediana y Q1–Q3.

| Experimento | Configuración | Resultado principal (medianas) |
|---|---|---|
| 1. Integración y contenedor | media 9×9, 4000×4000 | kernel 3,57 ms (directo anfitrión) · 3,43 ms (directo contenedor) · 3,29 ms (servicio) |
| 2. Resolución | media 9×9; 2000², 4000², 6000² | kernel 0,85 → 3,29 → 7,41 ms · servidor 211 → 905 → 1894 ms · cliente 416 → 1588 → 3254 ms |
| 3. Filtros y máscara | 4000²; media 3/9/21, nitidez, laplaciano | kernel 0,98 / 3,30 / 16,9 ms (3×3, 9×9, 21×21); servidor 823–1259 ms según la salida PNG |
| 4. Concurrencia | media 9×9, 4000²; 1 y 2 clientes | 0,67 → 1,16 solicitudes/s; latencia 1491 → 1670 ms; espera en cola ≤ 3,6 ms; 0 fallos |

Medición: CPU→GPU, kernel y GPU→CPU con **eventos CUDA** (y sincronización); el resto con `time.perf_counter`
en el servidor; el cliente mide desde el envío hasta recibir el PNG completo (incluye comunicación; no equivale
al tiempo del kernel). Cliente en el mismo equipo (localhost).

## 9. Reproducir desde una copia limpia

```bash
git clone <URL del repositorio> copia_limpia && cd copia_limpia
git log --oneline -1
docker build -f Practica03/Dockerfile -t conv-pycuda:p03-limpia .
docker run --rm --gpus all conv-pycuda:p03-limpia python3 directo.py --test
```

## Limitaciones

- No se contó con créditos en un proveedor de nube con GPU: el servicio se desplegó en un **servidor local** con
  GPU compatible, accesible desde otros dispositivos de la red local (se demostró desde un celular por WiFi).
- Las mediciones de concurrencia se hicieron con el cliente en el mismo equipo (localhost): el tiempo del
  cliente no incluye una red física.
- La laptop ajusta las frecuencias de la GPU según la carga y la energía: la primera solicitud tras un periodo
  sin uso es más lenta (arranque en frío, registrado aparte en `arranque_p03.csv`).
- Los registros del contenedor están en UTC (la imagen no define zona horaria); los de los scripts, en hora local (UTC−5).