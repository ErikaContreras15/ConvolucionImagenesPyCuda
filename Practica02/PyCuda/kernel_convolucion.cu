



// Cada hilo calcula UN píxel de salida leyendo la imagen desde memoria global.

__global__ void convolucionGPU(const float* in, float* out, int ancho, int alto,
                               const float* m, int k) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;   // índice global (bloques e hilos en X)
    int N  = ancho * alto;
    if (id >= N) return;                              // hilos sobrantes del último bloque

    int fila = id / ancho;                            // índice global -> fila y columna
    int col  = id % ancho;
    int r    = k / 2;

    float suma = 0.0f;
    for (int i = -r; i <= r; ++i) {
        int y = fila + i;
        if (y < 0 || y >= alto) continue;             // extensión con ceros
        for (int j = -r; j <= r; ++j) {
            int x = col + j;
            if (x < 0 || x >= ancho) continue;        // extensión con ceros
            suma += in[y * ancho + x] * m[(i + r) * k + (j + r)];
        }
    }
    out[id] = suma;                                  
}
