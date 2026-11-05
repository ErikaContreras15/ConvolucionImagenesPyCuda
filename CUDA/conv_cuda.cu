// conv_cuda.cu — convolución en GPU con CUDA C (un hilo por píxel de salida)

#include "common.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

// Verifica cada llamada CUDA 
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {            \
    std::fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__,          \
                 cudaGetErrorString(e)); std::exit(1); } } while (0)

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
    out[id] = suma;                                   // arreglo de salida separado
}

struct Tiempos { float h2d, kernel, d2h, total; };

// Ejecuta una vez el ciclo completo (H->D, kernel, D->H) y mide cada fase con eventos.
// La memoria de GPU (d_in, d_out, d_m) 
Tiempos ejecutar(const std::vector<float>& h_in, std::vector<float>& h_out,
                 const std::vector<float>& h_m, float* d_in, float* d_out, float* d_m,
                 int ancho, int alto, int k, int hilos) {
    const int N = ancho * alto;
    const int bloques = (N + hilos - 1) / hilos;
    const size_t bytesImg = static_cast<size_t>(N) * sizeof(float);

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2)); CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), bytesImg, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m, h_m.data(), h_m.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    convolucionGPU<<<bloques, hilos>>>(d_in, d_out, ancho, alto, d_m, k);
    CUDA_CHECK(cudaGetLastError());                   // error al lanzar el kernel
    CUDA_CHECK(cudaEventRecord(e2));
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, bytesImg, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaEventSynchronize(e3));             // asegurar que todo terminó

    Tiempos t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d,    e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h,    e2, e3));
    CUDA_CHECK(cudaEventElapsedTime(&t.total,  e0, e3));
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3);
    return t;
}

// Prueba manual: matriz 5x5 
// Se usan 4 hilos por bloque: 25 píxeles -> 7 bloques = 28 hilos (3 sobrantes),
// se comprueba el control de límites.
int pruebaManual() {
    const int w = 5, h = 5, hilos = 4;
    std::vector<float> in = { 35, 40, 41, 45, 50,
                              40, 40, 42, 46, 52,
                              42, 46, 50, 55, 55,
                              48, 52, 56, 58, 60,
                              56, 60, 65, 70, 75 };
    int k = 3;
    auto m = crearMascara("laplaciano", k);
    std::vector<float> out(w * h);
    float *d_in, *d_out, *d_m;
    CUDA_CHECK(cudaMalloc(&d_in,  w * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, w * h * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m,   m.size() * sizeof(float)));
    ejecutar(in, out, m, d_in, d_out, d_m, w, h, k, hilos);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_m);

    std::cout << "Salida laplaciano 5x5 (GPU, " << hilos << " hilos/bloque):\n";
    for (int f = 0; f < h; ++f) {
        for (int c = 0; c < w; ++c) std::printf("%6.0f", out[f * w + c]);
        std::printf("\n");
    }
    struct { int f, c; float esperado; } casos[] = { {0,0,-60}, {1,1,8}, {2,2,-1} };
    bool ok = true;
    for (auto& t : casos) {
        float v = out[t.f * w + t.c];
        bool bien = std::fabs(v - t.esperado) < 1e-4f;
        std::printf("(%d,%d): obtenido %.1f, esperado %.1f -> %s\n",
                    t.f, t.c, v, t.esperado, bien ? "OK" : "ERROR");
        ok &= bien;
    }
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--test") return pruebaManual();
    if (argc != 8) {
        std::cerr << "Uso: " << argv[0]
                  << " <entrada.pgm> <filtro> <k> <hilos_bloque> <reps> <prefijo_salida> <resultados.csv>\n"
                  << "     " << argv[0] << " --test\n";
        return 1;
    }
    try {
        std::string rutaIn = argv[1], filtro = argv[2], prefijo = argv[6], csv = argv[7];
        int k = std::stoi(argv[3]), hilos = std::stoi(argv[4]), reps = std::stoi(argv[5]);

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (hilos < 1 || hilos > prop.maxThreadsPerBlock)
            throw std::runtime_error("hilos_bloque fuera del rango del dispositivo");

        // --- Preparación (NO se mide) ---
        Imagen img = leerPGM(rutaIn);
        auto mascara = crearMascara(filtro, k);
        const int N = img.ancho * img.alto;
        const int bloques = (N + hilos - 1) / hilos;
        std::vector<float> salida(N);

        float *d_in, *d_out, *d_m;                    // reserva previa, fuera de la medición
        CUDA_CHECK(cudaMalloc(&d_in,  static_cast<size_t>(N) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(N) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_m,   mascara.size() * sizeof(float)));

        // --- Calentamiento (no se registra) ---
        ejecutar(img.px, salida, mascara, d_in, d_out, d_m, img.ancho, img.alto, k, hilos);

        // --- Repeticiones medidas ---
        std::vector<double> tk, tt;
        for (int rep = 1; rep <= reps; ++rep) {
            Tiempos t = ejecutar(img.px, salida, mascara, d_in, d_out, d_m,
                                 img.ancho, img.alto, k, hilos);
            escribirCSV(csv, "cuda", filtro, img.ancho, img.alto, k, hilos, bloques, rep, "h2d",    t.h2d);
            escribirCSV(csv, "cuda", filtro, img.ancho, img.alto, k, hilos, bloques, rep, "kernel", t.kernel);
            escribirCSV(csv, "cuda", filtro, img.ancho, img.alto, k, hilos, bloques, rep, "d2h",    t.d2h);
            escribirCSV(csv, "cuda", filtro, img.ancho, img.alto, k, hilos, bloques, rep, "total",  t.total);
            tk.push_back(t.kernel); tt.push_back(t.total);
        }
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_m);

        // --- Salidas (NO se miden) ---
        guardarRAW(prefijo + ".f32", salida);
        guardarPGM(prefijo + ".pgm", salida, img.ancho, img.alto);

        std::printf("CUDA %s k=%d %dx%d | %d hilos x %d bloques | kernel %.3f ms | total %.3f ms (mediana, %d reps)\n",
                    filtro.c_str(), k, img.ancho, img.alto, hilos, bloques,
                    mediana(tk), mediana(tt), reps);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
