
#include "common.h"
#include <chrono>
#include <cstdio>
#include <iostream>

// Un "píxel de salida" por iteración de los dos bucles externos
void convolucionCPU(const float* in, float* out, int ancho, int alto,
                    const float* m, int k) {
    const int r = k / 2;
    for (int fila = 0; fila < alto; ++fila) {
        for (int col = 0; col < ancho; ++col) {
            float suma = 0.0f;
            for (int i = -r; i <= r; ++i) {
                int y = fila + i;
                if (y < 0 || y >= alto) continue;          // fuera: aporta 0
                for (int j = -r; j <= r; ++j) {
                    int x = col + j;
                    if (x < 0 || x >= ancho) continue;     // fuera: aporta 0
                    suma += in[y * ancho + x] * m[(i + r) * k + (j + r)];
                }
            }
            out[fila * ancho + col] = suma;                // arreglo de salida separado
        }
    }
}

// Prueba manual con la matriz 5x5 y  el laplaciano
// Valores calculados a mano:
//   (0,0): 0+0 -4*35 +40 +40       = -60
//   (1,1): 40+40 -4*40 +42 +46     =   8
//   (2,2): 42+46 -4*50 +55 +56     =  -1
int pruebaManual() {
    const int w = 5, h = 5;
    std::vector<float> in = { 35, 40, 41, 45, 50,
                              40, 40, 42, 46, 52,
                              42, 46, 50, 55, 55,
                              48, 52, 56, 58, 60,
                              56, 60, 65, 70, 75 };
    int k = 3;
    auto m = crearMascara("laplaciano", k);
    std::vector<float> out(w * h);
    convolucionCPU(in.data(), out.data(), w, h, m.data(), k);

    std::cout << "Salida laplaciano 5x5:\n";
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
    if (argc != 7) {
        std::cerr << "Uso: " << argv[0]
                  << " <entrada.pgm> <filtro> <k> <reps> <prefijo_salida> <resultados.csv>\n"
                  << "     " << argv[0] << " --test\n";
        return 1;
    }
    try {
        std::string rutaIn = argv[1], filtro = argv[2], prefijo = argv[5], csv = argv[6];
        int k = std::stoi(argv[3]), reps = std::stoi(argv[4]);

        // --- Preparación (NO se mide) ---
        Imagen img = leerPGM(rutaIn);
        auto mascara = crearMascara(filtro, k);
        std::vector<float> salida(img.px.size());   // reserva fuera de la medición

        // --- Calentamiento (no se registra) ---
        convolucionCPU(img.px.data(), salida.data(), img.ancho, img.alto, mascara.data(), k);

        // --- Repeticiones medidas ---
        std::vector<double> tiempos;
        for (int rep = 1; rep <= reps; ++rep) {
            auto t0 = std::chrono::steady_clock::now();
            convolucionCPU(img.px.data(), salida.data(), img.ancho, img.alto, mascara.data(), k);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            tiempos.push_back(ms);
            escribirCSV(csv, "cpu", filtro, img.ancho, img.alto, k, 0, 0, rep, "total", ms);
        }

        // --- Salidas (NO se miden) ---
        guardarRAW(prefijo + ".f32", salida);
        guardarPGM(prefijo + ".pgm", salida, img.ancho, img.alto);

        std::printf("CPU %s k=%d %dx%d | mediana %.2f ms (%d reps)\n",
                    filtro.c_str(), k, img.ancho, img.alto, mediana(tiempos), reps);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
