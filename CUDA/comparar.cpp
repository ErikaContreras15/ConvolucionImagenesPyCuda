
// comparar.cpp — Validación numérica CPU vs GPU

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>


// leer: carga un archivo .f32 completo en un vector de float.

std::vector<float> leer(const std::string& ruta) {
    std::ifstream f(ruta, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "No se pudo abrir %s\n", ruta.c_str()); std::exit(1); }
    size_t bytes = f.tellg(); f.seekg(0);
    std::vector<float> v(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), bytes);
    return v;
}

int main(int argc, char** argv) {
    // --- Argumentos: dos archivos obligatorios y tolerancia---
    if (argc < 3) { std::fprintf(stderr, "Uso: %s <cpu.f32> <gpu.f32> [tolerancia]\n", argv[0]); return 1; }
    double tol = argc > 3 ? std::stod(argv[3]) : 1e-2;

    // a = salida CPU (referencia), b = salida GPU
    auto a = leer(argv[1]), b = leer(argv[2]);
    // Si los tamaños difieren, se compararon imágenes distintas: no tiene sentido seguir.
    if (a.size() != b.size()) { std::fprintf(stderr, "Tamaños distintos\n"); return 1; }

    // --- Comparación píxel por píxel ---
    // maxErr: mayor diferencia encontrada     posMax: índice 1D donde ocurrió
    // sumErr: suma de diferencias (para la media)   fuera: píxeles con error > tol
    double maxErr = 0, sumErr = 0;
    size_t posMax = 0, fuera = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double e = std::fabs(static_cast<double>(a[i]) - b[i]);
        sumErr += e;
        if (e > maxErr) { maxErr = e; posMax = i; }
        if (e > tol) ++fuera;
    }

    // --- Reporte ---
    // posMax es índice 1D: fila = posMax / ancho, columna = posMax % ancho.
    // Error medio = sumErr / número de píxeles.
    std::printf("Pixeles: %zu\nError absoluto maximo: %.3e (posicion %zu)\n"
                "Error absoluto medio:  %.3e\nTolerancia: %.1e -> %s (%zu pixeles fuera)\n",
                a.size(), maxErr, posMax, sumErr / a.size(), tol,
                fuera == 0 ? "OK" : "FALLA", fuera);
    return fuera == 0 ? 0 : 1;
}