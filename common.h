
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Imagen {
    int ancho = 0, alto = 0;
    std::vector<float> px;   // escala de grises, fila por fila (row-major)
};

// ---------- Lectura / escritura PGM (P5, 8 bits) ----------
inline void saltarComentarios(std::ifstream& f) {
    f >> std::ws;
    while (f.peek() == '#') { std::string l; std::getline(f, l); f >> std::ws; }
}

inline Imagen leerPGM(const std::string& ruta) {
    std::ifstream f(ruta, std::ios::binary);
    if (!f) throw std::runtime_error("No se pudo abrir " + ruta);
    std::string magic; f >> magic;
    if (magic != "P5") throw std::runtime_error("Se esperaba PGM binario (P5)");
    int w, h, maxv;
    saltarComentarios(f); f >> w;
    saltarComentarios(f); f >> h;
    saltarComentarios(f); f >> maxv;
    f.get();  // un solo espacio en blanco antes de los datos
    if (maxv != 255) throw std::runtime_error("Solo se admite maxval = 255");
    std::vector<uint8_t> buf(static_cast<size_t>(w) * h);
    f.read(reinterpret_cast<char*>(buf.data()), buf.size());
    if (!f) throw std::runtime_error("Archivo PGM incompleto");
    Imagen img;
    img.ancho = w; img.alto = h;
    img.px.assign(buf.begin(), buf.end());
    return img;
}

// Conversión al rango de visualización: redondeo + recorte a [0, 255].
// (Los valores negativos del laplaciano quedan en 0
inline void guardarPGM(const std::string& ruta, const std::vector<float>& d, int w, int h) {
    std::ofstream f(ruta, std::ios::binary);
    f << "P5\n" << w << " " << h << "\n255\n";
    std::vector<uint8_t> buf(d.size());
    for (size_t i = 0; i < d.size(); ++i)
        buf[i] = static_cast<uint8_t>(std::clamp(std::lround(d[i]), 0L, 255L));
    f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
}

// Salida cruda en float (sin recortar) para la validación numérica CPU vs GPU.
inline void guardarRAW(const std::string& ruta, const std::vector<float>& d) {
    std::ofstream f(ruta, std::ios::binary);
    f.write(reinterpret_cast<const char*>(d.data()), d.size() * sizeof(float));
}

// ---------- Máscaras ----------
// filtro: "media" (k libre, impar), "nitidez" o "laplaciano" (k = 3 forzado)
inline std::vector<float> crearMascara(const std::string& filtro, int& k) {
    if (filtro == "media") {
        if (k < 1 || k % 2 == 0) throw std::runtime_error("k debe ser impar");
        return std::vector<float>(k * k, 1.0f / (k * k));
    }
    k = 3;
    if (filtro == "nitidez")    return { 0, -1,  0,  -1, 5, -1,   0, -1,  0 };
    if (filtro == "laplaciano") return { 0,  1,  0,   1, -4, 1,   0,  1,  0 };
    throw std::runtime_error("Filtro desconocido: " + filtro);
}

// ---------- CSV ----------
// Columnas: implementacion,filtro,ancho,alto,k,hilos_bloque,bloques,repeticion,fase,tiempo_ms
inline void escribirCSV(const std::string& ruta, const std::string& impl,
                        const std::string& filtro, int w, int h, int k,
                        int hilos, int bloques, int rep,
                        const std::string& fase, double ms) {
    bool nuevo;
    { std::ifstream t(ruta); nuevo = !t.good() || t.peek() == std::ifstream::traits_type::eof(); }
    std::ofstream f(ruta, std::ios::app);
    if (nuevo)
        f << "implementacion,filtro,ancho,alto,k,hilos_bloque,bloques,repeticion,fase,tiempo_ms\n";
    f << impl << ',' << filtro << ',' << w << ',' << h << ',' << k << ','
      << hilos << ',' << bloques << ',' << rep << ',' << fase << ',' << ms << '\n';
}

inline double mediana(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
