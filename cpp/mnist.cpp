#include "mnist.hpp"
#include <fstream>
#include <stdexcept>
#include <cstdint>

MnistData load_mnist(const std::string &bin_path, int max_count) {
    std::ifstream f(bin_path, std::ios::binary);
    if (!f) throw std::runtime_error("nao foi possivel abrir " + bin_path);

    uint32_t n, rows, cols;
    f.read(reinterpret_cast<char *>(&n), 4);
    f.read(reinterpret_cast<char *>(&rows), 4);
    f.read(reinterpret_cast<char *>(&cols), 4);
    if (!f) throw std::runtime_error("erro lendo header de " + bin_path);

    uint32_t count = n;
    if (max_count > 0 && (uint32_t)max_count < count) count = (uint32_t)max_count;

    MnistData data;
    data.images.resize(count);
    data.labels.resize(count);

    size_t img_size = rows * cols;
    for (uint32_t i = 0; i < count; i++) {
        data.images[i].resize(img_size);
        f.read(reinterpret_cast<char *>(data.images[i].data()), img_size * sizeof(float));
        if (!f) throw std::runtime_error("EOF inesperado lendo imagens de " + bin_path);
    }
    /* pula as imagens restantes (se count < n) para chegar nos labels */
    if (count < n) f.seekg((size_t)(n - count) * img_size * sizeof(float), std::ios::cur);

    for (uint32_t i = 0; i < count; i++) {
        unsigned char lbl;
        f.read(reinterpret_cast<char *>(&lbl), 1);
        if (!f) throw std::runtime_error("EOF inesperado lendo labels de " + bin_path);
        data.labels[i] = (int)lbl;
    }

    return data;
}
