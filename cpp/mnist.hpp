#ifndef MNIST_HPP
#define MNIST_HPP

#include <vector>
#include <string>
#include "layers.hpp"

struct MnistData {
    std::vector<Vecf> images; /* cada uma com 28*28 floats normalizados em [0,1] */
    std::vector<int> labels;
};

/* Le o binario raw gerado por prepare_mnist.py:
 *   uint32 n_samples, uint32 rows, uint32 cols,
 *   float32[n_samples*rows*cols] imagens, uint8[n_samples] labels.
 * max_count <= 0 significa "ler tudo". */
MnistData load_mnist(const std::string &bin_path, int max_count);

#endif /* MNIST_HPP */
