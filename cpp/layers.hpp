#ifndef LAYERS_HPP
#define LAYERS_HPP

#include <vector>
#include <cstddef>

using std::size_t;
using Vecf = std::vector<float>;

/* ---------- Convolucao 2D (valid, stride 1) ---------- */

struct ConvParams {
    Vecf W;   /* [outC][inC][kh][kw] */
    Vecf b;   /* [outC] */
    int inC, outC, kh, kw;
};

struct ConvGrad {
    Vecf dW;
    Vecf db;
    void zero(const ConvParams &p) {
        dW.assign(p.W.size(), 0.0f);
        db.assign(p.b.size(), 0.0f);
    }
};

struct ConvCache {
    Vecf input;      /* copia da entrada, necessaria no backward */
    int inH, inW;
};

ConvParams conv_init(int inC, int outC, int kh, int kw, unsigned seed);

void conv_forward(const ConvParams &p, const Vecf &input, int inH, int inW,
                   Vecf &output, int &outH, int &outW, ConvCache &cache);

/* grad_output tem shape [outC][outH][outW]. Acumula (+=) em grad (dW, db) e
 * escreve o gradiente em relacao a entrada em grad_input. */
void conv_backward(const ConvParams &p, const ConvCache &cache,
                    const Vecf &grad_output, int outH, int outW,
                    Vecf &grad_input, ConvGrad &grad);

/* ---------- ReLU ---------- */

void relu_forward(const Vecf &input, Vecf &output);
void relu_backward(const Vecf &input, const Vecf &grad_output, Vecf &grad_input);

/* ---------- MaxPool 2x2 stride 2 ---------- */

struct PoolCache {
    std::vector<int> argmax; /* indice (dentro da janela 2x2) do maximo, por posicao de saida */
};

void maxpool2x2_forward(const Vecf &input, int C, int H, int W,
                         Vecf &output, int &outH, int &outW, PoolCache &cache);

void maxpool2x2_backward(const PoolCache &cache, const Vecf &grad_output,
                          int C, int H, int W, int outH, int outW,
                          Vecf &grad_input);

/* ---------- Fully connected (dense) ---------- */

struct DenseParams {
    Vecf W;  /* [out][in] */
    Vecf b;  /* [out] */
    int in, out;
};

struct DenseGrad {
    Vecf dW;
    Vecf db;
    void zero(const DenseParams &p) {
        dW.assign(p.W.size(), 0.0f);
        db.assign(p.b.size(), 0.0f);
    }
};

struct DenseCache {
    Vecf input;
};

DenseParams dense_init(int in, int out, unsigned seed);

void dense_forward(const DenseParams &p, const Vecf &input, Vecf &output, DenseCache &cache);

void dense_backward(const DenseParams &p, const DenseCache &cache,
                     const Vecf &grad_output, Vecf &grad_input, DenseGrad &grad);

/* ---------- Softmax + cross-entropy ---------- */

/* Retorna a perda (escalar) e escreve em grad_input o gradiente d(loss)/d(logits). */
float softmax_cross_entropy(const Vecf &logits, int label, Vecf &grad_input);

/* argmax dos logits, para calcular acuracia */
int argmax(const Vecf &v);

#endif /* LAYERS_HPP */
