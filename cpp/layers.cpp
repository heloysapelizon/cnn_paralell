#include "layers.hpp"
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

/* ---------------- indexacao auxiliar ---------------- */
static inline size_t idx3(int c, int h, int w, int H, int W) {
    return (size_t)c * H * W + (size_t)h * W + w;
}
static inline size_t idxW4(int oc, int ic, int kh_, int kw_, int inC, int KH, int KW) {
    return (((size_t)oc * inC + ic) * KH + kh_) * KW + kw_;
}

/* ---------------- Conv2D ---------------- */

ConvParams conv_init(int inC, int outC, int kh, int kw, unsigned seed) {
    ConvParams p;
    p.inC = inC; p.outC = outC; p.kh = kh; p.kw = kw;
    p.W.resize((size_t)outC * inC * kh * kw);
    p.b.assign(outC, 0.0f);

    /* inicializacao He, adequada para camadas seguidas de ReLU */
    float fan_in = (float)(inC * kh * kw);
    float std_dev = std::sqrt(2.0f / fan_in);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (auto &v : p.W) v = dist(rng);
    return p;
}

void conv_forward(const ConvParams &p, const Vecf &input, int inH, int inW,
                   Vecf &output, int &outH, int &outW, ConvCache &cache) {
    outH = inH - p.kh + 1;
    outW = inW - p.kw + 1;
    output.assign((size_t)p.outC * outH * outW, 0.0f);

    cache.input = input;
    cache.inH = inH;
    cache.inW = inW;

    for (int oc = 0; oc < p.outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float acc = p.b[oc];
                for (int ic = 0; ic < p.inC; ic++) {
                    for (int kh_ = 0; kh_ < p.kh; kh_++) {
                        for (int kw_ = 0; kw_ < p.kw; kw_++) {
                            float x = input[idx3(ic, oh + kh_, ow + kw_, inH, inW)];
                            float w = p.W[idxW4(oc, ic, kh_, kw_, p.inC, p.kh, p.kw)];
                            acc += x * w;
                        }
                    }
                }
                output[idx3(oc, oh, ow, outH, outW)] = acc;
            }
        }
    }
}

void conv_backward(const ConvParams &p, const ConvCache &cache,
                    const Vecf &grad_output, int outH, int outW,
                    Vecf &grad_input, ConvGrad &grad) {
    int inH = cache.inH, inW = cache.inW;
    grad_input.assign((size_t)p.inC * inH * inW, 0.0f);
    /* grad.dW / grad.db ja devem ter sido zerados (ou nao) pelo chamador —
     * aqui SEMPRE acumulamos (+=), permitindo somar gradientes de varias
     * amostras do mesmo batch/thread antes da reducao final. */

    for (int oc = 0; oc < p.outC; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float go = grad_output[idx3(oc, oh, ow, outH, outW)];
                grad.db[oc] += go;
                for (int ic = 0; ic < p.inC; ic++) {
                    for (int kh_ = 0; kh_ < p.kh; kh_++) {
                        for (int kw_ = 0; kw_ < p.kw; kw_++) {
                            size_t xi = idx3(ic, oh + kh_, ow + kw_, inH, inW);
                            size_t wi = idxW4(oc, ic, kh_, kw_, p.inC, p.kh, p.kw);
                            grad.dW[wi] += go * cache.input[xi];
                            grad_input[xi] += go * p.W[wi];
                        }
                    }
                }
            }
        }
    }
}

/* ---------------- ReLU ---------------- */

void relu_forward(const Vecf &input, Vecf &output) {
    output.resize(input.size());
    for (size_t i = 0; i < input.size(); i++)
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}

void relu_backward(const Vecf &input, const Vecf &grad_output, Vecf &grad_input) {
    grad_input.resize(input.size());
    for (size_t i = 0; i < input.size(); i++)
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
}

/* ---------------- MaxPool 2x2 stride 2 ---------------- */

void maxpool2x2_forward(const Vecf &input, int C, int H, int W,
                         Vecf &output, int &outH, int &outW, PoolCache &cache) {
    outH = H / 2;
    outW = W / 2;
    output.assign((size_t)C * outH * outW, 0.0f);
    cache.argmax.assign((size_t)C * outH * outW, 0);

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int base_h = oh * 2, base_w = ow * 2;
                float best = -1e30f;
                int best_k = 0;
                for (int k = 0; k < 4; k++) {
                    int dh = k / 2, dw = k % 2;
                    float v = input[idx3(c, base_h + dh, base_w + dw, H, W)];
                    if (v > best) { best = v; best_k = k; }
                }
                output[idx3(c, oh, ow, outH, outW)] = best;
                cache.argmax[idx3(c, oh, ow, outH, outW)] = best_k;
            }
        }
    }
}

void maxpool2x2_backward(const PoolCache &cache, const Vecf &grad_output,
                          int C, int H, int W, int outH, int outW,
                          Vecf &grad_input) {
    grad_input.assign((size_t)C * H * W, 0.0f);
    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int k = cache.argmax[idx3(c, oh, ow, outH, outW)];
                int dh = k / 2, dw = k % 2;
                int base_h = oh * 2, base_w = ow * 2;
                grad_input[idx3(c, base_h + dh, base_w + dw, H, W)] +=
                    grad_output[idx3(c, oh, ow, outH, outW)];
            }
        }
    }
}

/* ---------------- Dense (fully connected) ---------------- */

DenseParams dense_init(int in, int out, unsigned seed) {
    DenseParams p;
    p.in = in; p.out = out;
    p.W.resize((size_t)out * in);
    p.b.assign(out, 0.0f);

    float std_dev = std::sqrt(2.0f / (float)in);
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, std_dev);
    for (auto &v : p.W) v = dist(rng);
    return p;
}

void dense_forward(const DenseParams &p, const Vecf &input, Vecf &output, DenseCache &cache) {
    cache.input = input;
    output.assign(p.out, 0.0f);
    for (int o = 0; o < p.out; o++) {
        float acc = p.b[o];
        const float *wrow = &p.W[(size_t)o * p.in];
        for (int i = 0; i < p.in; i++) acc += wrow[i] * input[i];
        output[o] = acc;
    }
}

void dense_backward(const DenseParams &p, const DenseCache &cache,
                     const Vecf &grad_output, Vecf &grad_input, DenseGrad &grad) {
    grad_input.assign(p.in, 0.0f);
    for (int o = 0; o < p.out; o++) {
        float go = grad_output[o];
        grad.db[o] += go;
        const float *wrow = &p.W[(size_t)o * p.in];
        float *dwrow = &grad.dW[(size_t)o * p.in];
        for (int i = 0; i < p.in; i++) {
            dwrow[i] += go * cache.input[i];
            grad_input[i] += go * wrow[i];
        }
    }
}

/* ---------------- Softmax + cross-entropy ---------------- */

float softmax_cross_entropy(const Vecf &logits, int label, Vecf &grad_input) {
    float maxv = *std::max_element(logits.begin(), logits.end());
    Vecf exps(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        exps[i] = std::exp(logits[i] - maxv);
        sum += exps[i];
    }
    grad_input.resize(logits.size());
    for (size_t i = 0; i < logits.size(); i++) {
        float prob = exps[i] / sum;
        grad_input[i] = prob - ((int)i == label ? 1.0f : 0.0f);
    }
    float loss = -std::log(std::max(exps[label] / sum, 1e-12f));
    return loss;
}

int argmax(const Vecf &v) {
    return (int)(std::max_element(v.begin(), v.end()) - v.begin());
}
