#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <string>
#include "layers.hpp"

struct LeNetParams {
    ConvParams conv1;   /* 1  -> 6,  5x5 : 28x28 -> 24x24 */
    ConvParams conv2;   /* 6  -> 16, 5x5 : 12x12 -> 8x8   */
    DenseParams fc1;    /* 256 -> 120 */
    DenseParams fc2;    /* 120 -> 84  */
    DenseParams fc3;    /* 84  -> 10  */
};

struct LeNetGrad {
    ConvGrad conv1, conv2;
    DenseGrad fc1, fc2, fc3;
    void zero(const LeNetParams &p) {
        conv1.zero(p.conv1);
        conv2.zero(p.conv2);
        fc1.zero(p.fc1);
        fc2.zero(p.fc2);
        fc3.zero(p.fc3);
    }
    void add(const LeNetGrad &o) {
        for (size_t i = 0; i < conv1.dW.size(); i++) conv1.dW[i] += o.conv1.dW[i];
        for (size_t i = 0; i < conv1.db.size(); i++) conv1.db[i] += o.conv1.db[i];
        for (size_t i = 0; i < conv2.dW.size(); i++) conv2.dW[i] += o.conv2.dW[i];
        for (size_t i = 0; i < conv2.db.size(); i++) conv2.db[i] += o.conv2.db[i];
        for (size_t i = 0; i < fc1.dW.size(); i++) fc1.dW[i] += o.fc1.dW[i];
        for (size_t i = 0; i < fc1.db.size(); i++) fc1.db[i] += o.fc1.db[i];
        for (size_t i = 0; i < fc2.dW.size(); i++) fc2.dW[i] += o.fc2.dW[i];
        for (size_t i = 0; i < fc2.db.size(); i++) fc2.db[i] += o.fc2.db[i];
        for (size_t i = 0; i < fc3.dW.size(); i++) fc3.dW[i] += o.fc3.dW[i];
        for (size_t i = 0; i < fc3.db.size(); i++) fc3.db[i] += o.fc3.db[i];
    }
};

LeNetParams lenet_init(unsigned seed);

/* Forward + backward de UMA amostra. Acumula (+=) o gradiente em `grad`
 * (que deve ja estar zerado ou acumulando de amostras anteriores do mesmo
 * batch/thread). Retorna a perda (cross-entropy) e escreve em `predicted`
 * a classe prevista (argmax dos logits). Nao possui nenhum estado
 * compartilhado mutavel alem de `p` (somente leitura) -> seguro para
 * chamar concorrentemente de varias threads, desde que cada uma passe
 * seu proprio `grad`. */
float lenet_forward_backward(const LeNetParams &p, const Vecf &image, int label,
                              LeNetGrad &grad, int &predicted);

/* Apenas inferencia (sem gradiente), usada para medir acuracia em teste. */
int lenet_predict(const LeNetParams &p, const Vecf &image);

/* Atualiza os pesos via SGD: p := p - lr * grad (grad ja deve vir
 * normalizado, tipicamente dividido pelo tamanho do batch). */
void lenet_sgd_update(LeNetParams &p, const LeNetGrad &grad, float lr);

/* Salva/carrega os pesos treinados (binario raw, sem cabecalho de
 * shape -- a arquitetura e fixa, entao a ordem dos campos basta).
 * lenet_load_params lanca std::runtime_error se o arquivo nao existir
 * ou estiver truncado. */
void lenet_save_params(const LeNetParams &p, const std::string &path);
LeNetParams lenet_load_params(const std::string &path);

#endif /* NETWORK_HPP */
