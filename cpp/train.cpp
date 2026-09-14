/*
 * train.cpp
 *
 * Treina uma CNN pequena (LeNet-like) em MNIST. Paralelizado com OpenMP
 * (ver README.md). Uso:
 *   OMP_NUM_THREADS=4 ./train --ref-size N --batch B --iters K --lr LR
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Politica de schedule do #pragma omp for abaixo, parametrizavel via
 * -DSCHED_POLICY=<static|dynamic|guided> na compilacao (ver Makefile,
 * variavel SCHEDFLAG). Default: dynamic. */
#ifndef SCHED_POLICY
#define SCHED_POLICY static
#endif

#include "network.hpp"
#include "mnist.hpp"

static const char *arg_value(int argc, char **argv, const char *flag, const char *def) {
    for (int i = 0; i < argc - 1; i++)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return def;
}

/* omp_get_wtime() so linka com -fopenmp; sem a flag cai para
 * std::chrono (ver README, secao "Build sequencial de referencia"). */
static double wtime() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
#endif
}

int main(int argc, char **argv) {
    std::string data_dir = arg_value(argc, argv, "--data-dir", "data");
    int ref_size = std::atoi(arg_value(argc, argv, "--ref-size", "2000"));
    int batch = std::atoi(arg_value(argc, argv, "--batch", "32"));
    int iters = std::atoi(arg_value(argc, argv, "--iters", "50"));
    float lr = std::atof(arg_value(argc, argv, "--lr", "0.05"));
    unsigned seed = (unsigned)std::atoi(arg_value(argc, argv, "--seed", "42"));

    printf("processadores_logicos=%d\n", (int)std::thread::hardware_concurrency());

    MnistData train = load_mnist(data_dir + "/train.bin", ref_size);
    if ((int)train.images.size() < ref_size) {
        fprintf(stderr, "aviso: ref-size pedido=%d, disponivel=%zu\n", ref_size, train.images.size());
        ref_size = (int)train.images.size();
    }

    LeNetParams params = lenet_init(seed);
    LeNetGrad grad;
    grad.zero(params);

    /* Um LeNetGrad por thread, reduzido apos o parallel for (ver README). */
    std::vector<LeNetGrad> thread_grads;
    int n_threads_used = 1; /* capturado via omp_get_num_threads() dentro do parallel abaixo */

    double total_loss = 0.0;
    long total_correct = 0;
    long total_samples = 0;

    /* Tempo por eixo (ver README, secao "Saida do ./train"). */
    double t_zero_grad = 0.0;
    double t_parallel_region = 0.0; /* so o que esta dentro de #pragma omp parallel */
    double t_reduction = 0.0;       /* soma serial de thread_grads em grad, fora do parallel */
    double t_batch_loop = 0.0;      /* t_parallel_region + t_reduction */
    double t_sgd_update = 0.0;

    double t0 = wtime();

    for (int it = 0; it < iters; it++) {
        double tz0 = wtime();
        /* zera reaproveitando os buffers ja alocados (evita realocar a cada iter) */
        std::fill(grad.conv1.dW.begin(), grad.conv1.dW.end(), 0.0f);
        std::fill(grad.conv1.db.begin(), grad.conv1.db.end(), 0.0f);
        std::fill(grad.conv2.dW.begin(), grad.conv2.dW.end(), 0.0f);
        std::fill(grad.conv2.db.begin(), grad.conv2.db.end(), 0.0f);
        std::fill(grad.fc1.dW.begin(), grad.fc1.dW.end(), 0.0f);
        std::fill(grad.fc1.db.begin(), grad.fc1.db.end(), 0.0f);
        std::fill(grad.fc2.dW.begin(), grad.fc2.dW.end(), 0.0f);
        std::fill(grad.fc2.db.begin(), grad.fc2.db.end(), 0.0f);
        std::fill(grad.fc3.dW.begin(), grad.fc3.dW.end(), 0.0f);
        std::fill(grad.fc3.db.begin(), grad.fc3.db.end(), 0.0f);
        t_zero_grad += wtime() - tz0;

        double iter_loss = 0.0;
        long iter_correct = 0;

        double tb0 = wtime();
        #pragma omp parallel
        {
            #ifdef _OPENMP
            int tid = omp_get_thread_num();
            #else
            int tid = 0;
            #endif

            #pragma omp single
            {
                #ifdef _OPENMP
                int nt = omp_get_num_threads();
                #else
                int nt = 1;
                #endif
                n_threads_used = nt;
                thread_grads.resize(nt);
                for (auto &tg : thread_grads) tg.zero(params);
            }

            #pragma omp for reduction(+:iter_loss,iter_correct) schedule(SCHED_POLICY)
            for (int s = 0; s < batch; s++) {
                int gi = (it * batch + s) % ref_size;
                int predicted;
                float loss = lenet_forward_backward(params, train.images[gi], train.labels[gi],
                                                      thread_grads[tid], predicted);
                iter_loss += loss;
                if (predicted == train.labels[gi]) iter_correct++;
            }
        }
        t_parallel_region += wtime() - tb0;

        double tr0 = wtime();
        for (auto &tg : thread_grads) grad.add(tg);
        t_reduction += wtime() - tr0;

        t_batch_loop += wtime() - tb0;

        double tu0 = wtime();
        lenet_sgd_update(params, grad, lr / (float)batch);
        t_sgd_update += wtime() - tu0;

        total_loss += iter_loss;
        total_correct += iter_correct;
        total_samples += batch;
    }

    double elapsed = wtime() - t0;

    printf("threads_usadas=%d\n", n_threads_used);
    /* printf("ref_size=%d batch=%d iters=%d total_amostras=%ld\n", ref_size, batch, iters, total_samples); */
    printf("tempo=%.6f loss_medio=%.4f acuracia_treino=%.4f\n",
           elapsed, total_loss / total_samples, (double)total_correct / total_samples);
    /* printf("tempo_zero_grad=%.6f (%.2f%%)\n", t_zero_grad, 100.0 * t_zero_grad / elapsed); */
    printf("tempo_batch_loop=%.6f (%.2f%%)  [eixo paralelizavel]\n", t_batch_loop, 100.0 * t_batch_loop / elapsed);
    printf("tempo_parallel_region=%.6f (%.2f%%)  [de fato concorrente]\n",
           t_parallel_region, 100.0 * t_parallel_region / elapsed);
    printf("tempo_reduction=%.6f (%.2f%%)  [sequencial, soma thread_grads]\n",
           t_reduction, 100.0 * t_reduction / elapsed);
    /* printf("tempo_sgd_update=%.6f (%.2f%%)\n", t_sgd_update, 100.0 * t_sgd_update / elapsed); */
    printf("tempo_total=%.6f\n \n", elapsed);

    return 0;
}
