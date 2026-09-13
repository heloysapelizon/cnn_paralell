# CNN (LeNet-like) em MNIST — paralelizada com OpenMP

## Status

**Paralelizada com OpenMP.** O laço de amostras do batch
(`cpp/train.cpp`, dentro do `for (int it...)`) roda em paralelo; o
mesmo binário compila como baseline sequencial de referência
compilando sem `-fopenmp` (ver "Build sequencial de referência"
abaixo). O profiling (`profile_seq.txt`, gerado com `sample`) e a
instrumentação por eixo (`tempo_batch_loop` na saída do `./train`)
confirmaram esse laço como o único eixo relevante antes de paralelizar
— ver "Eixo de paralelismo" abaixo.

## O que é

Uma CNN pequena, estilo LeNet-5, treinada em MNIST, implementada do zero em
C++ (convolução, max-pooling, camadas densas e backprop escritos à mão —
sem BLAS, sem framework de ML).

Arquitetura: `Conv(1→6,5x5) → ReLU → MaxPool2x2 → Conv(6→16,5x5) → ReLU →
MaxPool2x2 → FC(256→120) → ReLU → FC(120→84) → ReLU → FC(84→10) →
Softmax+CrossEntropy`.

## Dependências reais entre as partes

Ao contrário de um problema embaraçosamente paralelo (ex.: busca por força
bruta), o treino de uma rede neural tem dependência genuína:

1. **Entre camadas** — dentro de uma amostra, cada camada só pode rodar
   depois que a anterior terminou (forward), e o backward percorre as
   camadas na ordem inversa. Por isso a unidade paralelizada é a
   *amostra*, não a camada.
2. **Redução de gradientes** — o batch é paralelizado por amostra, cada
   thread acumulando seu próprio gradiente local, mas o *update* dos
   pesos só pode acontecer depois que os gradientes de **todas** as
   threads forem somados — ponto de sincronização explícito (ver "Eixo
   de paralelismo" abaixo).
3. **Entre iterações** — os pesos atualizados na iteração *t* são a
   entrada compartilhada da iteração *t+1* (dependência sequencial
   clássica de SGD), independente de paralelização.

`network.hpp`/`layers.hpp` foram desenhados sem estado mutável
compartilhado nas camadas (cache e gradiente passados explicitamente),
o que permitiu paralelizar por amostra sem reescrever a lógica de
forward/backward.

## Eixo de paralelismo

O laço candidato foi confirmado por duas medições independentes antes
de escrever qualquer `#pragma`:

- **Profiling** (`profile_seq.txt`, gerado com `sample` no binário
  sequencial): `lenet_forward_backward` e suas chamadas internas
  (`conv_forward`/`conv_backward`, `dense_forward`/`dense_backward`,
  `relu_*`, `maxpool2x2_*`) dominam o "sort by top of stack" quase por
  completo; `lenet_sgd_update` aparece em apenas 7 amostras.
- **Instrumentação por eixo** (`tempo_batch_loop` na saída do
  `./train`, medida com `wtime()`): o laço de amostras consome
  >99% do tempo total; zerar o gradiente e o SGD update são
  desprezíveis.

Paralelização escolhida: **um `LeNetGrad` por thread, sem lock**.
Dentro de `#pragma omp parallel`, cada thread lê seu id
(`omp_get_thread_num()`) e escreve só no seu índice de
`thread_grads`; uma única thread (`#pragma omp single`, com barreira
implícita) redimensiona e zera esse vetor por iteração, usando
`omp_get_num_threads()` — a contagem de threads nunca é consultada
fora da região paralela. O `#pragma omp for` distribui as amostras do
batch; `iter_loss`/`iter_correct` usam `reduction(+:...)`. Depois do
`parallel`, a soma dos gradientes por thread (`LeNetGrad::add`, ver
`network.hpp`) é feita **serialmente**, antes do `lenet_sgd_update` —
esse é o ponto de sincronização do item 2 acima.

## Build sequencial de referência

O binário sequencial de referência é o **mesmo `train.cpp`**,
compilado sem `-fopenmp` — não um arquivo separado:

```bash
make clean && make all OMPFLAG=
```

Sem `-fopenmp`, os `#pragma omp` são ignorados pelo compilador (warning,
não erro) e o código roda sequencial: `tid=0`, `nt=1`, um único
`LeNetGrad` em `thread_grads[0]`, equivalente ao laço sequencial
original. `wtime()` também cai para `std::chrono::steady_clock` nesse
caso, porque `omp_get_wtime()` só linka com `-fopenmp` (mesma semântica
de wall-clock, só muda a fonte do relógio).

O número de threads da versão paralela é controlado pela variável de
ambiente padrão do OpenMP:

```bash
OMP_NUM_THREADS=4 ./train --ref-size 2000 --batch 32 --iters 60
```

Sem `OMP_NUM_THREADS`, o runtime decide sozinho (tipicamente = núcleos
lógicos).

## Estrutura dos arquivos

Só o que participa do treino em si (paralelizado com OpenMP) fica em
C++, em `cpp/`. O resto — preparo de dados e validação numérica — é
Python, em `python/`.

```
cnn_paralell/
├── cpp/
│   ├── layers.hpp/.cpp    — forward/backward de cada camada, como funções puras
│   │                        (cache e gradiente passados explicitamente) — sem
│   │                        estado mutável compartilhado, o que permitiu paralelizar.
│   ├── network.hpp/.cpp   — compõe as camadas na arquitetura LeNet-like.
│   ├── mnist.hpp/.cpp     — loader do binário raw gerado por prepare_mnist.py
│   │                        (um único fread, sem parsing de formato externo).
│   └── train.cpp          — treino paralelo (OpenMP) + medição de tempo
│                             por eixo (ver "Eixo de paralelismo" e
│                             "Saída do ./train" acima/abaixo).
├── python/
│   ├── prepare_mnist.py   — baixa o MNIST (formato IDX original), descompacta
│   │                        e converte para o binário raw que mnist.cpp lê
│   │                        (data/train.bin, data/test.bin).
│   └── gradient_check.py  — reimplementação em NumPy da mesma arquitetura,
│                             valida a matemática do backward por diferenças
│                             finitas (40 checagens, todas OK — ver `make test`).
└── data/                  — arquivos baixados/gerados (não versionado)
```

## Como compilar e validar

```bash
python3 python/prepare_mnist.py   # baixa o MNIST e gera data/train.bin, data/test.bin
make all                           # compila ./train a partir de cpp/
make test                          # roda a checagem de gradiente em Python (40/40 devem passar)
./train --ref-size 2000 --batch 32 --iters 60
```

### Build no macOS

O `g++`/`gcc` do sistema no macOS é o Apple Clang, que não suporta
`-fopenmp`. `train.cpp` já usa `omp_get_wtime()` para a instrumentação
de tempo, então precisa de um GCC de verdade mesmo antes de paralelizar
de fato:

```bash
brew install gcc   # traz g++-<versão>, ex. g++-16
```

O `Makefile` já detecta `Darwin` (macOS) automaticamente e usa
`g++-16 -fopenmp -isysroot $(xcrun --show-sdk-path)` — o `-isysroot`
é necessário porque o GCC do Homebrew não acha `math.h`/`stdio.h`
sozinho no macOS. Se a versão do GCC instalada for outra (`g++-14`,
`g++-17`...), ajuste `CXX` no `Makefile` ou rode:

```bash
make all CXX=g++-17
```

Em Linux o `Makefile` usa `g++` do sistema direto, sem `-isysroot`.

## Saída do `./train`

Além do tempo total (`tempo_total=`), o binário imprime o tempo gasto
em cada eixo do loop de treino (medido com `wtime()`, ver "Build
sequencial de referência"):

- `tempo_batch_loop` — o laço paralelo de amostras (ver "Eixo de
  paralelismo").
- `tempo_zero_grad` — zerar os buffers de gradiente a cada iteração
  (sequencial).
- `tempo_sgd_update` — o update de pesos, `lenet_sgd_update`
  (sequencial: depende do gradiente já reduzido de todas as amostras).

## Próximo passo

Rodar os benchmarks de escalabilidade forte (`OMP_NUM_THREADS`
variando, tamanho do problema fixo) e fraca (tamanho do problema
crescendo com o número de threads), comparando contra o baseline
sequencial de referência (`make all OMPFLAG=`).
