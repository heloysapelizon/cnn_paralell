# CNN em MNIST — paralelizada com OpenMP

- Integrantes: Heloysa Pelizon & Marina Bon

Arquitetura: `Conv(1→6,5x5) → ReLU → MaxPool2x2 → Conv(6→16,5x5) → ReLU →
MaxPool2x2 → FC(256→120) → ReLU → FC(120→84) → ReLU → FC(84→10) →
Softmax+CrossEntropy`.

## Onde está o paralelismo

O treino processa o batch amostra por amostra, e cada amostra passa pela
rede inteira (forward + backward) de forma independente das outras — esse
é o laço que domina o tempo de execução (confirmado com profiling, ver
`results/profile_seq.txt`) e o que foi paralelizado.

Cada thread acumula seu próprio gradiente (`LeNetGrad`) em `train.cpp`,
sem lock, e só depois que todas as threads terminam o batch é que os
gradientes são somados e os pesos atualizados — esse é o único ponto de
sincronização, porque a atualização de pesos de uma iteração precisa
estar pronta antes da próxima começar (SGD é sequencial entre iterações
por natureza).

O `Makefile` permite escolher a política de escalonamento do
`#pragma omp for` (`static`, `dynamic` ou `guided`) na compilação — ver
"Trocar a política de escalonamento" abaixo.

## Estrutura do projeto

```
cnn_paralell/
├── cpp/
│   ├── layers.hpp/.cpp    — forward/backward de cada camada (Conv, Pool, FC, ReLU)
│   ├── network.hpp/.cpp   — monta as camadas na arquitetura LeNet-like
│   ├── mnist.hpp/.cpp     — carrega o binário gerado por prepare_mnist.py
│   └── train.cpp          — laço de treino paralelo (OpenMP) + medição de tempo
├── python/
│   ├── prepare_mnist.py   — baixa o MNIST e converte para o binário que mnist.cpp lê
│   ├── gradient_check.py  — valida o backward por diferenças finitas (checagem numérica)
│   └── classify.ipynb     — carrega os pesos treinados e mostra previsões nos dígitos de teste
├── results/               — saídas brutas das medições de desempenho (escalabilidade, profiling)
├── analyze/               — notebooks que processam results/ e geram os gráficos do relatório
└── data/                  — arquivos baixados/gerados pelo prepare_mnist.py (não versionado)
```

## Como rodar

```bash
python3 python/prepare_mnist.py   # baixa o MNIST e gera data/train.bin, data/test.bin
make all                          # compila ./train com OpenMP
make test                         # roda a checagem de gradiente em Python (opcional)
./train --ref-size 5000 --batch 64 --iters 3500
```

Argumentos aceitos por `./train`: `--data-dir`, `--model-out`, `--ref-size`,
`--batch`, `--iters`, `--lr`, `--seed` (todos com valor padrão, ver início
de `cpp/train.cpp`).

O número de threads é controlado pela variável de ambiente
`OMP_NUM_THREADS`:

```bash
OMP_NUM_THREADS=4 ./train --ref-size 5000 --batch 64 --iters 3500
```

Sem essa variável, o runtime OpenMP decide sozinho (geralmente o total de
núcleos lógicos da máquina).

### Baseline sequencial

Para comparar com a versão paralela, compile o mesmo `train.cpp` sem
`-fopenmp` (não é um arquivo separado):

```bash
make clean && make all OMPFLAG=
./train --ref-size 5000 --batch 64 --iters 3500
```

Nesse binário os `#pragma omp` são ignorados e o laço roda sequencial;
`OMP_NUM_THREADS` não tem efeito.

### Trocar a política de escalonamento

```bash
make clean && make all                                  # static (default)
make clean && make all SCHEDFLAG=-DSCHED_POLICY=dynamic  # dynamic
make clean && make all SCHEDFLAG=-DSCHED_POLICY=guided   # guided
```

A política fica fixa no binário — trocar exige recompilar.

### macOS

O clang do sistema não suporta `-fopenmp`, então é preciso um GCC de
verdade:

```bash
brew install gcc   # instala g++-<versão>, ex. g++-16
```

O `Makefile` já detecta macOS e usa `g++-16` por padrão. Se a versão
instalada for outra, ajuste com `make all CXX=g++-17`.

## Saída do `./train`

Além do tempo total (`tempo_total`), o binário imprime quanto tempo foi
gasto em cada parte do laço de treino — útil para saber onde o tempo
está indo e para comparar com o cálculo de speed-up:

- `threads_usadas` — threads OpenMP realmente usadas (sempre `1` no
  binário sequencial)
- `tempo_batch_loop` — o laço de amostras completo (região paralela + a
  soma serial dos gradientes ao final)
- `tempo_parallel_region` — só o que roda de fato em paralelo
- `tempo_reduction` — a soma serial dos gradientes de cada thread
- `tempo_zero_grad` / `tempo_sgd_update` — partes sequenciais do laço
- `modelo_salvo` — caminho onde os pesos foram salvos (`--model-out`)

## Visualização

`train.cpp` salva os pesos treinados em `model.bin` ao final. O notebook
`python/classify.ipynb` carrega esses pesos junto com o conjunto de teste
e mostra dígitos ao lado da previsão do modelo:

```bash
pip3 install --user matplotlib jupyter
jupyter notebook python/classify.ipynb
```
