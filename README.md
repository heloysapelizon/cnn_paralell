# CNN (LeNet-like) em MNIST — paralelizada com OpenMP

## Status

**Paralelizada com OpenMP.** O laço de amostras do batch
(`cpp/train.cpp`, dentro do `for (int it...)`) roda em paralelo; o
mesmo binário compila como baseline sequencial de referência
compilando sem `-fopenmp` (ver "Compilar sem paralelismo" abaixo).
O profiling (`profile_seq.txt`, gerado com `sample`) e a
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

**`tempo_batch_loop` inclui essa redução serial** (ela está fora do
`#pragma omp parallel`, mas dentro do intervalo medido). Isso significa
que os ~99% de "eixo paralelizável" reportados por essa métrica
superestimam levemente a fração de fato concorrente — uma parte
pequena, mas real e sequencial, está embutida ali. `tempo_parallel_region`
isola só o que está dentro do `#pragma omp parallel`, e `tempo_reduction`
isola só a soma serial (ver "Saída do ./train" abaixo). Medido
(`--ref-size 5000 --batch 320 --iters 100`), `tempo_reduction` cresce de
forma quase linear com o número de threads (0.01% do tempo total em
p=1 até 0.57% em p=10) — esperado, já que mais threads produzem mais
`LeNetGrad` locais para somar, e cada soma percorre todos os pesos da
rede. Em valores absolutos ainda é pequeno (cada `lenet_forward_backward`
domina o custo), mas é a manifestação concreta de Amdahl *dentro* do
próprio trecho paralelizado.

## Detalhes do build sem `-fopenmp`

Sem a flag, o código roda sequencial: `tid=0`, `nt=1`, um único
`LeNetGrad` em `thread_grads[0]`, equivalente ao laço sequencial
original. `wtime()` também cai para `std::chrono::steady_clock` nesse
caso, porque `omp_get_wtime()` só linka com `-fopenmp` (mesma semântica
de wall-clock, só muda a fonte do relógio). Ver "Compilar sem
paralelismo" acima para o comando.

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
make all                           # compila ./train a partir de cpp/ (com OpenMP)
make test                          # roda a checagem de gradiente em Python (40/40 devem passar)
./train --ref-size 5000 --batch 64 --iters 3500
```

### Compilar com paralelismo (padrão)

`make all` já compila com `-fopenmp` por padrão (variável `OMPFLAG` no
`Makefile`). O número de threads é controlado em tempo de execução pela
variável de ambiente `OMP_NUM_THREADS`:

```bash
make clean && make all
OMP_NUM_THREADS=4 ./train --ref-size 5000 --batch 64 --iters 3500
```

Sem `OMP_NUM_THREADS`, o runtime OpenMP decide sozinho (tipicamente =
núcleos lógicos da máquina).

### Compilar sem paralelismo (baseline sequencial)

Mesmo `train.cpp`, compilado sem `-fopenmp` — não é um arquivo
separado. Basta zerar `OMPFLAG`:

```bash
make clean && make all OMPFLAG=
./train --ref-size 5000 --batch 64 --iters 3500
```

Sem `-fopenmp`, os `#pragma omp` são ignorados pelo compilador
(warning, não erro) e o laço roda sequencial. `OMP_NUM_THREADS` não
tem efeito nesse binário. Use esse build para tirar o T1 (tempo
sequencial de referência) e comparar com as execuções paralelas.

### Escolher a política de scheduling (balanceamento de carga)

A política do `#pragma omp for` que distribui as amostras do batch
entre threads (ver "Eixo de paralelismo") é escolhida na compilação,
via a variável `SCHEDFLAG` do `Makefile` (macro `SCHED_POLICY` em
`train.cpp`). **Default: `static`** — o padrão do OpenMP quando
nenhuma cláusula `schedule()` é especificada: cada thread recebe um
bloco fixo e igual de amostras, decidido uma única vez no início.

```bash
make clean && make all                                  # static (default)
make clean && make all SCHEDFLAG=-DSCHED_POLICY=dynamic  # dynamic
make clean && make all SCHEDFLAG=-DSCHED_POLICY=guided   # guided
```

- **`dynamic`**: cada thread pega uma amostra por vez de uma fila
  compartilhada, em vez de um bloco fixo — ajuda quando threads
  correm em núcleos de velocidades diferentes (comum em CPUs com
  núcleos heterogêneos, ex. Apple Silicon com núcleos de performance
  e eficiência), já que uma thread mais lenta simplesmente processa
  menos amostras, em vez de travar as outras esperando seu bloco fixo
  terminar.
- **`guided`**: como `dynamic`, mas o tamanho dos blocos começa grande
  e diminui progressivamente — menos overhead de sincronização no
  início, ajuste fino no fim.

Cada troca exige recompilar (`make clean && make all ...`) — a política
fica fixa no binário, não é configurável em tempo de execução (diferente
de `OMP_NUM_THREADS`, que é lido pelo runtime a cada execução).

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

- `threads_usadas=` — número real de threads OpenMP usadas no laço
  paralelo (lido via `omp_get_num_threads()` dentro da região
  paralela, capturado uma vez no primeiro `it`). Útil para conferir
  contra `OMP_NUM_THREADS` numa bateria de medições de escalabilidade —
  sempre `1` no binário sequencial de referência (`OMPFLAG=`).

Além do tempo total (`tempo_total=`), o binário imprime o tempo gasto
em cada eixo do loop de treino (medido com `wtime()`, ver "Detalhes do
build sem `-fopenmp`"):

- `tempo_batch_loop` — o laço de amostras completo, incluindo a
  redução serial dos gradientes por thread (ver "Eixo de
  paralelismo"). Soma de `tempo_parallel_region` + `tempo_reduction`.
- `tempo_parallel_region` — só o que está dentro do `#pragma omp
  parallel` (de fato concorrente). Mais preciso que `tempo_batch_loop`
  para estimar a fração paralelizável real.
- `tempo_reduction` — a soma serial de `thread_grads` em `grad`
  (`LeNetGrad::add`), fora da região paralela. Pequeno em valor
  absoluto, mas cresce com o número de threads.
- `tempo_zero_grad` — zerar os buffers de gradiente a cada iteração
  (sequencial).
- `tempo_sgd_update` — o update de pesos, `lenet_sgd_update`
  (sequencial: depende do gradiente já reduzido de todas as amostras).
