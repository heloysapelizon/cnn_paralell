# CNN (LeNet-like) em MNIST — versão sequencial

## Status

**Ainda sem paralelização.** Esta é a versão sequencial, usada como
baseline correto (e de referência de tempo) antes de introduzir OpenMP.

## O que é

Uma CNN pequena, estilo LeNet-5, treinada em MNIST, implementada do zero em
C++ (convolução, max-pooling, camadas densas e backprop escritos à mão —
sem BLAS, sem framework de ML).

Arquitetura: `Conv(1→6,5x5) → ReLU → MaxPool2x2 → Conv(6→16,5x5) → ReLU →
MaxPool2x2 → FC(256→120) → ReLU → FC(120→84) → ReLU → FC(84→10) →
Softmax+CrossEntropy`.

## Onde vai entrar a dependência real entre as partes (próximo passo)

Ao contrário de um problema embaraçosamente paralelo (ex.: busca por força
bruta), o treino de uma rede neural tem dependência genuína em pontos que
vão aparecer quando a paralelização for adicionada:

1. **Entre camadas** — dentro de uma amostra, cada camada só pode rodar
   depois que a anterior terminou (forward), e o backward percorre as
   camadas na ordem inversa.
2. **Redução de gradientes** — se o batch for paralelizado por amostra,
   cada thread vai acumular seu próprio gradiente local, mas o *update*
   dos pesos só pode acontecer depois que os gradientes de **todas** as
   threads forem somados — ponto de sincronização/dependência explícito.
3. **Entre iterações** — os pesos atualizados na iteração *t* são a
   entrada compartilhada da iteração *t+1* (dependência sequencial
   clássica de SGD), independente de paralelização.

`network.hpp`/`layers.hpp` já foram desenhados sem estado mutável
compartilhado nas camadas (cache e gradiente passados explicitamente),
justamente para que o paralelismo por amostra possa ser adicionado depois
sem precisar reescrever a lógica de forward/backward.

## Estrutura dos arquivos

Só o que participa do treino em si (o que vai ser paralelizado com
OpenMP) fica em C++, em `cpp/`. O resto — preparo de dados e validação
numérica — é Python, em `python/`.

```
cnn_paralell/
├── cpp/
│   ├── layers.hpp/.cpp    — forward/backward de cada camada, como funções puras
│   │                        (cache e gradiente passados explicitamente) — sem
│   │                        estado mutável compartilhado, preparado para paralelizar.
│   ├── network.hpp/.cpp   — compõe as camadas na arquitetura LeNet-like.
│   ├── mnist.hpp/.cpp     — loader do binário raw gerado por prepare_mnist.py
│   │                        (um único fread, sem parsing de formato externo).
│   └── train.cpp          — treino sequencial + medição de tempo.
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

## Próximo passo

Paralelizar `train.cpp` com OpenMP (por amostra dentro do batch, com
redução dos gradientes antes do update) e então montar o benchmark de
4.1 (escalabilidade forte) e 4.2 (escalabilidade fraca).
