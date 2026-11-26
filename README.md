# Detecção de Diabetes por espectroscopia

Classificação binária de espectros (classe 0 = controle, 1 = diabetes) para uma
competição do Kaggle cuja métrica é o **macro F1**.

O projeto tem cinco experimentos, cada um no seu próprio contêiner Docker:

- três redes neurais (MLP) em NumPy puro, idênticas em tudo menos na função de
  ativação das camadas ocultas: **Leaky ReLU**, **GELU** e **SwiGLU**;
- uma **SVM RBF** (scikit-learn) sobre diferenças do espectro;
- uma **CNN 1D** (PyTorch) sobre o espectro e sua 1ª diferença.

Cada um gera um `submission.csv` no formato que o Kaggle aceita (`id,class`).

**Resumo do resultado:** o melhor modelo é a SVM RBF sobre a 5ª diferença do
espectro, com macro-F1 0,9587 ± 0,0204 na validação cruzada. A submissão está
em `outputs/svm_d5/submission.csv`. A maior parte da pontuação vem de um
confundimento na aquisição dos dados, descrito em
[O que limita a pontuação](#o-que-limita-a-pontuação).

## Estrutura

```
common/core.py        dados, pré-processamento, MLP, treino, CV, submissão (MLPs)
leaky_relu/           train.py + Dockerfile
gelu/                 train.py + Dockerfile
swiglu/               train.py + Dockerfile
svm/                  train.py + Dockerfile (scikit-learn)
cnn/                  train.py + Dockerfile (PyTorch, CPU)
docker-compose.yml    os cinco serviços
compare.py            tabela comparando os resultados
dataSets/             train_spectra.csv, test_spectra.csv, submission_template.csv
outputs/<modelo>/     submission.csv, metrics.json, oof_predictions.csv
```

Pastas em `outputs/`:

| Pasta | Conteúdo |
|---|---|
| `leaky_relu`, `gelu`, `swiglu` | as três MLPs |
| `svm` | SVM com a 1ª diferença (padrão do serviço `svm`) |
| `svm_d2` | SVM com a 2ª diferença |
| `svm_d5` | SVM com a 5ª diferença (melhor resultado) |
| `cnn` | CNN com a configuração final |
| `cnn_exp`, `cnn_exp_A`, `cnn_exp_B`, `cnn_exp_C` | busca de configuração da CNN (só validação, sem submissão) |

Os dados **não entram nas imagens Docker** (`.dockerignore`): são montados
somente para leitura na hora de rodar. A base é de uso exclusivo da disciplina;
`dataSets/` e `outputs/` estão no `.gitignore`.

## Rodar

Todos os comandos a partir da raiz do projeto.

### Todos de uma vez

```bash
docker compose up --build
python3 compare.py
```

### Um de cada vez

```bash
docker compose run --rm --build leaky_relu
docker compose run --rm --build gelu
docker compose run --rm --build swiglu
docker compose run --rm --build svm
docker compose run --rm --build cnn
```

### Sem o compose (docker puro)

```bash
# Leaky ReLU
docker build -f leaky_relu/Dockerfile -t diabetes-leaky-relu .
docker run --rm -v "$PWD/dataSets:/data:ro" -v "$PWD/outputs/leaky_relu:/out" diabetes-leaky-relu

# GELU
docker build -f gelu/Dockerfile -t diabetes-gelu .
docker run --rm -v "$PWD/dataSets:/data:ro" -v "$PWD/outputs/gelu:/out" diabetes-gelu

# SwiGLU
docker build -f swiglu/Dockerfile -t diabetes-swiglu .
docker run --rm -v "$PWD/dataSets:/data:ro" -v "$PWD/outputs/swiglu:/out" diabetes-swiglu

# SVM (1ª diferença)
docker build -f svm/Dockerfile -t diabetes-svm .
docker run --rm -v "$PWD/dataSets:/data:ro" -v "$PWD/outputs/svm:/out" diabetes-svm

# SVM com a 5ª diferença (melhor resultado)
docker run --rm -v "$PWD/dataSets:/data:ro" -v "$PWD/outputs/svm_d5:/out" diabetes-svm --diff 5

# CNN
docker build -f cnn/Dockerfile -t diabetes-cnn .
docker run --rm -v "$PWD/dataSets:/data:ro" -v "$PWD/outputs/cnn:/out" diabetes-cnn
```

Tempos aproximados: cada MLP 2 a 3 minutos, a SVM 10 a 30 segundos, a CNN cerca
de 37 minutos em CPU.

### Opções úteis

```bash
# MLPs: confere o backprop contra diferenças finitas (segundos; sai com erro se falhar)
docker compose run --rm leaky_relu --gradcheck
docker compose run --rm gelu --gradcheck
docker compose run --rm swiglu --gradcheck

# gera só a submissão, sem a validação cruzada
docker compose run --rm gelu --skip-cv
```

| Script | Opções |
|---|---|
| `svm/train.py` | `--diff N` (ordem da diferença, 1 a 7, padrão 1), `--C`, `--cv-repeats`, `--skip-cv` |
| `cnn/train.py` | `--inputs {feature,global}`, `--width`, `--kernel`, `--dropout`, `--epochs`, `--lr`, `--weight-decay`, `--batch-size`, `--shift`, `--cv-repeats`, `--ensemble`, `--skip-cv`, `--skip-final` |

### Fora do Docker (opcional)

```bash
python3 gelu/train.py --train dataSets/train_spectra.csv --test dataSets/test_spectra.csv --out outputs/gelu
```

As MLPs só precisam de NumPy; a SVM precisa de scikit-learn e a CNN de PyTorch.

## Dados

- `train_spectra.csv`: 624 espectros, 3736 pontos de 399 a 4000, 300 controles
  e 324 diabéticos.
- `test_spectra.csv`: 364 espectros, sem rótulo.
- Os ids vão de 10000 a 11039, intercalados entre treino e teste.

## Protocolo de validação

1. Validação cruzada estratificada de 5 folds, repetida 3 vezes (2 na CNN),
   **só com o treino**. Todo pré-processamento é reajustado dentro de cada fold.
2. Treino final em todo o conjunto de treino.
3. Só então o teste é lido, previsto e gravado em `submission.csv`.

O conjunto de teste não participa de ajuste, validação nem escolha de
hiperparâmetros. Todos os modelos usam os mesmos folds, o que permite comparar
fold a fold.

## Modelos

### MLPs (`common/core.py`)

**Pré-processamento:** SNV por espectro, padronização por ponto, PCA com 80
componentes, escala unitária.

**Modelo:** entrada 80 -> 64 -> 32 -> 1, dropout 0,2, ruído gaussiano 0,1 nas
entradas, AdamW (lr 2e-3 com decaimento cosseno, weight decay 1e-2), 300 épocas,
lote 32. No SwiGLU cada camada é `swish(xW + b) * (xV + c)` com 2/3 da largura
(43 e 21). A GELU usa a aproximação tanh, porque o NumPy não tem `erf`. A
submissão é a média das probabilidades de 5 modelos com sementes diferentes.

Os hiperparâmetros ficam no dicionário `CONFIG` em `common/core.py` e são os
mesmos para as três ativações.

### SVM (`svm/train.py`)

Diferença de ordem N do espectro (`--diff`, padrão 1), padronização por ponto e
`SVC` com kernel RBF, `C=10`, `gamma="scale"`.

### CNN 1D (`cnn/train.py`)

Dois canais de entrada (espectro e 1ª diferença, padronizados por ponto com
estatísticas do treino), 4 blocos `Conv1d(kernel 9, stride 2) + BatchNorm + GELU`
com 24, 48, 96 e 96 filtros, pooling global (média e máximo), dropout 0,3 e uma
camada linear. AdamW com OneCycleLR (lr 3e-3), 40 épocas, lote 32, deslocamento
aleatório de até 2 pontos como aumento de dados. Submissão pela média de 5 modelos.

## Resultados (validação cruzada no treino)

| Modelo               | macro-F1        | acurácia | tamanho                |
|----------------------|-----------------|----------|------------------------|
| SVM RBF, 5ª diferença | 0,9587 ± 0,0204 | 0,9589   | —                      |
| SVM RBF, 2ª diferença | 0,9502 ± 0,0193 | 0,9503   | —                      |
| SVM RBF, 1ª diferença | 0,9491 ± 0,0167 | 0,9493   | 295 vetores de suporte |
| MLP Leaky ReLU       | 0,9317 ± 0,0295 | 0,9322   | 7297 parâmetros        |
| MLP GELU             | 0,9295 ± 0,0312 | 0,9300   | 7297 parâmetros        |
| CNN 1D               | 0,9194 ± 0,0278 | 0,9199   | 136201 parâmetros      |
| MLP SwiGLU           | 0,9082 ± 0,0380 | 0,9092   | 8836 parâmetros        |

SVM e MLPs: 5 folds x 3 repetições. CNN: 5 folds x 2 repetições (os mesmos
folds das duas primeiras repetições).

- **SVM contra os demais:** a SVM com a 1ª diferença ganha da Leaky ReLU em 12
  dos 15 folds (diferença média 0,018, erro padrão 0,006) e da CNN nos 10 folds
  em comum (0,029 ± 0,005).
- **MLPs:** Leaky ReLU e GELU estão empatadas (diferença média por fold 0,002,
  erro padrão 0,002; as previsões no teste coincidem em 98,6% dos espectros).
  SwiGLU ficou cerca de 0,02 abaixo, perdendo em 12 a 13 dos 15 folds.
- **CNN:** cerca de 0,01 abaixo da Leaky ReLU (erro padrão 0,007), ou seja, não
  dá para separá-las; tem quase 20 vezes mais parâmetros e leva ~37 minutos.
- **Referências baseadas no teorema de Bayes**, na mesma validação: LDA com 150
  componentes 0,912; Naive Bayes gaussiano com 80 componentes 0,830.

Hiperparâmetros e ordem da diferença foram escolhidos olhando essa mesma
validação cruzada, então os números acima são um pouco otimistas.

### Ordem da diferença na SVM

Em 10 repetições dos 5 folds (50 folds), o macro-F1 sobe com a ordem até a 5ª e
depois cai:

| Ordem | macro-F1        | FP por repetição | FN por repetição |
|-------|-----------------|------------------|------------------|
| 1ª    | 0,9446 ± 0,0242 | 17,0             | 17,5             |
| 2ª    | 0,9492 ± 0,0204 | 17,7             | 13,9             |
| 3ª    | 0,9504 ± 0,0193 | 17,9             | 13,0             |
| 4ª    | 0,9523 ± 0,0212 | 17,4             | 12,3             |
| 5ª    | 0,9558 ± 0,0227 | 16,1             | 11,4             |
| 6ª    | 0,9540 ± 0,0223 | 17,1             | 11,5             |
| 7ª    | 0,9482 ± 0,0231 | 20,0             | 12,2             |
| 10ª   | 0,9085 ± 0,0260 | 36,3             | 20,5             |

- A 5ª ganha da 1ª nas 10 repetições (+0,011 ± 0,002 por fold). A 4ª e a 6ª
  ficam quase no mesmo nível, então não é um pico isolado.
- O ganho vem dos falsos negativos; os falsos positivos não mudam.
- Média dos scores das ordens 4, 5 e 6: 0,9550, sem ganho sobre a 5ª sozinha.
- Diferenças de ordem alta destacam a parte de alta frequência do espectro. Só
  o resíduo (espectro menos uma versão suavizada por Savitzky-Golay) já dá
  0,947; só a parte suave dá 0,940.

### Busca de configuração da CNN

Uma repetição dos 5 folds para cada configuração:

| Experimento | Configuração | macro-F1 |
|---|---|---|
| `cnn_exp_A` | padronização por ponto, largura 24, kernel 9, 40 épocas | 0,931 |
| `cnn_exp_C` | padronização por ponto, largura 16, kernel 5, 60 épocas | 0,914 |
| `cnn_exp` | escala global, largura 24, kernel 9, 40 épocas | 0,909 |
| `cnn_exp_B` | escala global, 80 épocas, dropout 0,5, sem deslocamento | 0,903 |

A configuração A virou o padrão. Com 2 repetições ela deu 0,919, abaixo do 0,931
obtido com 1 repetição.

## O que limita a pontuação

As análises desta seção foram feitas sobre o treino, com scripts avulsos que não
fazem parte do repositório. Salvo indicação, usam a SVM com a 1ª diferença e os
mesmos folds.

### Dois grupos de aquisição

O nível de ruído dos espectros (desvio da 2ª diferença entre 1800 e 2300, região
sem bandas fortes) é bimodal, com uma lacuna vazia entre os dois grupos:

| Grupo | Controle | Diabetes |
|---|---|---|
| Baixo ruído (379 espectros) | 62 | 317 |
| Alto ruído (245 espectros) | 238 | 7 |

- A regra "alto ruído = controle, baixo ruído = diabetes", sem modelo, já dá
  macro-F1 0,888. Um modelo que só vê o nível de ruído chega a 0,873.
- O teste tem a mesma estrutura: 140 espectros de alto ruído e 224 de baixo
  (38,5%, contra 39,3% no treino).
- Medir o ruído na janela 2400–2700 dá exatamente a mesma divisão.

Ou seja, boa parte da pontuação vem de um confundimento entre a aquisição e a
classe, e não da assinatura do diabetes no espectro.

### Onde os erros estão

Os 69 espectros "atípicos" (controle de baixo ruído, diabetes de alto ruído) são
os que separam os modelos:

| Modelo | Acerto nos 555 típicos | Acerto nos 69 atípicos |
|---|---|---|
| SVM | 97,8% | 72,5% |
| MLP Leaky ReLU | 96,8% | 68,1% |
| CNN | 97,3% | 50,7% |

Matriz de confusão da SVM por grupo:

| Grupo | VN | FP | FN | VP | Especificidade | Sensibilidade |
|---|---|---|---|---|---|---|
| Baixo ruído | 46 | 16 | 12 | 305 | 74,2% | 96,2% |
| Alto ruído | 238 | 0 | 3 | 4 | 100% | 57,1% |

- Todos os falsos positivos são controles de baixo ruído, minoria de 1 para 5
  dentro do seu grupo.
- 15 espectros são errados pelos cinco modelos, 12 deles controles. O voto da
  maioria entre os cinco modelos dá 0,944, pior que a SVM sozinha.

### O que foi descartado

- **Vazamento pelos ids:** ids vizinhos não têm rótulo nem espectro parecidos.
- **Diferença entre treino e teste:** um classificador não distingue os dois
  conjuntos (AUC 0,49).
- **Faixa espectral:** nenhuma faixa carrega o resultado sozinha; a de 2800–3700
  é a mais informativa (0,933 isolada).

### Falta de dados

- Curva de aprendizado da SVM: 0,907 com 100 espectros, 0,928 com 200, 0,933 com
  300, 0,943 com 400, 0,951 com 499. Ainda sobe.
- Retirando controles de baixo ruído do treino, a especificidade nesse grupo cai
  quase em linha reta:

| Controles de baixo ruído no treino | Especificidade no grupo | macro-F1 global |
|---|---|---|
| 12 | 18,1% | 0,9041 |
| 25 | 47,9% | 0,9284 |
| 37 | 63,2% | 0,9388 |
| 50 (todos) | 76,0% | 0,9491 |

O gargalo é a quantidade de controles reais de baixo ruído.

## Tentativas que não melhoraram

Todas comparadas com a SVM de 1ª diferença (0,9491) nos mesmos folds.

**Limiar de decisão.** O limiar 0 já é praticamente o ótimo. No grupo de baixo
ruído, subir o limiar para +0,3 tira 6 falsos positivos e cria 22 falsos
negativos. Com o limiar escolhido só no treino de cada fold (validação aninhada):
limiar único 0,9464; limiar por grupo 0,9438.

**Modelos cientes do grupo.** Um modelo por grupo de ruído: 0,9516 ± 0,024, mas
o acerto nos atípicos cai de 74,9% para 63,0%. Padronização dentro de cada
grupo: 0,9243. Peso balanceado por classe: idêntico ao atual.

**Suavização.** Média móvel de 11 pontos antes da derivada: 0,9336. Suavizar
piora, o que é coerente com o sinal estar na alta frequência.

**Aumento de dados** (sintéticos só no treino de cada fold):

| Variante | macro-F1 | FP | FN |
|---|---|---|---|
| Sem aumento | 0,9491 | 15,7 | 16,0 |
| Mistura de dois controles de baixo ruído, +50 | 0,9486 | 14,0 | 18,0 |
| Mistura de dois controles de baixo ruído, +250 | 0,9454 | 14,0 | 20,0 |
| Mesmo controle com ruído de outro espectro, +250 | 0,9300 | 12,0 | 31,7 |
| Controle de alto ruído suavizado, com ruído de baixo, +150 | 0,9395 | 16,0 | 21,7 |
| Diabético de baixo ruído com ruído de alto, +150 | 0,9442 | 19,7 | 15,0 |
| Os 4 subgrupos na mesma proporção, +100% | 0,9449 | 14,7 | 19,7 |
| Só o grupo de baixo ruído, +100% | 0,9470 | 14,7 | 18,3 |

Os sintéticos ficam entre dois espectros reais e não trazem informação nova: os
falsos positivos caem pouco e os falsos negativos sobem mais.

**Overfitting.** A SVM acerta 100% do treino e 94,9% na validação, mas
regularizar mais derruba os dois juntos:

| C (gamma padrão) | macro-F1 treino | macro-F1 validação |
|---|---|---|
| 0,3 | 0,891 | 0,882 |
| 1 | 0,929 | 0,904 |
| 3 | 0,987 | 0,927 |
| 10 | 1,000 | 0,949 |
| 30 ou 100 | 1,000 | 0,948 |

Outros valores de gamma não passam de 0,946. PCA com 10 a 80 componentes antes
da SVM dá 0,86 a 0,92; com 150 ou 300 empata em 0,949. Média em blocos de 4
pontos dá 0,944 e piora com blocos maiores (0,888 com 32).

**Outros modelos e atributos** sobre a 1ª diferença:

| Modelo | macro-F1 |
|---|---|
| SVM RBF, 1ª + 2ª diferença | 0,9491 |
| SVM RBF, derivada Savitzky-Golay (janela 7) | 0,9459 |
| Soma de scores SVM + logística + PLS-DA | 0,9403 |
| SVM RBF, espectro bruto + 1ª diferença | 0,9379 |
| Regressão logística L2 (melhor C) | 0,9324 |
| PLS-DA, 10 componentes | 0,9298 |
| Gradient boosting (1 repetição) | 0,9274 |
| SVM linear | 0,9263 |
| ExtraTrees (1 repetição) | 0,9192 |
| LDA com shrinkage | 0,9166 |
| kNN | 0,8217 |

**Pseudo-rótulos.** Simulados só com o treino (63% rotulado, 37% sem rótulo, a
mesma proporção de treino e teste, 20 divisões): base 0,9399; melhor variante
0,9412 (+0,0013 ± 0,0010). Sem ganho. Os espectros do teste não foram usados.

## Limitações

- A pontuação depende do confundimento entre grupo de aquisição e classe. Um
  modelo com macro-F1 0,95 aqui não demonstra que detecta diabetes pelo espectro.
- O sinal aproveitado pelas diferenças de ordem alta está na alta frequência e é
  provavelmente assinatura da aquisição, não bioquímica.
- Com 364 espectros no teste, a nota do Kaggle oscila cerca de ±0,01 só por
  acaso.
- Os erros que sobram são controles e diabéticos de baixo ruído que se
  sobrepõem. Só mais dados reais desse tipo mexem nisso.

## Submeter no Kaggle

Envie o arquivo `outputs/<modelo>/submission.csv` na página da competição
(limite de 2 submissões por dia).

| Arquivo | macro-F1 na validação | Previsto no teste [0, 1] |
|---|---|---|
| `outputs/svm_d5/submission.csv` | 0,9587 | 170 / 194 |
| `outputs/svm_d2/submission.csv` | 0,9502 | 170 / 194 |
| `outputs/svm/submission.csv` | 0,9491 | 171 / 193 |
| `outputs/leaky_relu/submission.csv` | 0,9317 | 161 / 203 |
| `outputs/gelu/submission.csv` | 0,9295 | 162 / 202 |
| `outputs/cnn/submission.csv` | 0,9194 | 155 / 209 |
| `outputs/swiglu/submission.csv` | 0,9082 | 155 / 209 |

A submissão da 5ª diferença difere da de 1ª diferença em 11 dos 364 espectros e
da de 2ª em 8. A de 2ª difere da de 1ª em apenas 3 e teve a mesma nota pública.
O primeiro colocado da competição estava com 0,96 quando este texto foi escrito.

`compare.py` lista só os cinco serviços do compose; as pastas `svm_d2` e
`svm_d5` não aparecem nele.
