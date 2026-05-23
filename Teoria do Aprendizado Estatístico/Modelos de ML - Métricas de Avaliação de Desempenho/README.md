# Métricas de Avaliação - Matriz de Confusão em Deep Learning

Este diretório contém uma implementação prática focada no uso da **Matriz de Confusão** como métrica de avaliação de desempenho para problemas de classificação multiclasse (reconhecimento de dígitos manuscritos utilizando o dataset MNIST), projetada para a **Formação Machine Learning Specialist**.

---

## 🚀 O que foi desenvolvido neste projeto?

O objetivo principal é entender como avaliar modelos de classificação além da acurácia global, analisando erros específicos classe a classe e integrando o acompanhamento desses resultados no **TensorBoard** em tempo real durante o treinamento.

O script principal [confusion_matrix.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Modelos%20de%20ML%20-%20M%C3%A9tricas%20de%20Avalia%C3%A7%C3%A3o%20de%20Desempenho/confusion_matrix.py) e o notebook [Confusion_Matrix.ipynb](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Modelos%20de%20ML%20-%20M%C3%A9tricas%20de%20Avalia%C3%A7%C3%A3o%20de%20Desempenho/Confusion_Matrix.ipynb) realizam os seguintes passos:

1. **Preparação dos Dados (MNIST)**:
   Carrega o dataset clássico do MNIST, realiza o redimensionamento para incluir a dimensão do canal de cor `(28, 28, 1)` e normaliza os pixels de `[0, 255]` para o intervalo `[0, 1]`.

2. **Treinamento e Avaliação Estática (Modelo 1)**:
   * Constrói uma Rede Neural Convolucional (CNN) clássica utilizando camadas `Conv2D`, `MaxPooling2D`, `Flatten` e `Dense`.
   * Treina o modelo por 5 épocas monitorando a perda e a acurácia.
   * Ao término do treinamento, calcula a matriz de confusão sobre o conjunto de teste utilizando `tf.math.confusion_matrix`.
   * Normaliza os valores para representar proporções e plota a matriz usando `seaborn.heatmap`. A imagem é salva em [confusion_matrix_model1.png](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Modelos%20de%20ML%20-%20M%C3%A9tricas%20de%20Avalia%C3%A7%C3%A3o%20de%20Desempenho/resultados/confusion_matrix_model1.png).

3. **Monitoramento Dinâmico com TensorBoard (Modelo 2)**:
   * Instancia uma segunda CNN com a mesma arquitetura.
   * Cria uma função de callback customizada [log_confusion_matrix](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Modelos%20de%20ML%20-%20M%C3%A9tricas%20de%20Avalia%C3%A7%C3%A3o%20de%20Desempenho/confusion_matrix.py#L140) executada via `tf.keras.callbacks.LambdaCallback` ao fim de cada época (`on_epoch_end`).
   * A cada época, a função gera a matriz de confusão no conjunto de teste, salva a imagem em disco (gerando os arquivos `confusion_matrix_model2_epoch_X.png`) e a registra nos logs do TensorBoard (`tf.summary.image`) de forma dinâmica.
   * Permite visualizar a evolução e melhoria do modelo classe por classe ao longo das épocas de treinamento.

4. **Persistência de Resultados**:
   * Todos os sumários dos modelos, históricos de métricas de treinamento (Loss, Accuracy, Val Loss, Val Accuracy) e as matrizes de confusão finais em formato textual são registrados no relatório [resultados.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Modelos%20de%20ML%20-%20M%C3%A9tricas%20de%20Avalia%C3%A7%C3%A3o%20de%20Desempenho/resultados/resultados.txt).

---

## 📊 Arquitetura do Modelo

Ambos os modelos construídos utilizam uma arquitetura CNN sequencial:

* **Conv2D**: 32 filtros (3x3), ativação ReLU.
* **MaxPooling2D**: Pool size (2x2).
* **Conv2D**: 64 filtros (3x3), ativação ReLU.
* **MaxPooling2D**: Pool size (2x2).
* **Conv2D**: 64 filtros (3x3), ativação ReLU.
* **Flatten**: Planificação dos mapas de features para a camada densa.
* **Dense**: 64 neurônios, ativação ReLU.
* **Dense**: 10 neurônios, ativação Softmax (uma para cada dígito de 0 a 9).

---

## 📁 Estrutura de Arquivos e Pastas

```
Modelos de ML - Métricas de Avaliação de Desempenho/
│
├── Confusion_Matrix.ipynb   # Notebook interativo contendo a análise e visualizações
├── confusion_matrix.py       # Script Python completo com o treinamento e geração de logs
├── log/                     # Diretório gerado contendo os logs binários para o TensorBoard
└── resultados/              # Imagens geradas e relatório final consolidado
    ├── resultados.txt       # Arquivo de texto detalhando a evolução de métricas e matrizes
    ├── confusion_matrix_model1.png
    ├── confusion_matrix_model2_epoch_1.png
    ├── confusion_matrix_model2_epoch_2.png
    ├── confusion_matrix_model2_epoch_3.png
    ├── confusion_matrix_model2_epoch_4.png
    └── confusion_matrix_model2_epoch_5.png
```

---

## 🛠️ Como Executar

### 1. Pré-requisitos
Certifique-se de ter as bibliotecas instaladas em seu ambiente virtual:
```bash
pip install tensorflow matplotlib seaborn pandas numpy
```

### 2. Executar o script de treinamento
No terminal (dentro da pasta raiz do projeto), execute:
```bash
python "Teoria do Aprendizado Estatístico/Modelos de ML - Métricas de Avaliação de Desempenho/confusion_matrix.py"
```

### 3. Visualizar no TensorBoard
Para iniciar o dashboard interativo do TensorBoard e visualizar a evolução da Matriz de Confusão em tempo real sob a aba "Images", execute:
```bash
tensorboard --logdir "Teoria do Aprendizado Estatístico/Modelos de ML - Métricas de Avaliação de Desempenho/log"
```

---

## 💡 Principais Insights e Aprendizados

* **Mapeamento de Confusão**: A acurácia global atinge ~99%, mas a matriz de confusão nos mostra detalhes ricos. Por exemplo, o dígito `9` pode ser ocasionalmente confundido com `8` ou `4`.
* **Monitoramento Dinâmico**: Visualizar a matriz de confusão a cada época via TensorBoard é crucial para notar o momento em que o modelo aprende a discernir classes mais difíceis (ex: diferenciar `3` de `8` ou `4` de `9`).
* **Valores Normalizados**: A exibição dos valores normalizados facilita a comparação visual do desempenho entre classes, independente de possíveis desbalanceamentos na distribuição do conjunto de testes.
