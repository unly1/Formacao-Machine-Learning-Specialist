# 🧠 Desafio — Matriz de Confusão & Métricas de Classificação

> **Formação Machine Learning Specialist**  
> Teoria do Aprendizado Estatístico

---

## 📋 Descrição

Este desafio implementa o cálculo completo das **métricas de avaliação de aprendizado** definidas na Tabela 1 do enunciado, aplicadas a dois modelos de CNN (Convolutional Neural Network) treinados no dataset **MNIST** (classificação de dígitos 0–9).

Para cada classe, a abordagem **One-vs-Rest** é usada para extrair:

| Sigla | Significado |
|-------|-------------|
| **VP** | Verdadeiro Positivo — acertou que **É** da classe |
| **VN** | Verdadeiro Negativo — acertou que **NÃO É** da classe |
| **FP** | Falso Positivo — classificou como classe, mas não é |
| **FN** | Falso Negativo — era da classe, mas não reconheceu |

---

## 📐 Métricas Implementadas (Tabela 1 do Desafio)

| Métrica | Fórmula |
|---------|---------|
| **Sensibilidade** | VP / (VP + FN) |
| **Especificidade** | VN / (FP + VN) |
| **Acurácia** | (VP + VN) / N |
| **Precisão** | VP / (VP + FP) |
| **F-score** | 2 × (Precisão × Sensibilidade) / (Precisão + Sensibilidade) |

---

## 🏗️ Arquitetura dos Modelos

Ambos os modelos são CNNs idênticas em arquitetura (**93.322 parâmetros treináveis**):

```
Conv2D(32, 3×3, relu) → MaxPooling2D(2×2)
Conv2D(64, 3×3, relu) → MaxPooling2D(2×2)
Conv2D(64, 3×3, relu)
Flatten → Dense(64, relu) → Dense(10, softmax)
```

A diferença entre eles está no processo de treinamento:

| | Modelo 1 | Modelo 2 |
|--|----------|----------|
| **Callback** | TensorBoard padrão | TensorBoard + callback de CM por época |
| **Verbosidade** | Normal | `verbose=0` |
| **Diferencial** | Baseline | Salva heatmap da CM a cada época |

---

## 📊 Resultados Obtidos

> TensorFlow `2.21.0` · Dataset MNIST · 5 épocas · Adam optimizer

### Histórico de Treinamento

#### Modelo 1
| Época | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1     | 0.1477 | 0.9536   | 0.0452   | 0.9843       |
| 2     | 0.0483 | 0.9852   | 0.0425   | 0.9870       |
| 3     | 0.0335 | 0.9895   | 0.0465   | 0.9862       |
| 4     | 0.0260 | 0.9915   | 0.0358   | 0.9888       |
| 5     | 0.0202 | 0.9936   | 0.0366   | **0.9889**   |

#### Modelo 2
| Época | Loss   | Accuracy | Val Loss | Val Accuracy |
|-------|--------|----------|----------|--------------|
| 1     | 0.1400 | 0.9569   | 0.0437   | 0.9869       |
| 2     | 0.0444 | 0.9858   | 0.0331   | 0.9899       |
| 3     | 0.0322 | 0.9900   | 0.0355   | 0.9876       |
| 4     | 0.0244 | 0.9924   | 0.0338   | 0.9892       |
| 5     | 0.0196 | 0.9937   | 0.0315   | **0.9907**   |

---

### Métricas Finais — Média Macro (Modelo 1 vs Modelo 2)

| Métrica | Modelo 1 | Modelo 2 | Diferença |
|---------|----------|----------|-----------|
| Sensibilidade | 0.9888 | **0.9907** | +0.0019 |
| Especificidade | 0.9988 | **0.9990** | +0.0002 |
| Acurácia | 0.9978 | **0.9981** | +0.0003 |
| Precisão | 0.9891 | **0.9906** | +0.0015 |
| F-score | 0.9889 | **0.9906** | +0.0017 |

> ✅ O **Modelo 2** superou o Modelo 1 em todas as métricas.

---

## 📁 Estrutura de Arquivos

```
Desafio/
├── confusion_matrix.py              # Script principal
├── README.md                        # Este arquivo
├── log/                             # Logs do TensorBoard
│   └── cm/                          # Imagens da CM por época (Modelo 2)
└── resultados/
    ├── resultados.txt               # Relatório completo em texto
    ├── confusion_matrix_model1.png  # Heatmap — Modelo 1
    ├── confusion_matrix_model2_epoch_1.png
    ├── confusion_matrix_model2_epoch_2.png
    ├── confusion_matrix_model2_epoch_3.png
    ├── confusion_matrix_model2_epoch_4.png
    ├── confusion_matrix_model2_epoch_5.png
    ├── metricas_Modelo_1.png        # Gráfico de barras — métricas M1
    ├── metricas_Modelo_2.png        # Gráfico de barras — métricas M2
    └── comparacao_modelos.png       # Comparação M1 vs M2
```

---

## ▶️ Como Executar

### Pré-requisitos

```bash
pip install tensorflow numpy pandas matplotlib seaborn
```

### Execução

```bash
python confusion_matrix.py
```

Os arquivos são salvos **automaticamente na pasta `Desafio/`**, independente do diretório de trabalho atual, graças ao uso de `os.path.abspath(__file__)`.

### Visualizar logs no TensorBoard

```bash
tensorboard --logdir "Desafio/log"
```

---

## 🔍 Detalhes de Implementação

- **Normalização da CM**: cada linha é dividida pela soma da linha (proporção por classe real)
- **One-vs-Rest**: para classificação multiclasse, cada classe é tratada como problema binário
- **Média Macro**: média simples das métricas de todas as classes (sem ponderação)
- **Callback de CM**: no Modelo 2, a matriz de confusão é recalculada e salva a cada época, permitindo acompanhar a evolução do modelo pelo TensorBoard

---

## 📚 Referências

- [TensorFlow — Confusion Matrix Tutorial](https://www.tensorflow.org/tensorboard/image_summaries)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- Formação Machine Learning Specialist — Teoria do Aprendizado Estatístico
