# 🧠 CNN — Classificação de Imagens com Keras

Rede Neural Convolucional (CNN) treinada no dataset **MNIST** para reconhecimento de dígitos manuscritos (0–9), implementada com **Keras + TensorFlow**.

---

## 📋 Descrição

Este projeto faz parte da formação **Machine Learning Specialist** e demonstra na prática como construir, treinar e avaliar uma CNN para classificação de imagens.

O modelo recebe imagens 28×28 pixels em escala de cinza e classifica cada uma em um dos 10 dígitos possíveis (0 a 9).

---

## 🏗️ Arquitetura da Rede

```
Input (28×28×1)
    │
    ├─ Conv2D(32, 5×5, relu) — padding same
    ├─ Conv2D(64, 5×5, relu) — padding same
    ├─ MaxPooling2D(2×2)
    ├─ Dropout(0.25)
    ├─ Flatten
    ├─ Dense(128, relu)
    ├─ Dropout(0.50)
    └─ Dense(10, softmax)   ← saída com 10 classes
```

| Componente | Detalhe |
|---|---|
| Otimizador | Adam (lr inicial = 0.001) |
| Função de perda | Categorical Crossentropy |
| Métrica | Accuracy |
| Callback | ReduceLROnPlateau (patience=3, factor=0.5) |

---

## 📊 Resultados Obtidos

Treinamento com **10 épocas** e **batch size 32** no dataset MNIST:

| Época | Acc Treino | Acc Validação | Loss Treino | Loss Validação |
|:-----:|:----------:|:-------------:|:-----------:|:--------------:|
| 1     | 93.91%     | 98.38%        | 0.2062      | 0.0544         |
| 2     | 97.54%     | 98.83%        | 0.0843      | 0.0423         |
| 3     | 98.25%     | 98.97%        | 0.0596      | 0.0378         |
| 4     | 98.52%     | 99.02%        | 0.0493      | 0.0333         |
| 5     | 98.77%     | 99.15%        | 0.0386      | 0.0350         |
| 6     | 98.95%     | 99.13%        | 0.0343      | 0.0355         |
| 7     | 99.08%     | 99.22%        | 0.0293      | 0.0322         |
| 8     | 99.14%     | 99.23%        | 0.0277      | 0.0367         |
| 9     | 99.21%     | 99.21%        | 0.0262      | 0.0332         |
| 10    | 99.20%     | 99.12%        | 0.0257      | 0.0332         |

### 🏆 Resultado Final no Conjunto de Teste

| Métrica | Valor |
|---|---|
| **Acurácia** | **99.20%** |
| Loss | 0.0312 |

> Na época 10, o `ReduceLROnPlateau` reduziu automaticamente o learning rate de `0.001` → `0.0005`.

---

## 📁 Estrutura do Projeto

```
Programando uma rede de Deep Learning/
│
├── cnn_classificacao_imagens.py   # Script principal
├── exemplo.py                     # Arquivo original (rascunho)
├── README.md                      # Este arquivo
│
└── resultados/                    # Gerado automaticamente ao rodar
    ├── resultados.txt             # Métricas por época + resultado final
    └── grafico_acuracia.png       # Gráfico de acurácia treino vs validação
```

---

## ▶️ Como Executar

### Pré-requisitos

```bash
pip install tensorflow keras matplotlib
```

### Rodando o script

```bash
python cnn_classificacao_imagens.py
```

Ao executar, o script irá automaticamente:
1. Baixar o dataset MNIST (na primeira execução)
2. Treinar a CNN por 10 épocas
3. Avaliar no conjunto de teste
4. Criar a pasta `resultados/` com o `.txt` de métricas e o gráfico `.png`

---

## 🛠️ Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| `TensorFlow / Keras` | Construção e treinamento da CNN |
| `Matplotlib` | Geração do gráfico de acurácia |
| `os` / `datetime` | Gerenciamento de arquivos e timestamp |

---

## 📚 Conceitos Abordados

- Redes Neurais Convolucionais (CNN)
- Normalização de pixels (escalonamento 0–1)
- One-hot encoding de rótulos
- Regularização com Dropout
- Redução adaptativa do learning rate (`ReduceLROnPlateau`)
- Avaliação de modelos (loss, accuracy, validation split)

---

## 📖 Referências

- [Documentação Keras](https://keras.io/)
- [Dataset MNIST](http://yann.lecun.com/exdb/mnist/)
- Formação Machine Learning Specialist
