# Redes de Classificação para Deep Learning — InceptionV3

> **Observação:** Baseado no tutorial original de Magnus Erik Hvass Pedersen, reescrito para funcionar com TF 2.x usando `tf.keras.applications.InceptionV3`.

Este diretório contém a implementação prática da rede neural profunda **InceptionV3** para a classificação de imagens utilizando **TensorFlow 2.x** e **Keras**.

O modelo InceptionV3 possui cerca de 25 milhões de parâmetros, foi treinado no dataset ImageNet com 1.000 classes e espera imagens de entrada no formato 299 × 299 pixels.

---

## 📁 Estrutura de Diretórios

O script está configurado para organizar automaticamente os arquivos de entrada e saída:

* **`imagens/`**: Pasta contendo as imagens de teste baixadas automaticamente para validação do script (`panda.jpg`, `macaw.jpg`, `elephant.jpg`, `dining_table.jpg`, `dog.jpg`).
* **`imagens_locais/`**: Pasta criada para você colocar suas próprias imagens locais. Qualquer imagem colocada aqui será automaticamente detectada e classificada pelo script.
* **`resultado/`**: Pasta que centraliza todas as saídas geradas durante a execução:
  * `resultado.txt`: Relatório em formato de texto contendo as top-10 predições e acurácia de cada imagem processada.
  * `comparacao_arquiteturas.png`: Gráfico comparando parâmetros e acurácia de várias redes neurais famosas.
  * `*_resized.png`: Comparação visual entre o tamanho original e o redimensionado visto pelo modelo.
  * `*_score_distribution.png`: Distribuição das probabilidades em gráfico horizontal.
  * `*_different_sizes.png`: Grade demonstrando o impacto de resoluções menores na classificação.
  * `*_feature_maps_*.png`: Visualização de filtros aplicados em camadas específicas.

---

## 🚀 Como Executar Localmente

Como o projeto possui um ambiente virtual configurado na raiz (`.venv`), você pode executar o script diretamente seguindo os passos abaixo.

### Passo 1: Abrir o Terminal
Abra o seu terminal (PowerShell ou Prompt de Comando) e navegue até a pasta deste módulo:

```powershell
cd "c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos e Práticas de Deep Learning/Redes de Classificação para Deep Learning"
```

### Passo 2: Executar o Script

#### No Windows (PowerShell):
Execute o script apontando para o interpretador Python do ambiente virtual da raiz:
```powershell
..\..\.venv\Scripts\python.exe Inception_model.py
```

*Nota: Caso queira executar sem abrir janelas gráficas do Matplotlib (ideal para terminal/headless), defina a variável de backend do matplotlib antes de rodar:*
```powershell
$env:MPLBACKEND="Agg"; ..\..\.venv\Scripts\python.exe Inception_model.py
```

#### No Windows (CMD):
```cmd
..\..\.venv\Scripts\python.exe Inception_model.py
```

---

## 🛠️ Dependências do Projeto

As principais dependências já se encontram instaladas no ambiente `.venv` da pasta raiz:
* `tensorflow` (versão 2.x)
* `numpy`
* `matplotlib`
* `pillow` (PIL)
* `ipython`

Caso queira configurar um novo ambiente virtual próprio, você pode instalá-las rodando:
```bash
pip install tensorflow numpy matplotlib pillow ipython
```
