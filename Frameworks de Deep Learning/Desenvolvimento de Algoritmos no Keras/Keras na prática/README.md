# Keras na Prática

Este módulo apresenta uma demonstração prática do uso do **Keras** integrado com o **TensorFlow** para a classificação automatizada de imagens utilizando redes profundas pré-treinadas.

## ResNet50 e Classificação de Imagens

Neste laboratório prático, utilizamos a rede **ResNet50** (Residual Networks de 50 camadas) com pesos pré-treinados no conhecido dataset ImageNet (composto por mais de 14 milhões de imagens e 1000 classes de objetos cotidianos). 

A aplicação demonstra a capacidade do modelo de realizar predições de alto nível em novas imagens sem a necessidade de treinar uma rede do zero.

## Estrutura dos Arquivos nesta Pasta

- **`Exemplo.py`**: Script em Python que executa os seguintes passos:
  - Importa e instancia a rede `ResNet50` pré-treinada com os pesos do `imagenet`.
  - Carrega a imagem de teste `gato.png` e a redimensiona para `224x224` pixels (requisito do modelo).
  - Pré-processa a imagem para atender aos padrões exigidos pelo tensor de entrada da ResNet50.
  - Executa a inferência (`predict`) e decodifica as 3 principais previsões com suas respectivas probabilidades (`decode_predictions`).
  - Salva os resultados das previsões em um arquivo `.txt` local.
  - Exibe a imagem de entrada em tela usando a biblioteca `matplotlib`.
- **`imagem/gato.png`**: Imagem teste de exemplo utilizada para a classificação.
- **`resultado.txt`**: Resultados das predições de classificação gerados automaticamente na execução do script.
