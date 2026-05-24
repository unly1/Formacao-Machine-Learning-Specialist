# Transfer Learning com Keras

Este módulo apresenta uma abordagem teórica e prática sobre **Transfer Learning (Aprendizado por Transferência)** utilizando a biblioteca Keras, com foco na reutilização de arquiteturas de redes neurais consagradas (como a ResNet50) para novas tarefas de classificação.

## O que é Transfer Learning?

Transfer Learning é uma técnica de Machine Learning em que um modelo desenvolvido para uma tarefa inicial é reutilizado como ponto de partida para um modelo em uma segunda tarefa relacionada. É amplamente utilizado em Deep Learning devido ao alto custo computacional e de tempo exigido para treinar redes profundas do zero.

As etapas fundamentais do Transfer Learning consistem em:
1. **Carregar um modelo pré-treinado** (geralmente treinado no dataset ImageNet).
2. **Congelar a base convolucional** do modelo (evitando que seus pesos bem-treinados sejam alterados).
3. **Remover a cabeça classificadora original** (camadas densas finais).
4. **Adicionar uma nova cabeça classificadora** específica para o seu problema (com o número correto de classes de saída).
5. **Treinar apenas as novas camadas** adicionadas.

## Estrutura dos Arquivos nesta Pasta

- **`Exemplo.py`**: Script demonstrativo em Python que:
  - Carrega a arquitetura `ResNet50` pré-treinada sem a camada final de classificação (`include_top=False`).
  - Congela todas as camadas do modelo base para fixar os pesos originais de detecção de características.
  - Constrói um novo modelo `Sequential` acoplando um pooling global, uma camada densa intermediária de 256 neurônios e uma camada de saída softmax com 2 classes personalizadas.
  - Demonstra a extração de características da imagem teste e gera predições com o novo classificador.
- **`imagem/gato.png`**: Imagem teste utilizada pelo script para validação prática do fluxo de trabalho.
- **`resultado_transfer_learning.txt`**: Resumo das predições e da arquitetura do modelo gerado após a execução do script.
