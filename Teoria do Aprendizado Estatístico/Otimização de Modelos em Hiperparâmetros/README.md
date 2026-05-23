# Otimização de Hiperparâmetros em Deep Learning

Este diretório contém uma implementação prática e conceitual sobre a sintonia de hiperparâmetros em modelos de Deep Learning (Redes Neurais Multicamadas - MLP), projetada para a **Formação Machine Learning Specialist**.

O script principal [otimizacao_hiperparametros.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Otimiza%C3%A7%C3%A3o%20de%20Modelos%20em%20Hiperpar%C3%A2metros/otimizacao_hiperparametros.py) simula o comportamento real de um treinamento e avalia o impacto de diferentes configurações na acurácia e perda (loss) do modelo sem a necessidade de frameworks externos pesados (como TensorFlow ou PyTorch).

---

## 🚀 O que o script faz?

1. **Simula o Treinamento de Redes Neurais**:
   A função [simular_treino](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Otimiza%C3%A7%C3%A3o%20de%20Modelos%20em%20Hiperpar%C3%A2metros/otimizacao_hiperparametros.py#L82) modela de forma determinística e estocástica (adicionando ruído gaussiano realista) os efeitos dos hiperparâmetros:
   * **Learning Rate (Taxa de Aprendizado)** muito alta provoca instabilidade/divergência; muito baixa atrasa o aprendizado.
   * **Dropout** auxilia na regularização evitando overfitting.
   * **Tamanho do lote (Batch Size)** menor introduz um ruído útil para a generalização.
   * **Otimizadores** adaptativos (Adam/RMSProp) aceleram a convergência frente ao SGD puro.
   * **Arquitetura (Camadas/Neurônios)** determina a capacidade de representação do modelo.

2. **Executa Estratégias de Busca**:
   * **Grid Search (Busca em Grade)**: Amostra e avalia sistematicamente combinações do espaço de busca através da função [grid_search](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Otimiza%C3%A7%C3%A3o%20de%20Modelos%20em%20Hiperpar%C3%A2metros/otimizacao_hiperparametros.py#L170).
   * **Random Search (Busca Aleatória)**: Amostra aleatoriamente o espaço através de [random_search](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Otimiza%C3%A7%C3%A3o%20de%20Modelos%20em%20Hiperpar%C3%A2metros/otimizacao_hiperparametros.py#L198), simulando a eficiência descrita na literatura científica (Bergstra & Bengio, 2012).

3. **Faz Análise de Sensibilidade**:
   Com a função [analisar_impacto](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Otimiza%C3%A7%C3%A3o%20de%20Modelos%20em%20Hiperpar%C3%A2metros/otimizacao_hiperparametros.py#L217), calcula o impacto relativo de cada hiperparâmetro com base na variação da acurácia média obtida durante a busca.

4. **Exporta um Relatório Completo**:
   Gera e salva o relatório formatado em [resultado/resultado_hiperparametros.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Teoria%20do%20Aprendizado%20Estat%C3%ADstico/Otimiza%C3%A7%C3%A3o%20de%20Modelos%20em%20Hiperpar%C3%A2metros/resultado/resultado_hiperparametros.txt) com análises gráficas em ASCII.

---

## 📊 Espaço de Busca de Hiperparâmetros

O espaço total de busca conta com **8.748 combinações possíveis** definidas no dicionário `HIPER_ESPACO`:

| Hiperparâmetro | Opções Disponíveis | Descrição / Papel |
| :--- | :--- | :--- |
| `learning_rate` | `[0.1, 0.01, 0.001, 0.0001]` | Passo do gradiente descendente |
| `epochs` | `[10, 30, 50]` | Ciclos de treino sobre o dataset |
| `batch_size` | `[16, 32, 64]` | Amostras por atualização de pesos |
| `neurons` | `[32, 64, 128]` | Neurônios por camada oculta |
| `dropout_rate` | `[0.0, 0.2, 0.5]` | Regularização para evitar overfitting |
| `activation` | `["relu", "tanh", "sigmoid"]` | Função de ativação das camadas ocultas |
| `optimizer` | `["sgd", "adam", "rmsprop"]` | Algoritmo de otimização |
| `num_layers` | `[1, 2, 3]` | Quantidade de camadas ocultas |

---

## 📄 Exemplo de Relatório Gerado

O script imprime no console e exporta um arquivo contendo:
* **Conceituação** teórica dos parâmetros e sua classificação.
* **Top 10 combinações** testadas no Grid Search e no Random Search com barra de progresso visual.
* **Ranking de Impacto**: Quais hiperparâmetros geraram a maior variação na acurácia média (ex: `learning_rate` e `epochs` costumam liderar o impacto).
* **Simulação de Épocas**: A curva de convergência da melhor configuração encontrada ao longo do tempo.
* **Comparação direta**: Tabela comparando a melhor acurácia obtida em ambas as estratégias.

---

## 🛠️ Como Executar

Certifique-se de estar com o ambiente virtual ativado e execute o script em seu terminal:

```bash
python "Teoria do Aprendizado Estatístico/Otimização de Modelos em Hiperparâmetros/otimizacao_hiperparametros.py"
```

Os resultados serão automaticamente exibidos no console e armazenados no arquivo:
📂 `Teoria do Aprendizado Estatístico/Otimização de Modelos em Hiperparâmetros/resultado/resultado_hiperparametros.txt`

---

## 💡 Melhores Práticas Consolidadas no Script

* **Learning Rate**: O ajuste mais crucial. Recomenda-se iniciar com `1e-3` (Adam) ou `1e-2` (SGD) e acoplar um scheduler como *ReduceLROnPlateau*.
* **Batch Size**: Comece com `32`. Lotes menores induzem ruído que ajuda a escapar de mínimos locais.
* **Regularização**: Use Dropout (`0.2` a `0.5`) apenas nas camadas ocultas (nunca na saída).
* **Ferramentas de Produção**: Em ambientes de produção reais, recomenda-se a utilização de **Optuna** (Otimização Bayesiana), **KerasTuner**, **Ray Tune** ou **Weights & Biases (wandb)**.
