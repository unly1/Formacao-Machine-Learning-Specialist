# 🤖 Aprendizado por Reforço em Machine Learning — Deep Q-Learning (DQN) com TensorFlow e Keras

Este diretório contém a implementação prática estruturada de um agente de **Aprendizado por Reforço** treinado via algoritmo **Deep Q-Network (DQN)** para resolver o clássico ambiente de controle **CartPole-v1** da biblioteca **Gymnasium**.

---

## 📁 Estrutura do Diretório

O diretório é composto pelos seguintes elementos:
*   **[dqn_cartpole.py](dqn_cartpole.py)**: Script Python principal contendo o pipeline completo de treinamento do agente DQN: inicialização do ambiente, definição da Q-Network e da Target Network, política $\epsilon$-greedy, Replay Buffer (Experience Replay), treinamento do modelo e persistência de arquivos.
*   **[resultado/](resultado)**: Pasta criada automaticamente ao executar o script para armazenar os artefatos gerados pelo treinamento.
    *   `dqn_cartpole.keras`: O modelo de rede neural profunda treinado e salvo no formato nativo do Keras/TensorFlow.
    *   `resultado_dqn.txt`: Relatório completo contendo o log textual de cada um dos passos e episódios de treinamento e o score médio obtido.

---

## 🕹️ O Problema: CartPole-v1

No ambiente **CartPole-v1**, um mastro está preso por uma articulação não motorizada a um carrinho, que se move ao longo de um trilho sem atrito. O objetivo é equilibrar o mastro aplicando forças para a esquerda ou direita no carrinho.

*   **Espaço de Estados (4 features)**: Posição do carrinho, velocidade do carrinho, ângulo do mastro e velocidade angular do mastro.
*   **Espaço de Ações (2 ações discretas)**: 
    *   `0`: Mover o carrinho para a esquerda.
    *   `1`: Mover o carrinho para a direita.
*   **Recompensa**: `+1` para cada passo que o mastro permanecer em pé.
*   **Critério de Parada/Resolução**: O ambiente é considerado resolvido se a pontuação média (score médio) dos últimos 100 episódios for maior ou igual a **475**.

---

## ⚙️ Funcionamento do Agente DQN

O agente implementa as principais inovações técnicas que tornaram as Redes Q Profundas estáveis:
1.  **Q-Network e Target Network**: Duas redes neurais com arquitetura idêntica são criadas. A `q_network` é otimizada a cada passo de treino, enquanto a `target_network` serve como âncora fixa para calcular o valor de Q futuro desejado, sendo sincronizada com a rede principal apenas a cada $N$ episódios (definido por `TARGET_UPDATE`). Isso evita a oscilação rápida de alvos de treino.
2.  **Replay Buffer (Experience Replay)**: As transições do agente `(estado, ação, recompensa, próximo_estado, done)` são salvas em uma fila (`deque` com capacidade limitada). Lotes aleatórios são amostrados para treinamento, quebrando a correlação temporal inerente ao jogo sequencial.
3.  **Política $\epsilon$-Greedy**: O agente começa explorando totalmente (`epsilon = 1.0` - escolhas aleatórias) e decai gradativamente (`epsilon_decay = 0.995`) até um limite de exploração mínima (`epsilon_min = 0.01`), onde passa a apenas explorar em 1% do tempo e agir de forma gulosa (explotar previsões da rede) nos outros 99%.

---

## 🚀 Como Executar Localmente

### 1. Ativar o Ambiente Virtual
A partir desta pasta, ative o ambiente virtual configurado no projeto:

No Windows (PowerShell):
```powershell
..\..\.venv\Scripts\Activate.ps1
```

### 2. Instalar as Dependências Necessárias
Certifique-se de que os pacotes necessários estão instalados em seu ambiente virtual:
```bash
pip install tensorflow gymnasium numpy
```

### 3. Executar o Script de Treinamento
Rode o script pelo console com o parâmetro `-u` para garantir que o console mostre os logs de progresso sem atrasos (unbuffered):
```bash
python -u dqn_cartpole.py
```

Durante a execução, você verá a arquitetura da rede neural compilada e o log de treinamento impresso no terminal a cada 20 episódios:
```text
Ambiente : CartPole-v1
Estados  : 4  |  Ações: 2

Model: "sequential"
...
Iniciando treinamento...

Ep   20 | Score médio (últimos 20):   20.2 | epsilon: 0.905
Ep   40 | Score médio (últimos 20):   24.2 | epsilon: 0.818
...
```

Após atingir a resolução ou o total de 500 episódios, verifique os artefatos salvos dentro do diretório [resultado/](resultado).
