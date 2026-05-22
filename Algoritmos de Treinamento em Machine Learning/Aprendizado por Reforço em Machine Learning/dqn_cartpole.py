"""
Script de Deep Q-Learning (DQN)
Framework : TensorFlow / Keras
Ambiente  : CartPole-v1 (Gymnasium)

Instale as dependências antes de rodar:
    pip install tensorflow gymnasium numpy
"""

import random
import numpy as np
import gymnasium as gym
import tensorflow as tf
from tensorflow import keras
from collections import deque
import io
import sys
from datetime import datetime
from pathlib import Path

# Pasta de saída: subpasta "resultado" criada automaticamente
OUTPUT_DIR = Path(__file__).parent / "resultado"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Buffer para capturar todo o output e salvar no .txt
_buffer = io.StringIO()


class _Tee:
    """Redireciona o print para o terminal E para o buffer ao mesmo tempo."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                if hasattr(s, "encoding") and s.encoding:
                    safe_data = data.encode(
                        s.encoding, errors="replace").decode(s.encoding)
                    s.write(safe_data)
                else:
                    s.write(data.encode(
                        "ascii", errors="replace").decode("ascii"))

    def flush(self):
        for s in self.streams:
            s.flush()


sys.stdout = _Tee(sys.__stdout__, _buffer)

# ─────────────────────────────────────────
# HIPERPARÂMETROS
# ─────────────────────────────────────────
ENV_NAME = "CartPole-v1"
EPISODES = 500          # total de episódios de treino
MAX_STEPS = 500          # máximo de passos por episódio

GAMMA = 0.99         # fator de desconto (quanto valorizar recompensas futuras)
LEARNING_RATE = 1e-3         # taxa de aprendizado do otimizador

EPSILON_START = 1.0          # exploração inicial (100% aleatório)
EPSILON_MIN = 0.01         # exploração mínima
EPSILON_DECAY = 0.995        # decaimento por episódio

MEMORY_SIZE = 10_000       # tamanho máximo do replay buffer
BATCH_SIZE = 64           # amostras por atualização
TARGET_UPDATE = 10           # a cada N episódios sincroniza a rede alvo

PRINT_EVERY = 20           # exibe progresso a cada N episódios
SOLVE_SCORE = 475          # considera resolvido se média >= este valor

# ─────────────────────────────────────────
# 1. AMBIENTE
# ─────────────────────────────────────────
env = gym.make(ENV_NAME)
state_size = int(env.observation_space.shape[0])   # 4 features no CartPole
# 2 ações: esquerda / direita
action_size = int(env.action_space.n)

print(f"Ambiente : {ENV_NAME}")
print(f"Estados  : {state_size}  |  Ações: {action_size}\n")

# ─────────────────────────────────────────
# 2. CONSTRUÇÃO DA REDE NEURAL (Q-Network)
# ─────────────────────────────────────────


def build_model(state_size: int, action_size: int) -> keras.Model:
    """Rede fully-connected simples: estado → Q-valores para cada ação."""
    model = keras.Sequential([
        keras.layers.Input(shape=(state_size,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        # saída linear = Q-valores
        keras.layers.Dense(action_size, activation="linear"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )
    return model


# Rede principal (atualizada a cada passo)
q_network = build_model(state_size, action_size)

# Rede alvo (atualizada periodicamente — estabiliza o treino)
target_network = build_model(state_size, action_size)
target_network.set_weights(q_network.get_weights())

q_network.summary()

# ─────────────────────────────────────────
# 3. REPLAY BUFFER
# ─────────────────────────────────────────
memory = deque(maxlen=MEMORY_SIZE)


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# ─────────────────────────────────────────
# 4. POLÍTICA EPSILON-GREEDY
# ─────────────────────────────────────────


def choose_action(state: np.ndarray, epsilon: float) -> int:
    """Explora aleatoriamente com prob. epsilon; caso contrário usa a rede."""
    if random.random() < epsilon:
        return env.action_space.sample()
    q_values = q_network.predict(state[np.newaxis], verbose=0)
    return int(np.argmax(q_values[0]))

# ─────────────────────────────────────────
# 5. ATUALIZAÇÃO DA REDE (Experience Replay)
# ─────────────────────────────────────────


def replay():
    if len(memory) < BATCH_SIZE:
        return  # aguarda buffer encher

    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

    # Q-valores atuais
    q_current = q_network.predict(states, verbose=0)

    # Q-valores futuros estimados pela rede ALVO
    q_next = target_network.predict(next_states, verbose=0)

    # Equação de Bellman: Q(s,a) ← r + γ · max Q(s',a')
    targets = rewards + GAMMA * np.max(q_next, axis=1) * (1 - dones)

    # Atualiza apenas a ação tomada
    q_current[np.arange(BATCH_SIZE), actions] = targets

    q_network.train_on_batch(states, q_current)


# ─────────────────────────────────────────
# 6. LOOP DE TREINAMENTO
# ─────────────────────────────────────────
epsilon = EPSILON_START
scores = []

print("Iniciando treinamento...\n")

for episode in range(1, EPISODES + 1):

    state, _ = env.reset()
    total_reward = 0

    for step in range(MAX_STEPS):
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        remember(state, action, reward, next_state, done)
        replay()

        state = next_state
        total_reward += reward

        if done:
            break

    scores.append(total_reward)
    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    # Sincroniza rede alvo periodicamente
    if episode % TARGET_UPDATE == 0:
        target_network.set_weights(q_network.get_weights())

    # Log de progresso
    if episode % PRINT_EVERY == 0:
        mean_score = np.mean(scores[-PRINT_EVERY:])
        print(
            f"Ep {episode:4d} | Score médio (últimos {PRINT_EVERY}): {mean_score:6.1f} | epsilon: {epsilon:.3f}")

    # Critério de parada
    if len(scores) >= 100 and np.mean(scores[-100:]) >= SOLVE_SCORE:
        print(f"\n[OK] Ambiente resolvido no episódio {episode}! "
              f"Média últimos 100 eps: {np.mean(scores[-100:]):.1f}")
        break

# ─────────────────────────────────────────
# 7. SALVAR O MODELO E LOGS
# ─────────────────────────────────────────
path_modelo = OUTPUT_DIR / "dqn_cartpole.keras"
q_network.save(path_modelo)
print(f"\nModelo salvo em '{path_modelo}'")

# ─────────────────────────────────────────
# 8. SALVAR RESULTADO EM .TXT
# ─────────────────────────────────────────
sys.stdout = sys.__stdout__  # restaura o stdout original

path_resultado = OUTPUT_DIR / "resultado_dqn.txt"
conteudo = _buffer.getvalue()

with open(path_resultado, "w", encoding="utf-8") as f:
    f.write(
        f"Resultado gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
    )
    f.write("=" * 60 + "\n\n")
    f.write(conteudo)

print(f"Resultado salvo em '{path_resultado}'")

# ─────────────────────────────────────────
# 8. AVALIAÇÃO VISUAL (opcional)
# ─────────────────────────────────────────
# Para ver o agente jogar após o treino:

# env_render = gym.make(ENV_NAME, render_mode="human")
# state, _ = env_render.reset()
# for _ in range(500):
#     action = int(np.argmax(q_network.predict(state[np.newaxis], verbose=0)))
#     state, _, terminated, truncated, _ = env_render.step(action)
#     if terminated or truncated:
#         break
# env_render.close()
