"""
==============================================================
  OTIMIZAÇÃO DE MODELOS EM HIPERPARÂMETROS
  Referência: Ajuste de Hiperparâmetros em Deep Learning
  Formação Machine Learning Specialist
==============================================================

Conceito:
  Hiperparâmetros são configurações definidas ANTES do treino
  (ex: learning rate, batch size, camadas, dropout).
  A otimização busca a combinação que maximiza a performance
  do modelo sem overfitting.

Estratégias implementadas:
  1. Grid Search        - busca exaustiva em grade
  2. Random Search      - busca aleatória no espaço
  3. Simulação de Treino de Rede Neural (sem framework externo)
"""

import os
import math
import random
import itertools
from datetime import datetime

random.seed(42)

# ══════════════════════════════════════════════════════════════
# 1. DEFINIÇÃO DO ESPAÇO DE HIPERPARÂMETROS
# ══════════════════════════════════════════════════════════════

HIPER_ESPACO = {
    # Taxa de aprendizado — controla o passo do gradiente
    "learning_rate":    [0.1, 0.01, 0.001, 0.0001],

    # Número de épocas — quantas vezes o modelo vê o dataset
    "epochs":           [10, 30, 50],

    # Tamanho do lote — amostras por atualização de peso
    "batch_size":       [16, 32, 64],

    # Número de neurônios por camada oculta
    "neurons":          [32, 64, 128],

    # Taxa de dropout — regularização contra overfitting
    "dropout_rate":     [0.0, 0.2, 0.5],

    # Função de ativação das camadas ocultas
    "activation":       ["relu", "tanh", "sigmoid"],

    # Otimizador de gradiente
    "optimizer":        ["sgd", "adam", "rmsprop"],

    # Número de camadas ocultas
    "num_layers":       [1, 2, 3],
}


# ══════════════════════════════════════════════════════════════
# 2. FUNÇÕES DE ATIVAÇÃO (implementação manual)
# ══════════════════════════════════════════════════════════════

def relu(x):
    return max(0.0, x)


def tanh_fn(x):
    return math.tanh(x)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-min(max(x, -500), 500)))


ATIVACOES = {"relu": relu, "tanh": tanh_fn, "sigmoid": sigmoid}


# ══════════════════════════════════════════════════════════════
# 3. SIMULAÇÃO DE TREINO DE REDE NEURAL (Deep Learning)
# ══════════════════════════════════════════════════════════════

def simular_treino(hparams, verbose=False):
    """
    Simula o comportamento de uma rede neural densa (MLP)
    com os hiperparâmetros fornecidos.

    Modela os efeitos reais de cada hiperparâmetro:
      - learning_rate alto  → instabilidade / divergência
      - dropout alto        → mais regularização, menos overfit
      - mais camadas        → mais capacidade (e risco de overfit)
      - batch_size pequeno  → ruído útil no gradiente
      - adam/rmsprop        → convergência mais rápida que sgd
    """
    lr = hparams["learning_rate"]
    epochs = hparams["epochs"]
    batch_size = hparams["batch_size"]
    neurons = hparams["neurons"]
    dropout = hparams["dropout_rate"]
    activation = hparams["activation"]
    optimizer = hparams["optimizer"]
    num_layers = hparams["num_layers"]

    # ── Capacidade base da arquitetura ──
    capacidade = math.log2(neurons) * num_layers  # ex: 7*2 = 14

    # ── Penalidade pelo learning rate ──
    if lr >= 0.1:
        penalidade_lr = 0.55   # muito alto → diverge
    elif lr == 0.01:
        penalidade_lr = 0.10
    elif lr == 0.001:
        penalidade_lr = 0.0    # ideal
    else:
        penalidade_lr = 0.08   # muito baixo → lento

    # ── Bônus do otimizador ──
    bonus_opt = {"adam": 0.08, "rmsprop": 0.05, "sgd": 0.0}[optimizer]

    # ── Efeito do dropout (regularização) ──
    bonus_dropout = dropout * 0.15  # até +0.075 para dropout=0.5

    # ── Efeito do batch size ──
    bonus_batch = 0.02 if batch_size == 32 else (
        0.01 if batch_size == 16 else 0.0)

    # ── Efeito da ativação ──
    bonus_ativ = {"relu": 0.06, "tanh": 0.03, "sigmoid": 0.0}[activation]

    # ── Convergência ao longo das épocas ──
    fator_epocas = 1.0 - math.exp(-epochs / 20.0)   # satura em ~1.0

    # ── Acurácia base simulada ──
    acuracia_base = 0.50 + (capacidade / 60.0)        # entre 0.5 e ~0.9
    acuracia_base = min(acuracia_base, 0.95)

    # ── Acurácia final com todos os fatores ──
    acuracia = (
        acuracia_base
        + bonus_opt
        + bonus_dropout
        + bonus_batch
        + bonus_ativ
        - penalidade_lr
    ) * fator_epocas

    # ── Ruído realista de treinamento ──
    acuracia += random.gauss(0, 0.012)
    acuracia = max(0.30, min(0.995, acuracia))

    # ── Loss inversamente proporcional à acurácia ──
    loss = -math.log(max(acuracia, 0.01)) + random.gauss(0, 0.02)
    loss = max(0.005, loss)

    historico = []
    if verbose:
        acc_atual = 0.3
        for ep in range(1, epochs + 1):
            acc_atual += (acuracia - acc_atual) * \
                (lr * 5) * random.uniform(0.8, 1.2)
            acc_atual = min(acc_atual, acuracia)
            historico.append((ep, round(acc_atual, 4)))

    return round(acuracia, 4), round(loss, 4), historico


# ══════════════════════════════════════════════════════════════
# 4. GRID SEARCH — Busca Exaustiva
# ══════════════════════════════════════════════════════════════

def grid_search(espaco, max_combinacoes=30):
    """
    Testa todas as combinações do espaço de hiperparâmetros.
    Limitado a max_combinacoes para viabilidade de execução.
    """
    chaves = list(espaco.keys())
    valores = list(espaco.values())
    todas = list(itertools.product(*valores))

    # Embaralha para variedade (grid parcial)
    random.shuffle(todas)
    subset = todas[:max_combinacoes]

    resultados = []
    for combo in subset:
        hparams = dict(zip(chaves, combo))
        acc, loss, _ = simular_treino(hparams)
        resultados.append((hparams, acc, loss))

    # Ordena pelo maior accuracy
    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados


# ══════════════════════════════════════════════════════════════
# 5. RANDOM SEARCH — Busca Aleatória
# ══════════════════════════════════════════════════════════════

def random_search(espaco, n_iter=20):
    """
    Amostra aleatoriamente n_iter combinações do espaço.
    Mais eficiente que Grid Search em espaços grandes.
    """
    resultados = []
    for _ in range(n_iter):
        hparams = {k: random.choice(v) for k, v in espaco.items()}
        acc, loss, _ = simular_treino(hparams)
        resultados.append((hparams, acc, loss))

    resultados.sort(key=lambda x: x[1], reverse=True)
    return resultados


# ══════════════════════════════════════════════════════════════
# 6. COMPARAÇÃO E ANÁLISE DOS RESULTADOS
# ══════════════════════════════════════════════════════════════

def analisar_impacto(resultados_grid):
    """
    Analisa qual hiperparâmetro tem maior impacto médio na acurácia.
    """
    impacto = {}
    for chave in HIPER_ESPACO:
        grupos = {}
        for hparams, acc, _ in resultados_grid:
            val = str(hparams[chave])
            grupos.setdefault(val, []).append(acc)
        medias = {v: round(sum(lst)/len(lst), 4) for v, lst in grupos.items()}
        variacao = round(max(medias.values()) - min(medias.values()), 4)
        impacto[chave] = {"medias": medias, "variacao": variacao}
    return impacto


# ══════════════════════════════════════════════════════════════
# 7. MAIN — EXECUÇÃO E GERAÇÃO DO RELATÓRIO
# ══════════════════════════════════════════════════════════════

def main():
    linhas = []

    def L(txt=""):
        linhas.append(txt)

    L("=" * 65)
    L("   OTIMIZAÇÃO DE MODELOS EM HIPERPARÂMETROS")
    L("   Referência: Ajuste em Redes de Deep Learning (MLP)")
    L("   Formação Machine Learning Specialist")
    L("=" * 65)
    L(f"   Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    L("=" * 65)

    # ── Conceitos ──
    L()
    L("O QUE SÃO HIPERPARÂMETROS?")
    L("  São configurações definidas ANTES do treinamento. Diferente")
    L("  dos parâmetros (pesos), eles não são aprendidos pelo modelo.")
    L()
    L("HIPERPARÂMETROS DE ARQUITETURA (estrutura da rede):")
    L("  • num_layers    : número de camadas ocultas")
    L("  • neurons       : neurônios por camada")
    L("  • activation    : função de ativação (ReLU, Tanh, Sigmoid)")
    L()
    L("HIPERPARÂMETROS DE TREINAMENTO (processo de aprendizado):")
    L("  • learning_rate : tamanho do passo no gradiente descendente")
    L("  • batch_size    : amostras por atualização de peso")
    L("  • epochs        : número de ciclos completos sobre o dataset")
    L("  • optimizer     : algoritmo de otimização (SGD, Adam, RMSProp)")
    L()
    L("HIPERPARÂMETROS DE REGULARIZAÇÃO (combate ao overfitting):")
    L("  • dropout_rate  : fração de neurônios desativados por época")
    L()
    L("ESPAÇO DE BUSCA DEFINIDO:")
    for k, v in HIPER_ESPACO.items():
        L(f"  • {k:<18}: {v}")

    total = 1
    for v in HIPER_ESPACO.values():
        total *= len(v)
    L(f"\n  Total de combinações possíveis: {total:,}")
    L()

    # ── ESTRATÉGIAS ──
    L("=" * 65)
    L("  ESTRATÉGIAS DE BUSCA DE HIPERPARÂMETROS")
    L("=" * 65)
    L()
    L("1. GRID SEARCH (Busca em Grade)")
    L("   Testa sistematicamente todas as combinações.")
    L("   Vantagem : garante encontrar o melhor no espaço definido.")
    L("   Desvantagem: custo computacional exponencial (maldição")
    L("   da dimensionalidade).")
    L()
    L("2. RANDOM SEARCH (Busca Aleatória)")
    L("   Amostra aleatoriamente o espaço de configurações.")
    L("   Vantagem : eficiente — cobre bem espaços grandes.")
    L("   Referência: Bergstra & Bengio (2012) mostraram que")
    L("   Random Search supera Grid Search na maioria dos casos.")
    L()
    L("3. BAYESIAN OPTIMIZATION (referência teórica)")
    L("   Usa modelo probabilístico (ex: Gaussian Process) para")
    L("   guiar a busca. Implementações: Optuna, Hyperopt, KerasTuner.")
    L()
    L("4. EARLY STOPPING")
    L("   Interrompe o treino quando a val_loss para de melhorar.")
    L("   Evita overfitting e economiza tempo de computação.")
    L()

    # ── GRID SEARCH ──
    L("=" * 65)
    L("  RESULTADO — GRID SEARCH (30 combinações amostradas)")
    L("=" * 65)
    grid_res = grid_search(HIPER_ESPACO, max_combinacoes=30)

    L()
    L(f"  {'#':<4} {'Acurácia':>9} {'Loss':>7}  Hiperparâmetros")
    L("  " + "-" * 60)
    for i, (hp, acc, loss) in enumerate(grid_res[:10], 1):
        barra = "█" * int(acc * 20)
        L(f"  {i:<4} {acc:>8.4f}  {loss:>6.4f}  lr={hp['learning_rate']} | "
          f"opt={hp['optimizer']} | act={hp['activation']}")
        L(f"       [{barra:<20}]  layers={hp['num_layers']} | "
          f"neurons={hp['neurons']} | drop={hp['dropout_rate']}")

    melhor_grid = grid_res[0]
    L()
    L("  ★ MELHOR CONFIGURAÇÃO (Grid Search):")
    for k, v in melhor_grid[0].items():
        L(f"    • {k:<18}: {v}")
    L(f"    → Acurácia : {melhor_grid[1]:.4f} ({melhor_grid[1]*100:.2f}%)")
    L(f"    → Loss     : {melhor_grid[2]:.4f}")

    # ── RANDOM SEARCH ──
    L()
    L("=" * 65)
    L("  RESULTADO — RANDOM SEARCH (20 iterações)")
    L("=" * 65)
    rand_res = random_search(HIPER_ESPACO, n_iter=20)

    L()
    L(f"  {'#':<4} {'Acurácia':>9} {'Loss':>7}  Hiperparâmetros")
    L("  " + "-" * 60)
    for i, (hp, acc, loss) in enumerate(rand_res[:10], 1):
        barra = "█" * int(acc * 20)
        L(f"  {i:<4} {acc:>8.4f}  {loss:>6.4f}  lr={hp['learning_rate']} | "
          f"opt={hp['optimizer']} | act={hp['activation']}")
        L(f"       [{barra:<20}]  layers={hp['num_layers']} | "
          f"neurons={hp['neurons']} | drop={hp['dropout_rate']}")

    melhor_rand = rand_res[0]
    L()
    L("  ★ MELHOR CONFIGURAÇÃO (Random Search):")
    for k, v in melhor_rand[0].items():
        L(f"    • {k:<18}: {v}")
    L(f"    → Acurácia : {melhor_rand[1]:.4f} ({melhor_rand[1]*100:.2f}%)")
    L(f"    → Loss     : {melhor_rand[2]:.4f}")

    # ── ANÁLISE DE IMPACTO ──
    L()
    L("=" * 65)
    L("  ANÁLISE DE IMPACTO POR HIPERPARÂMETRO")
    L("=" * 65)
    L("  (Variação = diferença entre o melhor e pior valor médio)")
    L()
    impacto = analisar_impacto(grid_res)
    impacto_ord = sorted(
        impacto.items(), key=lambda x: x[1]["variacao"], reverse=True)

    for rank, (hiper, dados) in enumerate(impacto_ord, 1):
        var = dados["variacao"]
        barra = "█" * int(var * 100)
        L(f"  {rank}. {hiper:<18} variação: {var:.4f}  [{barra:<20}]")
        for val, med in sorted(dados["medias"].items(), key=lambda x: x[1], reverse=True):
            L(f"       {str(val):<12} → acc médio: {med:.4f}")
        L()

    # ── SIMULAÇÃO DETALHADA DO MELHOR MODELO ──
    L("=" * 65)
    L("  SIMULAÇÃO DE TREINO — MELHOR MODELO (GRID SEARCH)")
    L("=" * 65)
    L()
    acc_final, loss_final, historico = simular_treino(
        melhor_grid[0], verbose=True)
    L("  Época | Acurácia | Progresso")
    L("  " + "-" * 45)
    for ep, acc in historico[::max(1, len(historico)//10)]:
        barra = "█" * int(acc * 30)
        L(f"  {ep:>5} | {acc:.4f}   | [{barra:<30}]")
    L(f"\n  Acurácia final simulada : {acc_final:.4f} ({acc_final*100:.2f}%)")
    L(f"  Loss final simulada     : {loss_final:.4f}")

    # ── COMPARAÇÃO FINAL ──
    L()
    L("=" * 65)
    L("  COMPARAÇÃO FINAL DAS ESTRATÉGIAS")
    L("=" * 65)
    L()
    L(f"  {'Estratégia':<22} {'Melhor Acurácia':>16} {'Melhor Loss':>12}")
    L("  " + "-" * 52)
    L(f"  {'Grid Search':<22} {melhor_grid[1]:>16.4f} {melhor_grid[2]:>12.4f}")
    L(f"  {'Random Search':<22} {melhor_rand[1]:>16.4f} {melhor_rand[2]:>12.4f}")
    L()

    # ── BOAS PRÁTICAS ──
    L("=" * 65)
    L("  BOAS PRÁTICAS DE AJUSTE DE HIPERPARÂMETROS")
    L("=" * 65)
    L()
    L("  LEARNING RATE:")
    L("  • É o hiperparâmetro mais crítico em Deep Learning")
    L("  • Comece com 1e-3 (Adam) ou 1e-2 (SGD)")
    L("  • Use Learning Rate Scheduler (ex: ReduceLROnPlateau)")
    L("  • Visualize a curva de loss: se oscilar, reduza o LR")
    L()
    L("  BATCH SIZE:")
    L("  • Lotes menores (16-32): mais ruído, melhor generalização")
    L("  • Lotes maiores (128+) : mais estável, mais rápido por época")
    L("  • Regra prática: comece com 32")
    L()
    L("  DROPOUT:")
    L("  • Use 0.2–0.5 nas camadas ocultas para regularização")
    L("  • Nunca aplique na camada de saída")
    L("  • Combine com Batch Normalization para melhor resultado")
    L()
    L("  ARQUITETURA:")
    L("  • Comece simples (1-2 camadas) e aumente a complexidade")
    L("  • ReLU é a ativação padrão para camadas ocultas")
    L("  • Mais neurônios e camadas = mais capacidade e mais risco")
    L()
    L("  OTIMIZADORES:")
    L("  • Adam    : melhor ponto de partida (adaptativo)")
    L("  • RMSProp : bom para RNNs e dados ruidosos")
    L("  • SGD     : pode superar Adam com LR scheduler afinado")
    L()
    L("  FERRAMENTAS RECOMENDADAS PARA PRODUÇÃO:")
    L("  • Optuna      : Bayesian Optimization automático")
    L("  • KerasTuner  : integrado ao TensorFlow/Keras")
    L("  • Ray Tune    : busca distribuída em cluster")
    L("  • Weights & Biases (wandb): rastreamento de experimentos")
    L()
    L("=" * 65)
    L("  Fim do Relatório")
    L("=" * 65)

    # ── SALVAR ──
    caminho_destino = (
        r"C:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist"
        r"\Teoria do Aprendizado Estatístico"
        r"\Otimização de Modelos em Hiperparâmetros"
        r"\resultado"
    )
    os.makedirs(caminho_destino, exist_ok=True)
    arquivo_saida = os.path.join(
        caminho_destino, "resultado_hiperparametros.txt")
    conteudo = "\n".join(linhas)

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(conteudo)

    print(conteudo)
    print(f"\n✔ Arquivo salvo em:\n  {arquivo_saida}")


if __name__ == "__main__":
    main()
