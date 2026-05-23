"""
==============================================================
  ALGORITMOS FUZZY (Lógica Nebulosa) - Script Básico
  Formação Machine Learning Specialist
  Teoria do Aprendizado Estatístico | Métodos de Otimização
==============================================================

Conceito:
  A Lógica Fuzzy permite que variáveis assumam graus de pertinência
  entre 0 e 1, em vez de apenas verdadeiro/falso (0 ou 1).
  Útil para modelar incerteza e linguagem natural.

Exemplo aplicado: Sistema de Controle de Temperatura
  - Entrada: Temperatura atual (°C)
  - Saída: Velocidade do ventilador (%)
"""

import os
import math
from datetime import datetime

# ─────────────────────────────────────────────
# 1. FUNÇÕES DE PERTINÊNCIA (Membership Functions)
# ─────────────────────────────────────────────

def pertinencia_triangular(x, a, b, c):
    """
    Função de pertinência triangular.
    a = ponto mínimo, b = pico (pertinência 1.0), c = ponto máximo
    """
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)


def pertinencia_trapezoid(x, a, b, c, d):
    """
    Função de pertinência trapezoidal.
    Pertinência máxima entre b e c.
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif b <= x <= c:
        return 1.0
    else:
        return (d - x) / (d - c)


def pertinencia_gaussiana(x, media, sigma):
    """
    Função de pertinência gaussiana (curva em sino).
    """
    return math.exp(-0.5 * ((x - media) / sigma) ** 2)


# ─────────────────────────────────────────────
# 2. CONJUNTOS FUZZY PARA TEMPERATURA (entrada)
# ─────────────────────────────────────────────

def fuzzificar_temperatura(temp):
    """
    Fuzzifica a temperatura em 3 conjuntos linguísticos:
    FRIA, CONFORTAVEL, QUENTE
    """
    fria        = pertinencia_trapezoid(temp, 0, 0, 15, 25)
    confortavel = pertinencia_triangular(temp, 15, 22, 30)
    quente      = pertinencia_trapezoid(temp, 25, 32, 50, 50)
    return {
        "FRIA":        round(fria, 4),
        "CONFORTAVEL": round(confortavel, 4),
        "QUENTE":      round(quente, 4),
    }


# ─────────────────────────────────────────────
# 3. REGRAS FUZZY (Base de Regras)
# ─────────────────────────────────────────────

def aplicar_regras(fuzzy_temp):
    """
    SE temperatura é FRIA     → ventilador é LENTO
    SE temperatura é CONFORT  → ventilador é MÉDIO
    SE temperatura é QUENTE   → ventilador é RÁPIDO
    """
    ativacoes = {
        "LENTO":  fuzzy_temp["FRIA"],
        "MEDIO":  fuzzy_temp["CONFORTAVEL"],
        "RAPIDO": fuzzy_temp["QUENTE"],
    }
    return ativacoes


# ─────────────────────────────────────────────
# 4. DEFUZZIFICAÇÃO (Método do Centroide)
# ─────────────────────────────────────────────

def defuzzificar(ativacoes):
    """
    Defuzzificação pelo método do Centroide (Center of Gravity).
    Universo de saída: 0% a 100% de velocidade do ventilador.
    
    Centros dos conjuntos de saída:
      LENTO  → 20%
      MEDIO  → 50%
      RAPIDO → 90%
    """
    centros = {
        "LENTO":  20.0,
        "MEDIO":  50.0,
        "RAPIDO": 90.0,
    }

    numerador   = sum(ativacoes[k] * centros[k] for k in ativacoes)
    denominador = sum(ativacoes[k] for k in ativacoes)

    if denominador == 0:
        return 0.0
    return round(numerador / denominador, 2)


# ─────────────────────────────────────────────
# 5. SISTEMA FUZZY COMPLETO
# ─────────────────────────────────────────────

def sistema_fuzzy(temperatura):
    """
    Pipeline completo:
    Entrada crisp → Fuzzificação → Regras → Defuzzificação → Saída crisp
    """
    fuzzy_temp  = fuzzificar_temperatura(temperatura)
    ativacoes   = aplicar_regras(fuzzy_temp)
    velocidade  = defuzzificar(ativacoes)
    return fuzzy_temp, ativacoes, velocidade


# ─────────────────────────────────────────────
# 6. EXECUÇÃO E GERAÇÃO DO RELATÓRIO
# ─────────────────────────────────────────────

def main():
    temperaturas_teste = [5, 12, 18, 22, 26, 30, 38, 45]

    linhas = []
    linhas.append("=" * 62)
    linhas.append("   RELATÓRIO - SISTEMA DE CONTROLE FUZZY")
    linhas.append("   Formação Machine Learning Specialist")
    linhas.append("   Teoria do Aprendizado Estatístico")
    linhas.append("   Métodos de Otimização de Aprendizado")
    linhas.append("=" * 62)
    linhas.append(f"   Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    linhas.append("=" * 62)
    linhas.append("")
    linhas.append("CONCEITO:")
    linhas.append("  A Lógica Fuzzy (nebulosa) permite trabalhar com graus de")
    linhas.append("  pertinência entre 0 e 1, modelando incerteza e linguagem")
    linhas.append("  natural de forma matemática.")
    linhas.append("")
    linhas.append("PIPELINE DO SISTEMA:")
    linhas.append("  [Entrada Crisp] → [Fuzzificação] → [Base de Regras]")
    linhas.append("                 → [Defuzzificação] → [Saída Crisp]")
    linhas.append("")
    linhas.append("FUNÇÕES DE PERTINÊNCIA UTILIZADAS:")
    linhas.append("  • Triangular  : pertinencia_triangular(x, a, b, c)")
    linhas.append("  • Trapezoidal : pertinencia_trapezoid(x, a, b, c, d)")
    linhas.append("  • Gaussiana   : pertinencia_gaussiana(x, media, sigma)")
    linhas.append("")
    linhas.append("CONJUNTOS FUZZY - TEMPERATURA (Entrada):")
    linhas.append("  • FRIA        : Trapezoidal [0, 0, 15, 25]")
    linhas.append("  • CONFORTÁVEL : Triangular  [15, 22, 30]")
    linhas.append("  • QUENTE      : Trapezoidal [25, 32, 50, 50]")
    linhas.append("")
    linhas.append("BASE DE REGRAS (SE-ENTÃO):")
    linhas.append("  • SE temp é FRIA        → Ventilador LENTO  (centro: 20%)")
    linhas.append("  • SE temp é CONFORTÁVEL → Ventilador MÉDIO  (centro: 50%)")
    linhas.append("  • SE temp é QUENTE      → Ventilador RÁPIDO (centro: 90%)")
    linhas.append("")
    linhas.append("DEFUZZIFICAÇÃO: Método do Centroide (Center of Gravity)")
    linhas.append("  Fórmula: saída = Σ(μᵢ × cᵢ) / Σ(μᵢ)")
    linhas.append("")
    linhas.append("-" * 62)
    linhas.append("RESULTADOS DOS TESTES:")
    linhas.append("-" * 62)

    for temp in temperaturas_teste:
        fuzzy, ativ, veloc = sistema_fuzzy(temp)

        linhas.append("")
        linhas.append(f"  ▶ Temperatura de entrada : {temp}°C")
        linhas.append(f"    Fuzzificação (graus de pertinência):")
        linhas.append(f"      - FRIA        : {fuzzy['FRIA']}")
        linhas.append(f"      - CONFORTÁVEL : {fuzzy['CONFORTAVEL']}")
        linhas.append(f"      - QUENTE      : {fuzzy['QUENTE']}")
        linhas.append(f"    Ativação das Regras:")
        linhas.append(f"      - LENTO  : {ativ['LENTO']}")
        linhas.append(f"      - MÉDIO  : {ativ['MEDIO']}")
        linhas.append(f"      - RÁPIDO : {ativ['RAPIDO']}")
        linhas.append(f"    ✔ Velocidade do Ventilador (saída crisp): {veloc}%")

        # Barra visual
        barra = int(veloc / 5)
        linhas.append(f"    [{'█' * barra}{'░' * (20 - barra)}] {veloc}%")

    linhas.append("")
    linhas.append("=" * 62)
    linhas.append("  VANTAGENS DA LÓGICA FUZZY EM MACHINE LEARNING:")
    linhas.append("=" * 62)
    linhas.append("  1. Lida com incerteza e imprecisão de forma natural")
    linhas.append("  2. Regras interpretáveis (linguagem humana)")
    linhas.append("  3. Não requer grande volume de dados rotulados")
    linhas.append("  4. Base para sistemas híbridos (Neuro-Fuzzy, ANFIS)")
    linhas.append("  5. Útil em controle, classificação e tomada de decisão")
    linhas.append("")
    linhas.append("  APLICAÇÕES COMUNS:")
    linhas.append("  • Controle industrial (temperatura, velocidade, pressão)")
    linhas.append("  • Sistemas de recomendação")
    linhas.append("  • Diagnóstico médico assistido")
    linhas.append("  • Processamento de linguagem natural")
    linhas.append("  • Robótica e veículos autônomos")
    linhas.append("")
    linhas.append("=" * 62)
    linhas.append("  Fim do Relatório")
    linhas.append("=" * 62)

    # ── Criar pasta 'resultado' e salvar o arquivo .txt ──
    caminho_destino = (
        r"C:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist"
        r"\Teoria do Aprendizado Estatístico\Métodos de Otimização de Aprendizado"
        r"\resultado"
    )
    os.makedirs(caminho_destino, exist_ok=True)

    arquivo_saida = os.path.join(caminho_destino, "resultado_fuzzy.txt")
    conteudo = "\n".join(linhas)

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(conteudo)

    # Exibe no console também
    print(conteudo)
    print(f"\n✔ Arquivo salvo em:\n  {arquivo_saida}")


if __name__ == "__main__":
    main()
