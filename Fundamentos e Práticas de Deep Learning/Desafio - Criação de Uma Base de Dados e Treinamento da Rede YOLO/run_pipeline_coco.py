"""
run_pipeline_coco.py
====================
Script auxiliar para executar o pipeline completo do YOLO no COCO de ponta a ponta:
1. Download e Conversão do Dataset (car, person, bicycle)
2. Treinamento
3. Inferência/Teste de Detecção
4. Validação do Modelo

Diferença em relação ao Open Images:
    - Usa COCO-2017 (sem necessidade de credenciais AWS)
    - Nomes de classes em minúsculo: "car", "person", "bicycle"
    - Dataset mais padronizado e amplamente utilizado na literatura

Uso:
    python run_pipeline_coco.py

    Ou com venv:
    .venv/Scripts/python.exe run_pipeline_coco.py
"""

from yolo_coco import download_and_convert, train, infer, validate, RESULT_DIR
import sys
from pathlib import Path

# Configurar diretórios
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS CONFIGURÁVEIS
# ─────────────────────────────────────────────────────────────────────────────

# ATENÇÃO: no COCO os nomes de classe são em minúsculo
# (diferente do Open Images que usa "Car", "Person", "Bicycle")
CLASSES = ["car", "person", "bicycle"]

# Quantidade de amostras a baixar por classe (aumente para melhor resultado)
SAMPLES = 100

# Número de épocas de treinamento (aumente para treinar mais, ex: 30-50)
EPOCHS = 5

# Tamanho do batch (reduza para 4 se der erro de memória)
BATCH_SIZE = 8

# "auto" detecta GPU automaticamente; use "cpu" para forçar CPU
DEVICE = "auto"

# Arquitetura base: yolov8n (mais rápida) | yolov8s | yolov8m (mais precisa)
MODEL_BASE = "yolov8n"

# Diretório onde o dataset COCO processado será salvo
DATASET_DIR = SCRIPT_DIR / "datasets" / "coco_yolo"

# Split para download ("validation" é menor e mais rápido que "train")
SPLIT = "validation"


def main():
    print("=" * 60)
    print("  PIPELINE YOLO + COCO-2017")
    print("=" * 60)

    # ── PASSO 1: Download e Conversão ────────────────────────────────────────
    print("\n--- PASSO 1: BAIXANDO E PREPARANDO DATASET COCO ---")
    print(f"  Classes   : {CLASSES}")
    print(f"  Amostras  : {SAMPLES}")
    print(f"  Split     : {SPLIT}")

    yaml_path = download_and_convert(
        classes=CLASSES,
        samples=SAMPLES,
        split=SPLIT,
        out_dir=DATASET_DIR,
    )

    # ── PASSO 2: Treinamento ─────────────────────────────────────────────────
    print("\n--- PASSO 2: INICIANDO TREINAMENTO YOLO ---")
    best_weights = train(
        yaml_path=yaml_path,
        model_name=MODEL_BASE,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        device=DEVICE,
    )

    # ── PASSO 3: Inferência ──────────────────────────────────────────────────
    print("\n--- PASSO 3: INFERÊNCIA / DETECÇÃO NAS IMAGENS ---")
    images_source = str(DATASET_DIR / "images" / SPLIT)
    infer(
        weights=str(best_weights),
        source=images_source,
        conf=0.25,
        save=True,
    )

    # ── PASSO 4: Validação ───────────────────────────────────────────────────
    print("\n--- PASSO 4: VALIDANDO MÉTRICAS DO MODELO ---")
    validate(
        weights=str(best_weights),
        data_yaml=str(yaml_path),
    )

    # ── Resumo final ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] Pipeline COCO completo executado com sucesso!")
    print(f"* Resultados e resumos .txt em: {RESULT_DIR.resolve()}")
    print("=" * 60)
    print("\nEstrutura de arquivos gerada:")
    print(f"  datasets/coco_yolo/")
    print(f"    images/{SPLIT}/     ← imagens COCO baixadas")
    print(f"    labels/{SPLIT}/     ← labels no formato YOLO")
    print(f"    dataset.yaml        ← configuração do dataset")
    print(f"  resultados_coco/")
    print(f"    resultado_treino.txt")
    print(f"    resultado_inferencia.txt")
    print(f"    resultado_validacao.txt")
    print(f"    runs/train/         ← pesos e gráficos do treino")
    print(f"    runs/detect/        ← imagens com bounding boxes")
    print(f"    runs/val/           ← métricas de validação")
    print("=" * 60)


if __name__ == "__main__":
    main()
