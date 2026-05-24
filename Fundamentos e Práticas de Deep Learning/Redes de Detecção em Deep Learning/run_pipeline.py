"""
run_pipeline.py
===============
Script auxiliar para executar o pipeline completo do YOLO no Open Images de ponta a ponta:
1. Download e Conversão do Dataset (Car, Person, Bicycle)
2. Treinamento
3. Inferência/Teste de Detecção
4. Validação do Modelo

Uso:
    & .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/run_pipeline.py"
"""

from yolo_openimages import download_and_convert, train, infer, validate, RESULT_DIR
import sys
from pathlib import Path

# Configurar diretórios locais
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Importar as funções do script principal

# ─────────────────────────────────────────────────────────────────────────────
# PARÂMETROS CONFIGURÁVEIS
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["Car", "Person", "Bicycle"]
# Quantidade de amostras a baixar (aumente para treinar um modelo melhor)
SAMPLES = 100
# Número de épocas (aumente para treinar mais a fundo, ex: 30)
EPOCHS = 5
BATCH_SIZE = 8      # Tamanho do batch (seguro para CPU/GPU simples)
DEVICE = "auto"     # "auto" detecta GPU se houver, caso contrário usa CPU
MODEL_BASE = "yolov8n"  # Arquitetura base (yolov8n é a mais rápida)

# Diretório onde o dataset será salvo
DATASET_DIR = SCRIPT_DIR / "datasets" / "openimages_yolo"


def main():
    print("=" * 60)
    # 1. Download e Conversão
    print("\n--- PASSO 1: BAIXANDO E PREPARANDO DATASET ---")
    yaml_path = download_and_convert(
        classes=CLASSES,
        samples=SAMPLES,
        split="validation",
        out_dir=DATASET_DIR
    )

    # 2. Treinamento
    print("\n--- PASSO 2: INICIANDO TREINAMENTO YOLO ---")
    best_weights = train(
        yaml_path=yaml_path,
        model_name=MODEL_BASE,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        device=DEVICE
    )

    # 3. Inferência (Teste de Detecção)
    print("\n--- PASSO 3: INFERÊNCIA / DETECÇÃO NAS IMAGENS ---")
    # Usando a pasta de imagens baixadas para fazer o teste
    images_source = str(DATASET_DIR / "images")
    infer(
        weights=str(best_weights),
        source=images_source,
        conf=0.25,
        save=True
    )

    # 4. Validação
    print("\n--- PASSO 4: VALIDANDO MÉTRICAS DO MODELO ---")
    validate(
        weights=str(best_weights),
        data_yaml=str(yaml_path)
    )

    print("\n" + "=" * 60)
    print("[OK] Pipeline completo executado com sucesso!")
    print(
        f"* Todos os resultados e resumos .txt estão em: {RESULT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
