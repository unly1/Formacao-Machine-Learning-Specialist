"""
main.py
-------
Ponto de entrada principal do projeto.
Orquestra todas as etapas: download, treino e inferência.

Uso rápido:
    python main.py --mode all           # pipeline completo
    python main.py --mode download      # só baixa o dataset
    python main.py --mode train         # só treina (ambos backends)
    python main.py --mode image --input foto.jpg
    python main.py --mode webcam
"""

import argparse
import subprocess
import sys
from pathlib import Path


STEPS = {
    "download": ("src/download_dataset.py",        "Preparação do Dataset LFW"),
    "train_tf": ("src/train_tensorflow.py",         "Treinamento TensorFlow"),
    "train_fr": ("src/train_face_recognition.py",   "Treinamento face_recognition"),
}


def run(script: str, extra_args: list = None):
    cmd = [sys.executable, script] + (extra_args or [])
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[!] Erro ao executar: {script}")
        sys.exit(result.returncode)


def banner(text: str):
    width = 57
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Reconhecimento Facial — Orquestrador",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["all", "download", "train", "train_tf", "train_fr", "image", "webcam"],
        help=(
            "all       → pipeline completo (download + treino + webcam)\n"
            "download  → baixa e prepara o dataset LFW\n"
            "train     → treina ambos os backends\n"
            "train_tf  → treina apenas TensorFlow\n"
            "train_fr  → treina apenas face_recognition\n"
            "image     → detecção em imagem (requer --input)\n"
            "webcam    → detecção em tempo real via webcam"
        ),
    )
    parser.add_argument("--input",      help="Caminho da imagem ou pasta (modo image)")
    parser.add_argument("--backend",    default="fr",  choices=["tensorflow", "fr"])
    parser.add_argument("--classifier", default="svm", choices=["svm", "knn"])
    parser.add_argument("--camera",     default=0,     type=int)
    parser.add_argument("--show",       action="store_true", help="Exibe janela de resultado")
    args = parser.parse_args()

    # ── Download ──────────────────────────────────────────────
    if args.mode in ("all", "download"):
        banner("Etapa 1 — Download e Preparo do Dataset")
        run("src/download_dataset.py")

    # ── Treino ────────────────────────────────────────────────
    if args.mode in ("all", "train", "train_tf"):
        banner("Etapa 2a — Treinamento TensorFlow")
        run("src/train_tensorflow.py")

    if args.mode in ("all", "train", "train_fr"):
        banner("Etapa 2b — Treinamento face_recognition")
        run("src/train_face_recognition.py")

    # ── Detecção em Imagem ────────────────────────────────────
    if args.mode == "image":
        if not args.input:
            print("[!] --input é obrigatório no modo 'image'")
            sys.exit(1)
        banner("Detecção em Imagem")
        extra = ["--input", args.input,
                 "--backend", args.backend,
                 "--classifier", args.classifier]
        if args.show:
            extra.append("--show")
        run("src/detect_images.py", extra)

    # ── Webcam ────────────────────────────────────────────────
    if args.mode in ("all", "webcam"):
        banner("Detecção em Tempo Real — Webcam")
        run("src/detect_webcam.py", [
            "--backend",    args.backend,
            "--classifier", args.classifier,
            "--camera",     str(args.camera),
        ])

    print("\n[✓] Concluído!\n")


if __name__ == "__main__":
    main()
