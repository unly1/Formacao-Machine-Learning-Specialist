"""
yolo_coco.py
============
Script completo: baixa subconjunto do COCO via FiftyOne, converte para
formato YOLO e treina / roda inferência / valida com YOLOv8.

Requisitos:
    pip install ultralytics fiftyone opencv-python pyyaml torch

Uso — treinar:
    python yolo_coco.py train --classes car person bicycle --samples 300

Uso — inferência em imagem/vídeo/webcam:
    python yolo_coco.py infer --weights runs/train/best.pt --source foto.jpg
    python yolo_coco.py infer --weights best.pt --source video.mp4
    python yolo_coco.py infer --weights best.pt --source 0

Uso — validação:
    python yolo_coco.py val --weights best.pt --data datasets/coco_yolo/dataset.yaml

Diferença em relação ao Open Images:
    - Usa o dataset COCO-2017 (detection) via FiftyOne Zoo
    - Nomes de classes em minúsculo (ex: "car", "person", "bicycle")
    - Sem necessidade de credenciais AWS (Open Images exige)
    - Mais classes disponíveis: 80 categorias COCO
"""

import argparse
import shutil
from pathlib import Path

import yaml

# ─────────────────────────────────────────────────────────────────────────────
# DIRETÓRIOS DE RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "resultados_coco"
RUNS_DIR = RESULT_DIR / "runs"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DETECÇÃO AUTOMÁTICA DE DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def auto_device(requested: str) -> str:
    """
    Detecta automaticamente se CUDA está disponível.
    - 'cpu'  → força CPU
    - 'auto' → testa CUDA; usa CPU se não houver GPU
    - '0'    → força GPU 0
    """
    import torch

    if requested.lower() == "cpu":
        print("      Device  : CPU (solicitado pelo usuário)")
        return "cpu"

    if torch.cuda.is_available():
        nome = torch.cuda.get_device_name(0)
        n = torch.cuda.device_count()
        print(f"      Device  : GPU '{nome}' ({n} disponível{'is' if n > 1 else ''})")
        return "0" if requested == "auto" else requested

    print("      Device  : CPU")
    print("      ⚠  CUDA não encontrado — treinando na CPU (mais lento).")
    print("         Para usar GPU: https://pytorch.org/get-started/locally/\n")
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD + CONVERSÃO  (COCO → formato YOLO)
# ─────────────────────────────────────────────────────────────────────────────

# Classes disponíveis no COCO (80 categorias):
# person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
# traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
# dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
# handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
# baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
# wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
# broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
# bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
# microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
# teddy bear, hair drier, toothbrush

def download_and_convert(
    classes: list,
    samples: int,
    split: str,
    out_dir: Path,
) -> Path:
    """
    Baixa amostras do COCO-2017 via FiftyOne e converte para formato YOLO.

    Formato YOLO por linha no .txt:
        <class_id> <cx> <cy> <width> <height>   (normalizados 0-1)

    Parâmetros:
        classes : lista de classes COCO (minúsculo, ex: ["car", "person", "bicycle"])
        samples : número máximo de imagens a baixar
        split   : "train", "validation" ou "test"
        out_dir : pasta de saída para imagens e labels

    Retorna o caminho do dataset.yaml gerado.
    """
    import fiftyone.zoo as foz

    print(f"\n[1/3] Baixando COCO-2017")
    print(f"      Classes  : {classes}")
    print(f"      Amostras : {samples}")
    print(f"      Split    : {split}\n")

    # FiftyOne usa "validation" para o split de validação do COCO
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"],
        classes=classes,
        max_samples=samples,
        dataset_name=f"coco_{'_'.join(c[:4] for c in classes[:3])}_{samples}",
        overwrite=True,
    )

    # Mapeamento classe → id (ordem alfabética para consistência)
    sorted_classes = sorted(classes)
    class_map = {name: idx for idx, name in enumerate(sorted_classes)}

    print(f"\n[2/3] Convertendo para formato YOLO")
    print(f"      Mapeamento: {class_map}\n")

    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    converted, skipped = 0, 0

    for sample in dataset:
        dets = sample.ground_truth.detections if sample.ground_truth else []

        # Filtrar apenas detecções das classes solicitadas
        valid = [d for d in dets if d.label in class_map]
        if not valid:
            skipped += 1
            continue

        # Copiar imagem
        src = Path(sample.filepath)
        shutil.copy2(src, img_dir / src.name)

        # Escrever arquivo de label no formato YOLO
        dst_lbl = lbl_dir / (src.stem + ".txt")
        with open(dst_lbl, "w") as f:
            for det in valid:
                # FiftyOne: bounding_box = [x_top_left, y_top_left, width, height] normalizados
                bx, by, bw, bh = det.bounding_box
                cx = bx + bw / 2
                cy = by + bh / 2
                cid = class_map[det.label]
                f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        converted += 1

    print(f"      Convertidas : {converted}")
    print(f"      Puladas     : {skipped} (sem objetos das classes pedidas)")

    if converted == 0:
        raise RuntimeError(
            "Nenhuma imagem foi convertida! Verifique se os nomes das classes "
            "estão corretos (minúsculo, ex: 'car', 'person', 'bicycle')."
        )

    # Gerar dataset.yaml compatível com YOLOv8
    yaml_path = out_dir / "dataset.yaml"
    cfg = {
        "path":  str(out_dir.resolve()),
        "train": f"images/{split}",
        "val":   f"images/{split}",
        "nc":    len(sorted_classes),
        "names": sorted_classes,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"\n      dataset.yaml -> {yaml_path}")
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. TREINAMENTO  (YOLOv8)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    yaml_path: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Path:
    """Treina o modelo YOLOv8 no dataset COCO preparado."""
    from ultralytics import YOLO

    device = auto_device(device)

    print(f"\n[3/3] Treinamento YOLO")
    print(f"      Modelo  : {model_name}")
    print(f"      Épocas  : {epochs}")
    print(f"      Imgsz   : {imgsz}")
    print(f"      Batch   : {batch}")
    print(f"      Device  : {device}\n")

    model = YOLO(f"{model_name}.pt")  # baixa pesos pré-treinados automaticamente

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR / "train"),
        name=f"{model_name}_coco",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\n  [OK] Treinamento concluído!")
    print(f"  * Melhores pesos : {best}")
    print(f"  * Resultados     : {results.save_dir}\n")

    # Salvar resumo em .txt
    txt_path = RESULT_DIR / "resultado_treino.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== RESULTADO DO TREINAMENTO (COCO) ===\n")
        f.write(f"Modelo base       : {model_name}\n")
        f.write(f"Épocas            : {epochs}\n")
        f.write(f"Tamanho da imagem : {imgsz}\n")
        f.write(f"Batch size        : {batch}\n")
        f.write(f"Dispositivo       : {device}\n")
        f.write(f"Dataset YAML      : {yaml_path}\n")
        f.write(f"Pasta de resultados: {results.save_dir}\n")
        f.write(f"Melhores pesos    : {best.resolve()}\n")
    print(f"  [OK] Resumo do treino salvo em: {txt_path}\n")

    return best


# ─────────────────────────────────────────────────────────────────────────────
# 3. INFERÊNCIA
# ─────────────────────────────────────────────────────────────────────────────

def infer(weights: str, source: str, conf: float, save: bool) -> None:
    """Roda detecção com o modelo treinado em uma fonte (imagem/vídeo/webcam)."""
    from ultralytics import YOLO

    print(f"\n[Inferência]")
    print(f"  Pesos    : {weights}")
    print(f"  Fonte    : {source}")
    print(f"  Confiança: {conf}\n")

    model = YOLO(weights)
    results = model.predict(
        source=source,
        conf=conf,
        save=save,
        save_txt=False,
        show=not save,
        line_width=2,
        verbose=True,
        project=str(RUNS_DIR / "detect"),
        name="predict",
        exist_ok=True,
    )

    # Salvar resumo das detecções
    txt_path = RESULT_DIR / "resultado_inferencia.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== RESULTADO DA INFERÊNCIA (COCO) ===\n")
        f.write(f"Pesos utilizados  : {Path(weights).resolve()}\n")
        f.write(f"Fonte das imagens : {source}\n")
        f.write(f"Limiar de confiança: {conf}\n\n")

        for r in results:
            img_name = Path(r.path).name
            if r.boxes is None or len(r.boxes) == 0:
                line = f"{img_name}: nenhum objeto detectado\n"
                f.write(line)
                print(f"  {line.strip()}")
                continue

            f.write(f"{img_name}:\n")
            for cls_id, conf_val in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                class_name = model.names[int(cls_id)]
                line = f"  - {class_name}: {conf_val:.2f}\n"
                f.write(line)
                print(f"  {img_name}: {class_name} ({conf_val:.2f})")

    if save:
        save_dir = RUNS_DIR / "detect" / "predict"
        print(f"\n  [OK] Imagens com detecções salvas em: {save_dir}")
    print(f"  [OK] Detecções salvas em: {txt_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALIDAÇÃO  (métricas mAP)
# ─────────────────────────────────────────────────────────────────────────────

def validate(weights: str, data_yaml: str) -> None:
    """Calcula métricas mAP no dataset de validação."""
    from ultralytics import YOLO

    print(f"\n[Validação]")
    print(f"  Pesos   : {weights}")
    print(f"  Dataset : {data_yaml}\n")

    model = YOLO(weights)
    metrics = model.val(
        data=data_yaml,
        verbose=True,
        project=str(RUNS_DIR / "val"),
        name="validation",
        exist_ok=True,
    )

    print("\n  ── Métricas ─────────────────────────")
    print(f"  mAP@50      : {metrics.box.map50:.4f}")
    print(f"  mAP@50-95   : {metrics.box.map:.4f}")
    print(f"  Precisão    : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("  ─────────────────────────────────────\n")

    txt_path = RESULT_DIR / "resultado_validacao.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== RESULTADO DA VALIDAÇÃO (COCO) ===\n")
        f.write(f"Pesos utilizados : {Path(weights).resolve()}\n")
        f.write(f"Dataset YAML     : {Path(data_yaml).resolve()}\n\n")
        f.write("Métricas de Box:\n")
        f.write(f"  mAP@50      : {metrics.box.map50:.4f}\n")
        f.write(f"  mAP@50-95   : {metrics.box.map:.4f}\n")
        f.write(f"  Precisão    : {metrics.box.mp:.4f}\n")
        f.write(f"  Recall      : {metrics.box.mr:.4f}\n")
    print(f"  [OK] Métricas salvas em: {txt_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLO + COCO-2017 — pipeline completo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Baixar COCO e treinar YOLO")
    p_train.add_argument(
        "--classes", nargs="+", required=True,
        help=(
            "Classes do COCO em minúsculo (ex: car person bicycle)\n"
            "80 classes disponíveis: person, bicycle, car, motorcycle,\n"
            "airplane, bus, train, truck, boat, dog, cat, chair, etc."
        )
    )
    p_train.add_argument("--samples", type=int, default=300,
                         help="Quantidade de imagens a baixar (default: 300)")
    p_train.add_argument("--split", default="validation",
                         choices=["train", "validation", "test"],
                         help="Split do COCO (default: validation)")
    p_train.add_argument("--model", default="yolov8n",
                         choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
                                  "yolo11n", "yolo11s", "yolo11m"],
                         help="Arquitetura YOLO (default: yolov8n = mais rápido)")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--imgsz",  type=int, default=640)
    p_train.add_argument("--batch",  type=int, default=8,
                         help="Tamanho do batch (default: 8 — seguro para CPU)")
    p_train.add_argument("--device", default="auto",
                         help="'auto' detecta GPU/CPU | 'cpu' força CPU | '0' força GPU 0")
    p_train.add_argument("--out-dir", default="datasets/coco_yolo",
                         help="Pasta de saída do dataset convertido")

    # ── infer ──────────────────────────────────────────────────────────────
    p_infer = sub.add_parser("infer", help="Rodar inferência com modelo treinado")
    p_infer.add_argument("--weights", required=True, help="Caminho para best.pt")
    p_infer.add_argument("--source",  required=True,
                         help="Imagem, vídeo, diretório ou '0' para webcam")
    p_infer.add_argument("--conf",    type=float, default=0.25,
                         help="Limiar de confiança (default: 0.25)")
    p_infer.add_argument("--save",    action="store_true",
                         help="Salvar resultado em vez de exibir na tela")

    # ── val ────────────────────────────────────────────────────────────────
    p_val = sub.add_parser("val", help="Calcular métricas mAP no dataset")
    p_val.add_argument("--weights", required=True)
    p_val.add_argument("--data",    required=True, help="dataset.yaml")

    args = parser.parse_args()

    if args.cmd == "train":
        out_dir = Path(args.out_dir)
        yaml_path = download_and_convert(
            classes=args.classes,
            samples=args.samples,
            split=args.split,
            out_dir=out_dir,
        )
        train(
            yaml_path=yaml_path,
            model_name=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )

    elif args.cmd == "infer":
        if not Path(args.weights).exists():
            parser.error(f"Pesos não encontrados: {args.weights}")
        infer(args.weights, args.source, args.conf, args.save)

    elif args.cmd == "val":
        for p in (args.weights, args.data):
            if not Path(p).exists():
                parser.error(f"Arquivo não encontrado: {p}")
        validate(args.weights, args.data)


if __name__ == "__main__":
    main()
