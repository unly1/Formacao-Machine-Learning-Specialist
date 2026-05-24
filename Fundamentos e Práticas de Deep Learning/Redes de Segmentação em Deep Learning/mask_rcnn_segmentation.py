"""
mask_rcnn_segmentation.py
=========================
Script básico de segmentação de imagens utilizando a rede Mask R-CNN.
Carrega um modelo pré-treinado do torchvision, executa inferência em uma imagem,
gera uma imagem segmentada e salva o relatório em um arquivo de texto.

Requisitos:
    pip install torch torchvision opencv-python pillow numpy

Execução:
    python mask_rcnn_segmentation.py --source minha_imagem.jpg
    python mask_rcnn_segmentation.py (baixa ou gera uma imagem de teste automaticamente)
"""

import os
import sys
import argparse
import random
import urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import functional as F

# Configuração do OpenCV
try:
    import cv2
except ImportError:
    print("Erro: O pacote 'opencv-python' é necessário. Instale-o com: pip install opencv-python")
    sys.exit(1)

# Caminhos
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "Resultado"
IMAGES_DIR = SCRIPT_DIR / "Imagens"
RESULT_IMAGES_DIR = RESULT_DIR / "Imagens"

# Criar as pastas se não existirem
RESULT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Classes do COCO dataset (usado pelo modelo pré-treinado)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_fallback_image() -> Path:
    """Tenta encontrar uma imagem local, baixar da internet ou criar uma sintética."""
    # 1. Procurar por imagens locais na pasta Imagens
    local_images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    if local_images:
        print(f"Imagem local encontrada na pasta Imagens e utilizada: {local_images[0]}")
        return local_images[0]
    
    # 2. Procurar em outras pastas de datasets do projeto
    parent_dir = SCRIPT_DIR.parent
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        found = list(parent_dir.rglob(ext))
        # Filtrar para não pegar pastas do sistema ou .venv
        found = [f for f in found if ".venv" not in f.parts and "runs" not in f.parts and "Resultado" not in f.parts and "Imagens" not in f.parts]
        if found:
            print(f"Imagem local encontrada em pasta vizinha e utilizada: {found[0]}")
            return found[0]

    # 3. Tentar baixar da internet
    img_name = "test_image.jpg"
    dest_path = IMAGES_DIR / img_name
    url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
    try:
        print(f"Baixando imagem de teste de: {url}")
        # Definir User-Agent para evitar erros 403 HTTP com alguns servidores
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Imagem salva com sucesso em: {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Falha ao baixar imagem de teste ({e}). Criando uma imagem sintética...")
        
    # 4. Criar imagem sintética se tudo falhar
    synthetic_path = IMAGES_DIR / "synthetic_test.jpg"
    # Criar uma imagem escura de 600x600
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    # Desenhar formas para testar
    cv2.circle(img, (300, 300), 100, (0, 0, 255), -1) # Círculo vermelho
    cv2.rectangle(img, (100, 100), (250, 250), (0, 255, 0), -1) # Retângulo verde
    cv2.putText(img, "Teste Mask R-CNN", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.imwrite(str(synthetic_path), img)
    print(f"Imagem sintética criada em: {synthetic_path}")
    return synthetic_path

def main():
    parser = argparse.ArgumentParser(description="Segmentação de Imagem com Mask R-CNN")
    parser.add_argument("--source", type=str, default=None, help="Caminho para a imagem de entrada")
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar de confiança (score mínimo)")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo para rodar o modelo ('cpu', 'cuda', 'auto')")
    args = parser.parse_args()

    # Determinar dispositivo
    if args.device.lower() == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Dispositivo selecionado: {device}")

    # Obter caminho da imagem
    if args.source:
        image_path = Path(args.source)
        if not image_path.exists():
            print(f"Erro: Arquivo {image_path} não encontrado.")
            sys.exit(1)
    else:
        image_path = get_fallback_image()

    # Garantir que a imagem de entrada esteja salva na pasta Imagens local
    if image_path.parent.resolve() != IMAGES_DIR.resolve():
        try:
            import shutil
            dest_copied = IMAGES_DIR / image_path.name
            shutil.copy2(image_path, dest_copied)
            print(f"Imagem de entrada copiada para a pasta Imagens: {dest_copied.name}")
            image_path = dest_copied
        except Exception as e:
            print(f"Aviso: Não foi possível copiar a imagem para a pasta Imagens ({e})")

    # Carregar imagem
    print(f"Carregando a imagem: {image_path.name}")
    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erro ao abrir imagem: {e}")
        sys.exit(1)

    # Converter imagem para tensor PyTorch
    image_tensor = F.to_tensor(pil_image).unsqueeze(0).to(device)

    # Carregar modelo pré-treinado Mask R-CNN
    print("Carregando o modelo Mask R-CNN pré-treinado (ResNet-50 FPN)...")
    try:
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        model = maskrcnn_resnet50_fpn(weights=weights)
    except ImportError:
        # Fallback para versões mais antigas do torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        model = maskrcnn_resnet50_fpn(pretrained=True)

    model = model.to(device)
    model.eval()

    print("Executando inferência de segmentação...")
    with torch.no_grad():
        predictions = model(image_tensor)

    # Carregar imagem original no formato OpenCV para desenho
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        original_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    h, w, _ = original_image.shape
    output_image = original_image.copy()

    # Processar predições
    pred = predictions[0]
    boxes = pred["boxes"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    masks = pred["masks"].cpu().numpy()

    # Filtrar por limiar de confiança (threshold)
    keep = scores >= args.threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    masks = masks[keep]

    print(f"Quantidade de objetos detectados acima do threshold ({args.threshold}): {len(boxes)}")

    # Preparar arquivo de resultado texto
    txt_path = RESULT_DIR / "resultado.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== RESULTADO DA SEGMENTAÇÃO MASK R-CNN ===\n")
        f.write(f"Imagem processada: {image_path.name}\n")
        f.write(f"Modelo: Mask R-CNN ResNet-50 FPN (COCO)\n")
        f.write(f"Dispositivo: {device}\n")
        f.write(f"Threshold de confiança: {args.threshold}\n")
        f.write(f"Objetos detectados: {len(boxes)}\n\n")

        # Gerar cores para cada objeto de forma consistente
        random.seed(42)

        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            label_idx = labels[i]
            score = scores[i]
            mask = masks[i, 0] # Dimensão [H, W]

            class_name = COCO_CLASSES[label_idx] if label_idx < len(COCO_CLASSES) else f"ID {label_idx}"
            xmin, ymin, xmax, ymax = box

            # Escrever no arquivo texto
            f.write(f"Objeto #{i+1}: {class_name}\n")
            f.write(f"  Confiança: {score:.4f}\n")
            f.write(f"  Box: [{xmin}, {ymin}, {xmax}, {ymax}]\n\n")

            print(f"  Detecção: {class_name} ({score:.2f}) - Box: [{xmin}, {ymin}, {xmax}, {ymax}]")

            # Gerar cor aleatória (BGR)
            color = [random.randint(0, 255) for _ in range(3)]

            # Desenhar máscara (sobreposição semi-transparente)
            mask_boolean = mask > 0.5
            
            # Ajustar tamanho caso haja discrepância
            if mask_boolean.shape[:2] != (h, w):
                mask_boolean = cv2.resize(mask_boolean.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            
            # Criar um overlay colorido para a máscara
            mask_overlay = output_image.copy()
            mask_overlay[mask_boolean] = color
            
            # Combinar imagem com overlay da máscara (50% de opacidade)
            cv2.addWeighted(mask_overlay, 0.5, output_image, 0.5, 0, output_image)

            # Desenhar caixa delimitadora
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)

            # Desenhar label de texto
            label_text = f"{class_name} {score:.2f}"
            
            # Calcular o tamanho do texto para o fundo do texto
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Evitar desenhar fora do limite superior da imagem
            text_ymin = max(ymin, text_h + 10)
            cv2.rectangle(output_image, (xmin, text_ymin - text_h - 4), (xmin + text_w, text_ymin), color, -1)
            
            cv2.putText(
                output_image,
                label_text,
                (xmin, text_ymin - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

    # Salvar a imagem segmentada de forma segura para caminhos com caracteres Unicode no Windows
    output_image_path = RESULT_IMAGES_DIR / f"resultado_{image_path.stem}.jpg"
    success, im_buf_arr = cv2.imencode(".jpg", output_image)
    if success:
        try:
            with open(output_image_path, "wb") as f_img:
                f_img.write(im_buf_arr.tobytes())
        except Exception as e:
            print(f"Erro ao escrever o arquivo de imagem segmentada ({e})")
    else:
        print("Erro ao codificar a imagem segmentada.")
    
    print("\n[OK] Processamento concluído com sucesso!")
    print(f"Relatório de detecção salvo em: {txt_path.resolve()}")
    print(f"Imagem segmentada salva em: {output_image_path.resolve()}\n")

if __name__ == "__main__":
    main()
