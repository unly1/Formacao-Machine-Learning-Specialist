# Segmentação de Imagens com Mask R-CNN

Este projeto contém um script básico em Python para realizar **Segmentação de Imagens (Instance Segmentation)** utilizando a rede **Mask R-CNN** com o modelo `resnet50_fpn` pré-treinado na base de dados COCO.

O modelo é capaz de detectar e contornar objetos em nível de pixel (gerando máscaras), além de prever suas respectivas classes e caixas delimitadoras (bounding boxes).

---

## 🛠️ Requisitos

Os seguintes pacotes são necessários para executar o script:
* **PyTorch** (`torch` e `torchvision`)
* **OpenCV** (`opencv-python` para manipulação de imagens e desenho das segmentações)
* **Pillow** (`PIL` para leitura robusta de caminhos com acentuação)
* **NumPy**

### Instalação rápida:
Utilize o ambiente virtual do seu projeto para rodar a instalação:
```bash
pip install torch torchvision opencv-python pillow numpy
```

---

## 📂 Estrutura do Projeto

A organização dos arquivos e pastas segue o seguinte padrão:

```text
Redes de Segmentação em Deep Learning/
│
├── mask_rcnn_segmentation.py     # Script principal em Python
├── README.md                     # Documentação do projeto
│
├── Imagens/                      # Pasta contendo imagens de entrada/teste
│   └── dining_table.jpg          # Exemplo de imagem de entrada
│
└── Resultado/                    # Pasta que armazena os resultados processados
    ├── resultado.txt             # Arquivo com logs detalhados das detecções
    └── Imagens/                  # Pasta para as imagens segmentadas de saída
        └── resultado_dining_table.jpg  # Imagem processada com máscaras e boxes
```

---

## 🚀 Como Executar

Abra o terminal no diretório do projeto e execute com o interpretador Python do seu ambiente virtual:

### 1. Execução Padrão (Sem argumentos)
Se você rodar o script sem nenhum argumento, ele irá procurar por imagens dentro da pasta `Imagens/`. Se não encontrar nenhuma, fará o download de uma imagem de teste ou gerará uma imagem sintética automaticamente.
```bash
.venv\Scripts\python.exe "Fundamentos e Práticas de Deep Learning\Redes de Segmentação em Deep Learning\mask_rcnn_segmentation.py"
```

### 2. Processando uma Imagem Específica
Use o argumento `--source` passando o caminho da imagem que deseja segmentar:
```bash
.venv\Scripts\python.exe "Fundamentos e Práticas de Deep Learning\Redes de Segmentação em Deep Learning\mask_rcnn_segmentation.py" --source "caminho/para/sua_imagem.jpg"
```
> **Nota:** Se a imagem enviada estiver fora da pasta `Imagens/`, o script salvará automaticamente uma cópia dela na pasta local `Imagens/` para manter o histórico de testes.

### 3. Ajustando o Limiar de Confiança (Threshold)
Por padrão, o modelo exibe apenas detecções com confiança superior a `0.50` (`50%`). Você pode alterar esse valor com o argumento `--threshold`:
```bash
.venv\Scripts\python.exe "Fundamentos e Práticas de Deep Learning\Redes de Segmentação em Deep Learning\mask_rcnn_segmentation.py" --threshold 0.75
```

### 4. Forçando uso de GPU (CUDA) ou CPU
O dispositivo (`cuda` ou `cpu`) é detectado automaticamente de acordo com seu hardware. Caso queira forçar um dispositivo específico:
```bash
.venv\Scripts\python.exe "Fundamentos e Práticas de Deep Learning\Redes de Segmentação em Deep Learning\mask_rcnn_segmentation.py" --device cpu
```

---

## 📊 Resultados Gerados

Ao finalizar o processamento, o script gera dois arquivos principais:

1. **Relatório em Texto (`Resultado/resultado.txt`)**:
   Registra o nome da imagem, número de detecções e os detalhes de cada objeto segmentado (Nome do objeto, nível de certeza do modelo e as coordenadas da caixa delimitadora `[xmin, ymin, xmax, ymax]`).
   
2. **Imagem Segmentada (`Resultado/Imagens/resultado_<nome>.jpg`)**:
   Salva a imagem original contendo a sobreposição de cores translúcidas (máscaras) sobre cada objeto detectado, suas caixas delimitadoras e etiquetas identificadoras.
