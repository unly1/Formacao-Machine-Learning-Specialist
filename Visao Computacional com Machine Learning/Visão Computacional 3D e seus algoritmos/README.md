# 🧊 Visão Computacional 3D e seus Algoritmos

Este módulo aborda os fundamentos e a prática da Visão Computacional 3D, cobrindo desde os conceitos teóricos introdutórios até a implementação de algoritmos modernos de deep learning para análise de dados tridimensionais.

---

## 📚 Estrutura do Módulo

| Aula | Tema | Tipo |
|------|------|------|
| Aula 1 | Introdução à Visão Computacional 3D | Teórica |
| Aula 2 | Sensores para Visão Computacional 3D | Teórica |
| Aula 3 | Geração e Manipulação de Imagens 3D | Teórica + Prática |
| Aula 4 | Classificação de Imagens em Nuvem de Pontos | Teórica + Prática |

---

## 🎓 Aula 1 — Introdução à Visão Computacional 3D

### Conceitos Fundamentais

A **Visão Computacional 3D** é uma área da inteligência artificial que se dedica à compreensão e interpretação do mundo tridimensional a partir de dados capturados por sensores ou câmeras. Diferente da visão computacional 2D tradicional, que trabalha com imagens planas, a VC3D permite que máquinas percebam profundidade, volume e estrutura espacial dos objetos.

### Principais Representações de Dados 3D

- **Imagens de Profundidade (Depth Maps):** cada pixel armazena a distância do objeto em relação ao sensor.
- **Nuvem de Pontos (Point Cloud):** conjunto de pontos no espaço tridimensional (x, y, z), podendo conter atributos como cor e intensidade.
- **Malhas Poligonais (Mesh):** superfícies compostas por vértices, arestas e faces (triângulos ou polígonos).
- **Voxels:** representação volumétrica discreta em uma grade 3D, análoga a pixels em 3D.
- **Mapas Normais:** armazenam a direção das normais da superfície para reconstrução de detalhes finos.

### Aplicações da Visão Computacional 3D

- **Veículos Autônomos:** percepção do ambiente ao redor, detecção de obstáculos, pedestres e outros veículos.
- **Robótica:** navegação, manipulação de objetos e interação com o ambiente.
- **Realidade Aumentada e Virtual:** sobreposição de objetos virtuais no espaço físico.
- **Medicina:** reconstrução 3D de estruturas anatômicas a partir de tomografias e ressonâncias.
- **Manufatura e Controle de Qualidade:** inspeção de peças, modelagem de superfícies e detecção de defeitos.
- **Reconstrução de Cenas:** criação de modelos 3D de ambientes internos e externos.

### Desafios da VC3D

- **Volume de dados:** nuvens de pontos e dados volumétricos exigem muito armazenamento e processamento.
- **Irregularidade:** diferentemente de imagens 2D (grades regulares), os dados 3D são frequentemente não estruturados.
- **Invariância:** os algoritmos precisam ser robustos a rotações, translações e variações de escala.
- **Ruído e oclusão:** sensores introduzem ruído e partes dos objetos podem estar ocluídas.

---

## 📡 Aula 2 — Sensores para Visão Computacional 3D

### Tipos de Sensores 3D

#### 1. Câmeras Estereoscópicas (Stereo Vision)
Utilizam duas câmeras dispostas em paralelo, simulando a visão binocular humana. A profundidade é estimada a partir da disparidade entre os dois quadros capturados.

- **Vantagem:** alto nível de detalhe de cor e textura.
- **Desvantagem:** requer calibração precisa e é sensível a ambientes sem textura.
- **Exemplo de hardware:** Intel RealSense D435, ZED Camera.

#### 2. Câmeras de Tempo de Voo (Time-of-Flight — ToF)
Emitem pulsos de luz infravermelha e medem o tempo que a luz leva para retornar após atingir um objeto. A distância é calculada diretamente pelo tempo de voo.

- **Vantagem:** operação em tempo real, independente de textura superficial.
- **Desvantagem:** resolução limitada e sensível a superfícies reflexivas ou translúcidas.
- **Exemplo de hardware:** Microsoft Azure Kinect, câmeras ToF da Texas Instruments.

#### 3. LiDAR (Light Detection and Ranging)
Emite feixes laser e mede o tempo de retorno para construir nuvens de pontos de alta precisão. Muito utilizado em veículos autônomos e mapeamento geoespacial.

- **Vantagem:** alta precisão, funciona em longa distância.
- **Desvantagem:** custo elevado, limitações em condições de chuva/neblina.
- **Exemplo de hardware:** Velodyne HDL-64E, SICK LiDAR, Ouster OS1.

#### 4. Câmeras de Luz Estruturada (Structured Light)
Projetam um padrão de luz conhecido (grade, franjas) sobre a cena e analisam como o padrão é deformado pela geometria dos objetos para calcular a profundidade.

- **Vantagem:** alta resolução, ideal para objetos próximos.
- **Desvantagem:** não funciona bem em ambientes com muita luz ambiente ou ao ar livre.
- **Exemplo de hardware:** Microsoft Kinect v1, Intel RealSense L515.

#### 5. Câmeras RGB-D
Combinam uma câmera de cor (RGB) com um sensor de profundidade (D), fornecendo imagens coloridas associadas a um mapa de profundidade.

- **Exemplo de hardware:** Microsoft Kinect, Intel RealSense.
- **Formatos de dados:** imagem RGB + imagem de profundidade alinhada pixel a pixel.

### Comparativo de Sensores

| Sensor | Precisão | Alcance | Custo | Ambiente |
|--------|----------|---------|-------|----------|
| Estereoscópico | Média | Médio | Baixo | Interno/Externo |
| ToF | Média | Curto/Médio | Médio | Interno |
| LiDAR | Alta | Longo | Alto | Interno/Externo |
| Luz Estruturada | Alta | Curto | Médio | Interno |
| RGB-D | Média | Curto/Médio | Baixo/Médio | Interno |

### Calibração de Sensores

A calibração é essencial para garantir a precisão das medidas. Envolve:
- **Calibração intrínseca:** parâmetros internos da câmera (distância focal, ponto principal, distorções).
- **Calibração extrínseca:** posição e orientação relativa entre sensores ou câmeras.

---

## 🖼️ Aula 3 — Geração e Manipulação de Imagens 3D

### 📄 Notebook Prático

**Arquivo:** [`depth_estimation.ipynb`](./Geração%20e%20manipulação%20de%20imagens%203D/depth_estimation.ipynb)

**Script original (Keras):** [Monocular Depth Estimation — Keras Examples](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/depth_estimation.ipynb)

> ⚠️ **Obs:** partes do código original foram adaptadas devido a mudanças na API do Keras ao longo do tempo (ex: `keras.utils.Sequence` → `keras.utils.PyDataset`).

---

### Estimativa de Profundidade Monocular

A **estimativa de profundidade monocular** (_monocular depth estimation_) é o processo de inferir a profundidade de uma cena a partir de uma **única imagem RGB**, sem o uso de sensores adicionais.

É uma tarefa fundamental para inferir a geometria de cenas 3D a partir de dados 2D.

#### Abordagem Utilizada
O notebook implementa um modelo de estimativa de profundidade com redes convolucionais (ConvNet) e funções de perda simples, utilizando:

- **Dataset:** [DIODE — Dense Indoor and Outdoor Depth Dataset](http://diode-dataset.s3.amazonaws.com/val.tar.gz)
  - Utiliza o conjunto de validação (2.6 GB) contendo imagens indoor e outdoor com mapas de profundidade densos.
  - Alternativas: [NYU-v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) e [KITTI](http://www.cvlibs.net/datasets/kitti/).

- **Framework:** Keras (backend TensorFlow)

#### Pipeline do Projeto

```
Imagens RGB (indoor DIODE)
        │
        ▼
┌─────────────────────┐
│   DataGenerator     │  ← Leitura, redimensionamento,
│  (keras.utils.      │     normalização de profundidade
│   PyDataset)        │     (log-scale, clipping, masking)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Modelo ConvNet    │  ← Encoder-Decoder com camadas
│   (Encoder-Decoder) │     convolucionais
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│   Mapa de           │  ← Imagem de profundidade predita
│   Profundidade      │     pixel a pixel
└─────────────────────┘
```

#### Hiperparâmetros

| Parâmetro | Valor |
|-----------|-------|
| HEIGHT | 256 |
| WIDTH | 256 |
| Learning Rate (LR) | 0.00001 |
| Épocas (EPOCHS) | 30 |
| Batch Size | 32 |

#### Pré-processamento dos Dados

O pipeline de dados realiza os seguintes passos:
1. Leitura e redimensionamento das imagens RGB para 256×256.
2. Leitura dos arquivos `.npy` de profundidade e máscara.
3. Clipping da profundidade no percentil 99 (máx. 300m).
4. Aplicação de escala logarítmica nos valores de profundidade válidos.
5. Mascaramento de regiões inválidas (sem medição).
6. Normalização para `float32`.

#### Bibliotecas Utilizadas

```python
import os
import tensorflow as tf
import keras
from keras import layers, ops
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

---

## 🔵 Aula 4 — Classificação de Imagens em Nuvem de Pontos

### 📄 Notebook Prático

**Arquivo:** [`pointnet.ipynb`](./Classificação%20de%20imagens%20em%20nuvem%20de%20pontos/pointnet.ipynb)

**Script original (Keras):** [Point Cloud Classification with PointNet — Keras Examples](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/pointnet.ipynb)

> ⚠️ **Obs:** partes do código original foram adaptadas (ex: ajuste do download e extração do dataset ModelNet10, correção de caminhos com `glob` para localizar arquivos `.off`).

---

### O que é uma Nuvem de Pontos?

Uma **nuvem de pontos** (_point cloud_) é um conjunto de pontos no espaço 3D, cada um representado por coordenadas (x, y, z). Esses dados podem ser capturados por sensores LiDAR, câmeras RGB-D ou reconstruídos a partir de múltiplas imagens 2D.

Tarefas clássicas com nuvem de pontos incluem:
- **Classificação:** identificar a categoria do objeto representado.
- **Detecção de objetos:** localizar e identificar múltiplos objetos na cena.
- **Segmentação semântica:** rotular cada ponto com uma categoria.

### PointNet

O **PointNet** ([Qi et al., 2017](https://arxiv.org/abs/1612.00593)) é o primeiro trabalho seminal que aplica deep learning diretamente em nuvens de pontos não ordenadas, sem convertê-las para representações intermediárias (voxels ou imagens 2D).

#### Principais Características do PointNet

- **Entrada não ordenada:** opera diretamente no conjunto desordenado de pontos (x, y, z).
- **Invariância a transformações:** utiliza uma rede de transformação (T-Net) para aprender transformações de alinhamento no espaço de entrada e de features.
- **Agregação global:** usa `max pooling` para extrair uma representação global invariante à permutação dos pontos.

#### Arquitetura do PointNet

```
Nuvem de Pontos [N × 3]
        │
        ▼
┌─────────────────────┐
│  Input Transform    │  ← T-Net 3×3 (alinha a nuvem
│  (T-Net)            │     no espaço de entrada)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  MLP Compartilhado  │  ← [64, 64] — features por ponto
│  (ponto a ponto)    │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Feature Transform  │  ← T-Net 64×64 (alinha no
│  (T-Net)            │     espaço de features)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  MLP Compartilhado  │  ← [64, 128, 1024] features/ponto
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Global Max Pooling │  ← Agregação global [1024]
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  MLP de Classificação│ ← [512, 256, K classes]
└─────────────────────┘
        │
        ▼
  Predição de Classe
```

#### Dataset Utilizado

- **ModelNet10:** versão reduzida do ModelNet40 com **10 categorias** de objetos 3D (mesa, cadeira, banheiro, cama, monitor, mesa de escritório, sofá, cômoda, noite, banheira).
- **Formato dos arquivos:** `.off` (Object File Format) — malhas poligonais 3D.
- **Tamanho:** ~4.899 arquivos de malha.
- **Download:** via `keras.utils.get_file` a partir do Princeton 3D Shape Repository.

#### Pré-processamento

Os dados passam pelos seguintes processos:
1. **Leitura das malhas** com a biblioteca `trimesh`.
2. **Amostragem de pontos** (_point sampling_): amostram-se N pontos aleatórios da superfície da malha para gerar a nuvem de pontos.
3. **Normalização:** os pontos são centralizados e normalizados para o intervalo [-1, 1].
4. **Aumento de dados (Data Augmentation):** jitter (ruído gaussiano), shuffle aleatório dos pontos.

#### Bibliotecas Utilizadas

```python
import os
import glob
import trimesh
import numpy as np
from tensorflow import data as tf_data
from keras import ops
import keras
from keras import layers
from matplotlib import pyplot as plt
```

#### Resultados Esperados

O modelo PointNet implementado atinge uma **acurácia de classificação superior a 85%** no conjunto de teste do ModelNet10 após o treinamento completo.

---

## 🗂️ Estrutura de Arquivos

```
Visão Computacional 3D e seus algoritmos/
│
├── README.md
│
├── Geração e manipulação de imagens 3D/
│   └── depth_estimation.ipynb         ← Estimativa de profundidade monocular
│
└── Classificação de imagens em nuvem de pontos/
    └── pointnet.ipynb                 ← Classificação 3D com PointNet
```

---

## 🔗 Referências e Recursos

### Papers
- **PointNet:** Qi, C. R., Su, H., Mo, K., & Guibas, L. J. (2017). [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593). CVPR 2017.
- **DIODE Dataset:** Vasiljevic, I., et al. (2019). [DIODE: A Dense Indoor and Outdoor Depth Dataset](https://arxiv.org/abs/1908.00463).

### Scripts Originais (Keras Examples)
- [Monocular Depth Estimation](https://keras.io/examples/vision/depth_estimation/) — Victor Basu (2021)
- [Point Cloud Classification with PointNet](https://keras.io/examples/vision/pointnet/) — David Griffiths (2020)

### Datasets
- [DIODE Dataset](http://diode-dataset.s3.amazonaws.com/val.tar.gz) — Dense Indoor and Outdoor Depth
- [ModelNet10](http://3dvision.princeton.edu/projects/2014/3DShapeNets/) — Princeton 3D Shape Repository
- [NYU-v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) — NYU Depth Dataset V2
- [KITTI](http://www.cvlibs.net/datasets/kitti/) — KITTI Vision Benchmark Suite

### Blog Posts e Tutoriais
- [An In-Depth Look at PointNet](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a) — Luis Gonzales
