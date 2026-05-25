# Introdução ao Processamento de Imagens Digitais

Este diretório contém os conceitos teóricos e práticos abordados no módulo de **Introdução ao Processamento de Imagens Digitais** como parte da Formação de Specialist em Machine Learning.

---

## 📸 Conceitos Fundamentais

O Processamento de Imagens Digitais (PID) envolve a manipulação de uma imagem por meio de algoritmos de computador com o objetivo de melhorar sua qualidade para interpretação humana ou prepará-la para tarefas de visão computacional.

### 📐 Representação de Imagens e Armazenamento em Matrizes
Toda imagem digital é representada computacionalmente como uma **matriz** (ou tensor multi-dimensional):
- **Imagens em Escala de Cinza:** Matrizes bidimensionais $M \times N$, onde cada elemento (pixel) possui um valor de intensidade luminosa (geralmente entre `0` - preto e `255` - branco).
- **Imagens Coloridas (RGB):** Tensores tridimensionais $M \times N \times 3$, representando os três canais de cores: vermelho (*Red*), verde (*Green*) e azul (*Blue*).

### 🛠️ Etapas e Técnicas de Processamento

1. **Estado das Imagens:**
   - **Com Ruído / Distorcida:** Imagens que sofreram degradação durante a captura ou transmissão (ex: ruído sal e pimenta, desfoque de movimento).
   - **Processada:** O resultado da aplicação de filtros e transformações para restaurar ou realçar informações úteis.
   
2. **Contraste:**
   - Ajuste na distribuição dos níveis de cinza de uma imagem para melhorar a distinção entre os objetos e o fundo (ex: equalização de histograma).

3. **Conversão de Cores:**
   - **Níveis de Cinza (Grayscale):** Conversão de RGB para um único canal de intensidade usando ponderações dos canais (ex: $Y = 0.299R + 0.587G + 0.114B$).
   - **Preto e Branco (Binarização / Limiarização):** Conversão da imagem para valores binários (`0` ou `1` / `0` ou `255`) com base em um valor de limiar (*threshold*).

4. **Filtragem de Ruídos:**
   - Aplicação de filtros espaciais (como filtros de média, mediana ou gaussianos) para suavizar a imagem e atenuar imperfeições indesejadas.

5. **Detecção e Segmentação de Bordas:**
   - Identificação de pontos em uma imagem digital onde a luminosidade muda bruscamente (usando operadores como Sobel, Canny ou Laplacian).
   - **Segmentação das Bordas:** Isolamento das fronteiras físicas dos objetos para facilitar a análise de formas.

6. **Detecção de Features (Características):**
   - Extração de pontos de interesse, cantos ou formas geométricas específicas (ex: cantos de Harris, SIFT, SURF) que descrevem unicamente partes da imagem.

7. **Redução de Dimensionalidade e Redução nas Dimensões:**
   - **Redução nas Dimensões (Redimensionamento):** Alteração física do tamanho da imagem (resolução) via interpolação.
   - **Redução de Dimensionalidade:** Técnicas como PCA (*Principal Component Analysis*) ou Pooling (em redes neurais) para reduzir o número de variáveis representativas mantendo a informação essencial.

---

## 👁️ Visão Computacional

> [!NOTE]
> **O que é Visão Computacional?**
> É a área da ciência da computação que busca capacitar os computadores a "enxergarem" e interpretarem o mundo visual de forma semelhante aos humanos.

O fluxo de trabalho típico da visão computacional compreende:
1. **Sensoriamento de Imagens:** Captura física da luz por sensores fotográficos (câmeras, LiDAR, etc.) convertendo-a em dados digitais.
2. **Processamento de Imagens:** Tratamento prévio da imagem para melhorar a qualidade dos dados (remoção de ruídos, ajustes de contraste, redimensionamento).
3. **Análise (Machine Learning / Deep Learning):** Extração de padrões e tomada de decisão fundamentada nos dados visuais tratados.

---

## 🚀 Aplicações Práticas

* **Detecção e Reconhecimento Facial:**
  * Uso de arquiteturas de **Deep Learning** (como Redes Neurais Convolucionais - CNNs) para identificar a presença de rostos (*Detecção*) e verificar a identidade da pessoa (*Reconhecimento*).
* **Detecção de Objetos nas Imagens:**
  * Identificação e localização de múltiplos objetos em uma cena, desenhando caixas delimitadoras (*bounding boxes*) e atribuindo classes (ex: YOLO, SSD, Faster R-CNN).
* **Reconhecimento de Pessoas:**
  * Monitoramento, contagem e rastreamento de indivíduos em sistemas de segurança, transporte público e comércio.
