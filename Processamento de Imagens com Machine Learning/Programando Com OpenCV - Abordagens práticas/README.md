# Programando com OpenCV: Abordagens Práticas

Este repositório contém o conteúdo teórico e prático desenvolvido ao longo do módulo de processamento de imagens com **OpenCV** (Open Source Computer Vision Library). O conteúdo está organizado por aulas, cobrindo desde conceitos fundamentais e instalação até aplicações de conversão de cores e segmentação por máscara de cor.

---

## 📅 Cronograma de Aulas

### 🎥 Aula 1 - Programando Com OpenCV
#### O que é OpenCV?
O **OpenCV** (Open Source Computer Vision Library) é uma biblioteca de código aberto amplamente utilizada no mercado e na academia para visão computacional, processamento de imagens e aprendizado de máquina. 

**Principais características:**
- **Linguagem Base:** Desenvolvida em C/C++, garantindo alta performance para processamento em tempo real.
- **Portabilidade:** Possui suporte oficial para Python, Java, C++, Android, iOS, Windows, Linux e macOS.
- **Aplicações Comuns:**
  - Filtragem e pré-processamento de imagens.
  - Detecção e reconhecimento de objetos e faces.
  - Rastreamento de movimento em vídeos.
  - Calibração de câmera e reconstrução 3D.
  - Integração com redes neurais profundas (Deep Learning) por meio do módulo `dnn`.

---

### 🎥 Aula 2 - Instalando OpenCV com Anaconda
#### Instalação do OpenCV
Para configurar o ambiente de desenvolvimento de forma isolada e evitar conflitos de dependências, recomenda-se a utilização do **Anaconda** ou **Miniconda**.

**Passos para instalação:**

1. **Criar um ambiente virtual (opcional, mas recomendado):**
   ```bash
   conda create -n opencv-env python=3.10
   conda activate opencv-env
   ```

2. **Instalação via Conda (Canal conda-forge):**
   ```bash
   conda install -c conda-forge opencv
   ```

3. **Instalação via pip (Alternativa):**
   Se preferir usar o `pip` padrão, instale a biblioteca principal e as extensões:
   ```bash
   pip install opencv-python opencv-contrib-python numpy matplotlib
   ```

*(Nota: O pacote `opencv-python` contém a biblioteca OpenCV principal, enquanto `opencv-contrib-python` inclui módulos extras úteis. O `numpy` e o `matplotlib` são essenciais para manipulação matricial e visualização de dados).*

---

### 🎥 Aula 3 - Exemplo de conversão de imagens com OpenCV
#### Script Prático: [imagens_rbg_cinza.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/Programando%20Com%20OpenCV%20-%20Abordagens%20pr%C3%A1ticas/Exemplo%20de%20convers%C3%A3o%20de%20imagens%20com%20OpenCV/imagens_rbg_cinza.py)

Nesta aula prática, aprendemos a realizar a conversão clássica de imagens do espaço de cores colorido para escala de cinza.

**Pontos-chave abordados no código:**
- **Leitura padrão do OpenCV:** O método `cv2.imread()` lê imagens no formato **BGR** (Blue, Green, Red).
- **Conversão BGR para RGB:** Para exibir corretamente a imagem usando bibliotecas externas como `matplotlib` (que esperam o padrão RGB), é necessária a conversão dos canais:
  ```python
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  ```
- **Conversão para Escala de Cinza:** Utilização da constante `cv2.COLOR_BGR2GRAY` para calcular a média ponderada dos canais e obter a intensidade monocromática:
  ```python
  img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
  ```
- **Salvar em disco:** Uso do `cv2.imwrite()` para exportar o arquivo processado final.

---

### 🎥 Aula 4 - Segmentação de imagens por cores
#### Script Prático: [segmentacao_cores_hsv.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/Programando%20Com%20OpenCV%20-%20Abordagens%20pr%C3%A1ticas/Segmenta%C3%A7%C3%A3o%20de%20imagens%20por%20cores/segmentacao_cores_hsv.py)

Esta aula aborda a técnica de segmentação por cor utilizando o espaço de cores **HSV** (Hue, Saturation, Value / Matiz, Saturação, Valor), que separa a informação cromática da luminosidade.

**Passo a passo lógico implementado:**
1. **Conversão para HSV:** A imagem é convertida para HSV pois o espaço RGB/BGR é muito sensível a variações de iluminação, dificultando a seleção de uma cor específica.
   ```python
   hsv_im = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)
   ```
2. **Definição de Limiares:** Criação de faixas (limites inferior e superior) de Hue para focar em uma matiz específica (no caso, a cor verde da grama).
3. **Criação de Máscara:** Uso do método `cv2.inRange()` para retornar uma máscara binária (onde pixels dentro da faixa são brancos `255` e fora dela são pretos `0`).
   ```python
   mask = cv2.inRange(hsv_im, lower_th, upper_th)
   ```
4. **Operação Bitwise:** Aplicação da máscara binária sobre a imagem original usando a operação lógica AND com `cv2.bitwise_and()` para preservar apenas a área de interesse (a grama) e remover o céu.
   ```python
   rgb_res = cv2.bitwise_and(rgb_im, rgb_im, mask=mask)
   ```
