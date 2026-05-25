# Processamento de Imagens: Filtros, Contraste, Transformação e Morfologia

Este diretório contém o script Python adaptado para execução local que aplica técnicas de detecção de bordas, suavização, realce de contraste, transformações geométricas e operações morfológicas em imagens digitais.

---

## 👤 Autor Original e Créditos

O código original foi desenvolvido como um Jupyter Notebook para a plataforma Google Colab, baseado na aula *"Image Processing in Python"* do Datacamp.

* **Autor Original:** [Chanseok Kang](https://github.com/goodboychan)
* **Notebook Original:** [2020-08-02-01-Filters-Contrast-Transformation-and-Morphology.ipynb](https://colab.research.google.com/github/goodboychan/chans_jupyter/blob/main/_notebooks/2020-08-02-01-Filters-Contrast-Transformation-and-Morphology.ipynb)
* **Repositório do Dataset:** [goodboychan/chans_jupyter](https://github.com/goodboychan/chans_jupyter)

---

## 🛠️ Alterações Realizadas para Execução Local

Para possibilitar que o script rode de forma autônoma e salve os resultados sem interrupções em sua máquina, foram feitas as seguintes melhorias:

### 1. Estrutura de Pastas Automatizada
O script agora cria automaticamente duas pastas no mesmo nível do arquivo `filters_contrast_transformation_and_morphology.py`:
* **`downloads/`**: Destinada ao armazenamento das imagens de entrada.
* **`resultados/`**: Destinada a salvar as saídas gráficas em formato `.png`.

### 2. Download Automatizado do Dataset
Foi inserida uma rotina em Python (`get_image_path`) que verifica se cada uma das imagens necessárias existe localmente na pasta `downloads/`. Caso estejam ausentes, o script baixa o arquivo correspondente diretamente do GitHub do autor original com suporte a fallback de URL automática.
* **Imagens baixadas:**
  * `soap_image.jpg`
  * `building_image.jpg`
  * `chest_xray_image.png`
  * `image_aerial.png`
  * `image_cat.jpg`
  * `dogs_banner.jpg`
  * `r5.png`
  * `world_image_binary.jpg`
  * *(Imagens extras `coffee` e `rocket` são carregadas diretamente do pacote `skimage.data`)*

### 3. Atualização de Compatibilidade do `scikit-image`
As versões modernas do pacote `scikit-image` (a partir da versão `0.20` e incluindo a `0.26` instalada no sistema) removeram alguns parâmetros legados e descontinuaram funções. Realizamos as seguintes correções de compatibilidade:
* **Substituição de `multichannel=True`**: Ajustado usando tratamento de exceção (`try-except`) para mapear para `channel_axis=-1` nas funções de redimensionamento e filtros. Isso evita falhas de `TypeError`.
* **Substituição de Funções Obsoletas de Morfologia**: As funções `binary_erosion` e `binary_dilation` geravam avisos (`FutureWarning`) sobre futura remoção. Atualizamos as chamadas para `skimage.morphology.erosion` e `skimage.morphology.dilation`, tratando também os arrays de entrada como booleanos puros.

### 4. Salvamento de Comparações Visuais
No Colab, os gráficos do matplotlib são mostrados na tela por padrão. Para a execução local, o script foi ajustado para gerar figuras lado a lado (Original vs Processada) e salvá-las silenciosamente na pasta `resultados/` sem travar a execução do terminal.

Ao todo, são salvos **10 arquivos de comparação**:
1. `1_edge_detection_sobel.png` (Detecção de bordas na imagem Soap)
2. `2_gaussian_blurring.png` (Redução de ruído com filtro Gaussiano no prédio)
3. `3_chest_xray_equalized.png` (Contraste e histograma do Raio-X de tórax)
4. `4_aerial_image_equalized.png` (Contraste da imagem aérea da cidade)
5. `5_coffee_adaptive_equalized.png` (Equalização adaptativa CLAHE do copo de café)
6. `6_cat_rescaled_comparison.png` (Rotação e redimensionamento com/sem anti-aliasing do gato)
7. `7_rocket_enlarged.png` (Ampliação 3x do foguete)
8. `8_dogs_banner_resized.png` (Redimensionamento proporcional do banner de cães)
9. `9_r5_eroded.png` (Erosão morfológica da letra R manuscrita)
10. `10_world_dilated.png` (Dilatação morfológica no mapa-múndi binário)

---

## 📦 Requisitos do Sistema

Para executar este script localmente, certifique-se de ter as seguintes bibliotecas instaladas em seu ambiente Python:

```bash
pip install numpy matplotlib scikit-image
```

---

## 🚀 Como Rodar o Script

1. Abra o terminal (PowerShell ou Prompt de Comando) no diretório do arquivo.
2. Execute o comando abaixo:

```bash
python filters_contrast_transformation_and_morphology.py
```

O script fará o download das imagens que faltam, aplicará todos os filtros e salvará todas as comparações automaticamente na pasta de resultados.
