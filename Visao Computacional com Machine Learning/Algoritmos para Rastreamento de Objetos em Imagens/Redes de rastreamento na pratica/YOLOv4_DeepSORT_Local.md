# YOLOv4 DeepSORT: Guia de Migração (Google Colab para Local Windows)

Este documento descreve detalhadamente as modificações de arquitetura, dependências e configurações realizadas para migrar o script de rastreamento de múltiplos objetos **YOLOv4-DeepSORT** do ambiente interativo do **Google Colab** ([YOLOv4-DeepSORT.ipynb](https://colab.research.google.com/github/hardik0/Multi-Object-Tracking-Google-Colab/blob/main/YOLOv4-DeepSORT.ipynb)) para uma execução local e nativa em **Windows** com **Python 3.12** e **TensorFlow 2.16+ / Keras 3**.

---

## 📊 Tabela de Comparação: Colab vs. Local Windows

| Recurso / Configuração | Notebook Google Colab (Original) | Ambiente Local Windows (Modificado) |
| :--- | :--- | :--- |
| **Interface de Execução** | Células interativas Jupyter / Comandos mágicos (`!`, `%`) | Script Python nativo (`yolov4_deepsort.py`) |
| **Acesso a Hardware** | GPU Dedicada do Google Colab (CUDA 10.1 pré-configurado) | CPU Local (Nativo Windows s/ dependência obrigatória de CUDA) |
| **Caminhos de Pastas** | Comandos `%cd` e caminhos absolutos do Linux `/content/` | Caminhos relativos padrão Windows (`./checkpoints/yolov4-416`) |
| **Compatibilidade de Nomes** | Pastas com acentos e espaços permitidos no Linux | Renomeados para ASCII (Ex: `Visão` ➡️ `Visao`) devido a bugs do TF no Windows |
| **Versão do TensorFlow** | TensorFlow 2.3.0 e Keras 2.4.3 | TensorFlow 2.16+ / 2.21+ e Keras 3 |
| **Versão do NumPy** | NumPy 1.18.x | NumPy 2.x (Removidos aliases depreciados como `np.int`) |
| **Salvamento do Modelo** | `model.save(filepath)` (Legado) | `model.export(filepath)` (Padrão Keras 3 para SavedModel) |
| **Estabilidade do Rastreador** | Sem tratamento de caixas (BBox) inválidas | Filtro para descartar caixas com `width <= 0` ou `height <= 0` (Evita NaN) |
| **Visualização de Vídeo** | Codificação Base64 com HTML para embutir na célula | Gravação direta em arquivo de saída (`outputs/result.avi`) |

---

## 🛠️ Detalhes das Modificações Realizadas

### 1. Tratamento de Acentos e Caminhos no Windows (TensorFlow VFS Bug)
> [!WARNING]
> A biblioteca nativa C++ do TensorFlow apresenta falhas de codificação sob o sistema de arquivos virtual (VFS) do Windows quando os caminhos de diretório contêm caracteres acentuados (como `Visão Computacional` ou `prática`). Isso causava falhas silenciosas ou exceções de IO ao tentar ler ou salvar os pesos do modelo (`saved_model.pb`).

* **Solução:** Renomeamos os diretórios de:
  - `Visão Computacional com Machine Learning` ➡️ `Visao Computacional com Machine Learning`
  - `Redes de rastreamento na prática` ➡️ `Redes de rastreamento na pratica`
* **Caminho dos Checkpoints:** O modelo agora é exportado e lido de forma limpa usando o caminho relativo do projeto: `./checkpoints/yolov4-416`.

---

### 2. Atualização de Código para TensorFlow 2.16+ e Keras 3
Como o código original foi desenvolvido para o ecossistema do TensorFlow 2.3, ele utilizava recursos e estruturas do Keras 2 que não são mais compatíveis com o Keras 3 (onde os tensores de saída de modelos funcionais são objetos simbólicos `KerasTensor`).

* **Tensores Simbólicos no Keras 3:** Operações diretas do TensorFlow (como `tf.concat` ou `tf.nn.max_pool`) aplicadas a tensores do Keras causavam falhas no fluxo de compilação. Atualizamos as funções do YOLOv4 no arquivo [core/common.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/core/common.py) e [core/yolov4.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/core/yolov4.py) para usar equivalentes universais do Keras:
  - `tf.concat` ➡️ `keras.ops.concatenate`
  - `tf.image.resize` ➡️ `keras.ops.image.resize`
* **Uso de Camadas Lambda:** No utilitário de conversão de modelo ([save_model.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/save_model.py)), as funções de pós-processamento de caixas delimitadoras (`decode` e `filter_boxes`) foram encapsuladas em camadas `keras.layers.Lambda` para que o grafo do Keras possa rastreá-las.
* **Leitura de Canais de Entrada:** No Keras 3, a propriedade `input_shape` de camadas convolucionais não instanciadas/compiladas não pode ser acessada diretamente. No arquivo [core/utils.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/core/utils.py), alteramos o acesso de `conv_layer.input_shape` para `conv_layer.input.shape[-1]`.
* **Exportação do Modelo:** Substituímos o método depreciado `model.save()` pelo padrão recomendado no Keras 3:
  ```python
  # save_model.py
  model.export(args.output)
  ```

---

### 3. Compatibilidade com NumPy 2.x
* **Erro Comum:** `AttributeError: module 'numpy' has no attribute 'int'` ou `has no attribute 'float'`.
* **Solução:** O NumPy 2.x removeu as referências depreciadas aos tipos básicos do Python. Fizemos uma varredura nos scripts internos do algoritmo de rastreamento (especialmente em `deep_sort/detection.py`, `deep_sort/preprocessing.py` e `tools/generate_detections.py`) para substituir todos os usos de `np.int` e `np.float` por `int` e `float` nativos do Python.

---

### 4. Filtro contra Caixas Delimitadoras Inválidas (Estabilidade do DeepSORT)
> [!IMPORTANT]
> Em frames complexos ou nas bordas da imagem, detectores de Deep Learning podem gerar caixas delimitadoras com dimensões inválidas (largura ou altura iguais a zero). 

* **Problema:** Essas caixas eram passadas ao DeepSORT e causavam divisão por zero no cálculo da razão de aspecto (`to_xyah()`), produzindo valores `NaN`/`Inf`. Consequentemente, o solver de matrizes de custo (`scipy.optimize.linear_sum_assignment`) falhava, parando a execução do rastreador.
* **Solução:** Adicionamos uma validação no arquivo [object_tracker.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/object_tracker.py) para filtrar as detecções antes do rastreamento:
  ```python
  # Filtra caixas com dimensões inválidas ou nulas
  if w <= 0 or h <= 0:
      continue
  ```

---

### 5. Conversão de Comandos Mágicos de Shell para Código Python
No notebook original, diversas ações eram executadas via terminal interativo (ex: `!python save_model.py --model yolov4`).

* **Abstração por subprocessos:** No script unificado [yolov4_deepsort.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/yolov4_deepsort.py), estas chamadas foram reescritas usando o módulo `subprocess` do Python, buscando dinamicamente o executável do ambiente virtual ativo:
  ```python
  import sys
  import subprocess

  # Execução local usando o python da .venv
  subprocess.run([sys.executable, "save_model.py", "--model", "yolov4"])
  ```
* **Remoção de downloads e instalações interativas:** Comandos de terminal como `!pip install -r requirements-gpu.txt` e `!git clone` foram omitidos do script para evitar re-instalações indesejadas durante a execução.

---

### 6. Tratamento de Vídeos e Dependências do FFmpeg
No Google Colab, o vídeo `.avi` resultante é obrigatoriamente convertido via `ffmpeg` para um arquivo `.mp4` para poder ser incorporado na página web via codificação em base64.

* **Flexibilidade Local:** Em execução local, o vídeo de saída original (`.avi`) pode ser reproduzido diretamente no sistema operacional. Para evitar erros na ausência do utilitário `ffmpeg` no Windows, o script [yolov4_deepsort.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Visao%20Computacional%20com%20Machine%20Learning/Algoritmos%20para%20Rastreamento%20de%20Objetos%20em%20Imagens/Redes%20de%20rastreamento%20na%20pratica/yolov4_deepsort.py) agora valida a presença do executável `ffmpeg` e, caso não o encontre, emite um aviso amigável sem interromper a conclusão da execução.

---

## 🚀 Como Executar o Script Localmente

Siga os passos abaixo no terminal PowerShell dentro do seu ambiente virtual:

1. **Baixe e insira os pesos oficiais** `yolov4.weights` na pasta `./data/`.
2. **Execute o script unificado** (ele realizará a conversão e rodará o rastreamento automaticamente):
   ```powershell
   python yolov4_deepsort.py
   ```
3. O vídeo final com as marcações de identificadores (IDs) de rastreamento estará salvo em:
   `./outputs/result.avi` (ou no formato correspondente).
