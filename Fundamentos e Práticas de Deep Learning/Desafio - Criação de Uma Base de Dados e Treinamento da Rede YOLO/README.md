# Desafio: Criação de Uma Base de Dados e Treinamento da Rede YOLO (COCO-2017)

Este diretório contém a solução do desafio de projeto voltado para a criação de uma base de dados personalizada e o treinamento de uma rede YOLOv8 utilizando o dataset **COCO-2017** através da biblioteca `FiftyOne`.

O pipeline foi projetado para automatizar todo o processo de obtenção de dados, conversão para o formato YOLO, treinamento, detecção e avaliação das métricas de desempenho.

As classes de interesse utilizadas são:
* **car** (Carro)
* **person** (Pessoa)
* **bicycle** (Bicicleta)

---

## 📂 Estrutura do Projeto

A estrutura de arquivos e pastas organizada no diretório é a seguinte:

* [yolo_coco.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos%20e%20Pr%C3%A1ticas%20de%20Deep%20Learning/Desafio%20-%20Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/yolo_coco.py): Script principal contendo as funções do pipeline (`download_and_convert`, `train`, `infer`, `validate`) e suporte a CLI.
* [run_pipeline_coco.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos%20e%20Pr%C3%A1ticas%20de%20Deep%20Learning/Desafio%20-%20Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/run_pipeline_coco.py): Script auxiliar para executar o pipeline completo (do download à validação) de ponta a ponta em lote com parâmetros configurados.
* **`datasets/coco_yolo/`**: Diretório que contém a base de dados baixada e convertida:
  * `images/validation/`: Imagens do COCO-2017 baixadas automaticamente.
  * `labels/validation/`: Anotações das classes selecionadas convertidas para o formato YOLO (`<class_id> <cx> <cy> <w> <h>`).
  * `dataset.yaml`: Arquivo de configuração contendo os caminhos absolutos/relativos e nomes das classes para o treinamento no YOLOv8.
* **`resultados_coco/`**: Resultados gerados a partir da última execução:
  * `runs/train/yolov8n_coco/`: Histórico do treinamento (pesos `best.pt` e `last.pt`, matriz de confusão, curvas de precisão-recall, gráficos de perda).
  * `runs/detect/predict/`: Imagens contendo as predições anotadas com bounding boxes (caixas delimitadoras) e as pontuações de confiança.
  * `runs/val/validation/`: Resultados e logs da validação do modelo com o conjunto de teste.
  * [resultado_treino.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos%20e%20Pr%C3%A1ticas%20de%20Deep%20Learning/Desafio%20-%20Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/resultados_coco/resultado_treino.txt): Resumo de texto das configurações utilizadas no treinamento.
  * [resultado_inferencia.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos%20e%20Pr%C3%A1ticas%20de%20Deep%20Learning/Desafio%20-%20Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/resultados_coco/resultado_inferencia.txt): Relatório das detecções individuais de cada imagem processada.
  * [resultado_validacao.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos%20e%20Pr%C3%A1ticas%20de%20Deep%20Learning/Desafio%20-%20Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/resultados_coco/resultado_validacao.txt): Relatório com as métricas detalhadas obtidas.

---

## 🛠️ Requisitos e Instalação

Certifique-se de que possui as dependências necessárias instaladas em seu ambiente virtual (`.venv`):

```bash
.venv/Scripts/pip install ultralytics fiftyone opencv-python pyyaml torch
```

*(Nota: O FiftyOne não necessita de credenciais adicionais para baixar imagens do COCO-2017, facilitando o download direto em relação ao Open Images).*

---

## 🚀 Como Executar

### Execução Completa (End-to-End)
Você pode rodar todo o pipeline com apenas um comando utilizando o script `run_pipeline_coco.py`:

```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/run_pipeline_coco.py"
```

> 💡 **Nota**: O script executa o download de 100 imagens da partição `validation` do COCO, treina por 5 épocas com o modelo `yolov8n` (nano) usando a CPU e salva as detecções e métricas na pasta `resultados_coco`. Os parâmetros podem ser facilmente modificados no início do arquivo `run_pipeline_coco.py`.

### Execução por Etapas Individuais (CLI)

Se desejar executar ações isoladamente ou personalizar via CLI usando o `yolo_coco.py`:

#### 1. Download e Treinamento
Baixa um subset do COCO-2017 e inicia o treinamento:
```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/yolo_coco.py" train --classes car person bicycle --samples 100 --epochs 5 --device cpu
```

#### 2. Inferência / Teste em Lote
Aplica o modelo treinado em um conjunto de imagens para desenhar bounding boxes:
```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/yolo_coco.py" infer --weights "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/resultados_coco/runs/train/yolov8n_coco/weights/best.pt" --source "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/datasets/coco_yolo/images/validation" --save
```

#### 3. Validação do Modelo
Mede a precisão média (mAP) e as taxas de acerto no dataset:
```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/yolo_coco.py" val --weights "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/resultados_coco/runs/train/yolov8n_coco/weights/best.pt" --data "Fundamentos e Práticas de Deep Learning/Desafio - Criação de Uma Base de Dados e Treinamento da Rede YOLO/datasets/coco_yolo/dataset.yaml"
```

---

## 📊 Resultados Obtidos (Última Execução)

O modelo `yolov8n` foi treinado com **100 amostras** por **5 épocas**. Seguem as métricas obtidas na validação final:

* **mAP@50**: `0.3070`
* **mAP@50-95**: `0.2029`
* **Precisão (Precision)**: `0.7076`
* **Cobertura (Recall)**: `0.2046`

Os relatórios detalhados com as detecções de cada imagem e os gráficos gerados encontram-se salvos sob a pasta [resultados_coco](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Fundamentos%20e%20Pr%C3%A1ticas%20de%20Deep%20Learning/Desafio%20-%20Cria%C3%A7%C3%A3o%20de%20Uma%20Base%20de%20Dados%20e%20Treinamento%20da%20Rede%20YOLO/resultados_coco/).
