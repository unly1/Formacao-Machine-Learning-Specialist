# Detecção de Objetos com YOLOv8 & Open Images v7

Este projeto implementa um pipeline completo para download de dados, preparação, treinamento, inferência e validação de modelos de detecção de objetos YOLOv8 utilizando o dataset Open Images v7.

As classes padrão configuradas no pipeline são:
* **Car** (Carro)
* **Person** (Pessoa)
* **Bicycle** (Bicicleta)

---

## 📂 Estrutura do Projeto

* **`yolo_openimages.py`**: O script principal que contém toda a lógica do pipeline através de subcomandos via CLI (`train`, `infer`, `val`).
* **`run_pipeline.py`**: Um script auxiliar que executa todo o fluxo (Download ➔ Treino ➔ Inferência ➔ Validação) de forma sequencial com um único comando.
* **`resultados/`**: Pasta contendo todas as saídas geradas:
  * `runs/`: Pasta nativa do YOLO contendo pesos salvos (`best.pt`), gráficos de treinamento e as imagens de teste anotadas.
  * `resultado_treino.txt`: Resumo das configurações do treinamento executado.
  * `resultado_inferencia.txt`: Relatório contendo os objetos detectados em cada imagem testada.
  * `resultado_validacao.txt`: Relatório das métricas de acurácia obtidas (`mAP@50`, `mAP@50-95`, `Precision`, `Recall`).
* **`datasets/`**: Pasta contendo a partição baixada do dataset localmente.

---

## 🛠️ Requisitos e Instalação

As dependências necessárias são:
* `ultralytics` (YOLO)
* `fiftyone` (para baixar e converter o dataset Open Images)
* `opencv-python`
* `pyyaml`
* `torch` e `torchvision` (instalados automaticamente com a ultralytics)

Caso utilize o ambiente virtual `.venv` do projeto, você pode instalar os pacotes executando:
```bash
.venv/Scripts/pip install ultralytics fiftyone opencv-python pyyaml
```

---

## 🚀 Como Executar

### Execução Completa (Pipeline de Ponta a Ponta)
Para baixar os dados, realizar o treinamento, rodar a inferência de teste e avaliar as métricas tudo de uma vez só, utilize o script auxiliar:

```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/run_pipeline.py"
```

> 💡 **Dica**: Você pode editar as variáveis de configuração no topo do arquivo `run_pipeline.py` para alterar o número de imagens (`SAMPLES`), o número de épocas de treinamento (`EPOCHS`), ou forçar o treinamento por CPU/GPU (`DEVICE`).

---

### Execução por Etapas Individuais (CLI)

Se preferir rodar cada etapa manualmente utilizando a interface de linha de comando (CLI) do `yolo_openimages.py`:

#### 1. Download e Treinamento
Baixa o dataset e inicia o treino do YOLO:
```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/yolo_openimages.py" train --classes Car Person Bicycle --samples 100 --epochs 5 --device auto
```

#### 2. Inferência / Teste em Lote
Aplica o modelo treinado em um conjunto de imagens para desenhar as caixas delimitadoras (*bounding boxes*):
```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/yolo_openimages.py" infer --weights "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/resultados/runs/train/yolov8n_openimages/weights/best.pt" --source "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/datasets/openimages_yolo/images" --save
```

#### 3. Validação de Métricas
Calcula as métricas de validação de precisão e cobertura do modelo:
```powershell
& .venv/Scripts/python.exe "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/yolo_openimages.py" val --weights "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/resultados/runs/train/yolov8n_openimages/weights/best.pt" --data "Fundamentos e Práticas de Deep Learning/Redes de Detecção em Deep Learning/datasets/openimages_yolo/dataset.yaml"
```
