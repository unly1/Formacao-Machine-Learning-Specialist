# Segmentação Semântica com DeepLabv3

Este projeto demonstra como utilizar o modelo **DeepLabv3** pré-treinado no TensorFlow para realizar segmentação semântica em imagens. O script foi modificado para suportar o processamento de múltiplas imagens sequencialmente em uma única execução.

---

## 🚀 Funcionalidades

- **Segmentação Semântica**: Identifica e rotula classes como pessoas, carros, animais, plantas, fundo, etc.
- **Processamento em Lote**: Permite processar múltiplas imagens (locais ou URLs da web) consecutivamente.
- **Cache de Modelos**: Baixa o modelo escolhido automaticamente apenas na primeira execução e o salva em cache na pasta `./models`.
- **Organização de Resultados**: Cria e salva as imagens resultantes da segmentação automaticamente na pasta `./resultados`.

---

## 🛠️ Pré-requisitos

Para executar o script, certifique-se de que possui as dependências necessárias instaladas no seu ambiente virtual:

```bash
pip install tensorflow numpy pillow matplotlib six
```

---

## 📖 Como Usar

Você pode rodar o script diretamente a partir do terminal passando diferentes argumentos.

### 1. Processar as Imagens de Teste Padrão (Batch)
Se você rodar o script sem nenhum argumento, ele baixará/carregará o modelo MobileNetv2 e processará as 3 imagens de exemplo da documentação oficial do TensorFlow:

```bash
python deeplab_segmentacao.py
```

### 2. Processar uma ou mais Imagens Específicas
Você pode passar imagens locais ou URLs externas usando o argumento `--image`:

- **Uma única imagem local:**
  ```bash
  python deeplab_segmentacao.py --image "caminho/para/sua/foto.jpg"
  ```

- **Múltiplas imagens locais ou URLs (separadas por espaços):**
  ```bash
  python deeplab_segmentacao.py --image "foto1.jpg" "https://site.com/foto2.png" "foto3.jpg"
  ```

- **Múltiplas imagens (separadas por vírgula):**
  ```bash
  python deeplab_segmentacao.py --image "foto1.jpg, foto2.jpg"
  ```

### 3. Escolher um Modelo de Deep Learning Diferente
O script disponibiliza 4 modelos pré-treinados para escolha no argumento `--model`:

```bash
python deeplab_segmentacao.py --model xception_coco_voctrainval
```

*Modelos disponíveis:*
- `mobilenetv2_coco_voctrainaug` (Padrão, mais leve e rápido)
- `mobilenetv2_coco_voctrainval`
- `xception_coco_voctrainaug` (Mais preciso, porém mais pesado)
- `xception_coco_voctrainval`

---

## 📂 Estrutura do Projeto

```text
├── deeplab_segmentacao.py   # Script principal de execução
├── README.md                # Documentação do projeto (este arquivo)
├── models/                  # Pasta gerada automaticamente com os modelos pré-treinados (.tar.gz)
└── resultados/              # Pasta com as imagens originais, mapas e overlays salvos
```

---

## 📊 Classes Identificadas (PASCAL VOC)

O modelo foi treinado para identificar as seguintes classes de objetos:

| ID | Classe | ID | Classe |
| :---: | :--- | :---: | :--- |
| **0** | Fundo (Background) | **11** | Mesa de Jantar (Diningtable) |
| **1** | Avião (Aeroplane) | **12** | Cachorro (Dog) |
| **2** | Bicicleta (Bicycle) | **13** | Cavalo (Horse) |
| **3** | Pássaro (Bird) | **14** | Motocicleta (Motorbike) |
| **4** | Barco (Boat) | **15** | Pessoa (Person) |
| **5** | Garrafa (Bottle) | **16** | Planta de Vaso (Pottedplant) |
| **6** | Ônibus (Bus) | **17** | Ovelha (Sheep) |
| **7** | Carro (Car) | **18** | Sofá (Sofa) |
| **8** | Gato (Cat) | **19** | Trem (Train) |
| **9** | Cadeira (Chair) | **20** | TV / Monitor (Tv) |
| **10** | Vaca (Cow) | | |
