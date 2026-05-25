# Métodos de Segmentação Semântica na Prática

Este repositório contém a demonstração prática do modelo **DeepLab** para a tarefa de segmentação semântica de imagens, atualizado e configurado para execução local moderna.

O script original é baseado no [Google Colab original](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb#scrollTo=c4oXKmnjw6i_). Ele foi atualizado e renomeado para [deeplab.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Com%20OpenCV/M%C3%A9todos%20de%20Segmenta%C3%A7%C3%A3o%20Sem%C3%A2ntica%20na%20pr%C3%A1tica/deeplab.py) para corrigir incompatibilidades com versões recentes do Python, TensorFlow e Pillow.

---

## 🛠️ Atualizações e Melhorias Realizadas

1. **Compatibilidade com TensorFlow 2.x:**
   - O código original utilizava a API legada do TensorFlow 1.x. Ele foi adaptado para utilizar o módulo de compatibilidade do TensorFlow 2.x (`tensorflow.compat.v1`) com a execução em modo clássico desabilitando o comportamento V2 (`tf.disable_v2_behavior()`). Isso permite rodar o modelo original sem precisar fazer downgrade do TensorFlow ou do Python.

2. **Compatibilidade com Pillow 10+:**
   - O parâmetro depreciado e removido `Image.ANTIALIAS` foi substituído pelo moderno `Image.Resampling.LANCZOS`, prevenindo erros de execução em ambientes modernos de manipulação de imagens.

3. **Sistema de Cache Local de Modelos (`/models`):**
   - No script original, o modelo era baixado em uma pasta temporária do sistema operacional a cada execução. O script foi refatorado para salvar o modelo em uma pasta local chamada `models`. Ele verifica se o modelo já existe antes de baixar, economizando tempo e banda de internet.

4. **Salvamento Automático de Resultados (`/resultados`):**
   - O script agora salva automaticamente o resultado do gráfico gerado (imagem original, máscara de segmentação e overlay) dentro de um diretório local chamado `resultados`. O nome do arquivo gerado acompanha o nome do arquivo original (exemplo: `image1_segmentation.png`).

5. **Interface de Linha de Comando (CLI):**
   - Adicionada a biblioteca `argparse` para que você possa parametrizar o modelo, a imagem de entrada e outras opções diretamente no terminal de forma flexível.

---

## 🚀 Como Executar

Utilize o ambiente virtual do projeto (`.venv`) para rodar o script no terminal:

```powershell
& "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\.venv\Scripts\python.exe" deeplab.py
```

### ⚙️ Opções e Argumentos

Você pode personalizar a execução passando argumentos por linha de comando:

*   **Verificar Opções Disponíveis:**
    ```powershell
    & "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\.venv\Scripts\python.exe" deeplab.py --help
    ```

*   **Segmentar uma Imagem Local ou URL Específica:**
    ```powershell
    & "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\.venv\Scripts\python.exe" deeplab.py --image "caminho/para/sua_imagem.jpg"
    # ou
    & "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\.venv\Scripts\python.exe" deeplab.py --image "https://site.com/foto.jpg"
    ```

*   **Alterar o Modelo de Segmentação:**
    Escolha entre os modelos disponíveis (`mobilenetv2_coco_voctrainaug`, `mobilenetv2_coco_voctrainval`, `xception_coco_voctrainaug`, `xception_coco_voctrainval`):
    ```powershell
    & "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\.venv\Scripts\python.exe" deeplab.py --model xception_coco_voctrainval
    ```

*   **Escolher uma Imagem de Teste (Sample):**
    Caso não queira passar uma imagem personalizada, você pode escolher entre os samples padrão (`image1`, `image2`, `image3`):
    ```powershell
    & "c:\Users\moliv\Documents\Formacao-Machine-Learning-Specialist\.venv\Scripts\python.exe" deeplab.py --sample image2
    ```

---

## 📂 Estrutura de Pastas Gerada

Após a primeira execução, a estrutura de arquivos da pasta estará organizada da seguinte forma:

```text
Métodos de Segmentação Semântica na prática/
├── deeplab.py             # Script principal atualizado
├── README.md              # Este arquivo de documentação
├── models/                # Pasta contendo os modelos DeepLab baixados (cache)
│   └── deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz
└── resultados/            # Pasta contendo os plots de resultados salvos
    └── image1_segmentation.png
```
