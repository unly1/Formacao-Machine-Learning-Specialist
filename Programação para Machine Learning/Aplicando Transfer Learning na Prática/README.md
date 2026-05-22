# Aplicando Transfer Learning na Prática 🐾

Este diretório contém os códigos e resultados práticos do treinamento de um modelo de Deep Learning voltado para a classificação binária de imagens (Gato vs. Cão) utilizando o dataset clássico do Kaggle **Cats and Dogs**. 

O projeto ilustra o pipeline completo de preparação de dados, definição de arquitetura convolucional (CNN), treinamento e validação, divididos em dois ambientes principais de execução:
1. **Google Colab (Nuvem):** [Aplicando_Transfer_Learning.ipynb](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Programa%C3%A7%C3%A3o%20para%20Machine%20Learning/Aplicando%20Transfer%20Learning%20na%20Pr%C3%A1tica/Aplicando_Transfer_Learning.ipynb)
2. **Ambiente Local (Python):** [aplicando_transfer_learning.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Programa%C3%A7%C3%A3o%20para%20Machine%20Learning/Aplicando%20Transfer%20Learning%20na%20Pr%C3%A1tica/aplicando_transfer_learning.py)

---

## 💻 1. Notebook Google Colab: `Aplicando_Transfer_Learning.ipynb`

O notebook foi estruturado para ser executado no ambiente de nuvem do Google Colab, aproveitando recursos do sistema Linux e GPUs gratuitas.

### Principais etapas realizadas:
* **Download do Dataset:** Utiliza o comando nativo do Linux `!wget` com a opção `--no-check-certificate` para fazer o download do dataset diretamente para a pasta temporária `/tmp` do ambiente virtual.
* **Extração e Estruturação:** Descompacta o arquivo zip contendo as imagens e cria a estrutura de diretórios para treino e teste (`/tmp/cats-v-dogs/training` e `/tmp/cats-v-dogs/testing`).
* **Tratamento de Dados Inválidos:** A função [split_data](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Programa%C3%A7%C3%A3o%20para%20Machine%20Learning/Aplicando%20Transfer%20Learning%20na%20Pr%C3%A1tica/Aplicando_Transfer_Learning.ipynb#L131) varre as imagens de origem e descarta automaticamente arquivos com tamanho igual a `0 bytes` (como os arquivos corrompidos `666.jpg` e `11702.jpg` presentes no dataset original).
* **Divisão Treino/Teste:** Distribui de forma aleatória as imagens em uma proporção de 90% para treinamento e 10% para teste (validação).
* **Modelo Sequencial (CNN):** Cria um modelo sequencial [tf.keras.models.Sequential](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Programa%C3%A7%C3%A3o%20para%20Machine%20Learning/Aplicando%20Transfer%20Learning%20na%20Pr%C3%A1tica/Aplicando_Transfer_Learning.ipynb#L224) contendo 3 camadas de convolução (`Conv2D` com 16, 32 e 64 filtros de tamanho 3x3), seguidas por camadas de max-pooling (`MaxPooling2D`), uma camada densa de 512 neurônios com ativação ReLU e uma camada de saída de 1 neurônio com ativação Sigmoide.
* **Visualização:** Utiliza o `%matplotlib inline` para plotar os gráficos de evolução de Acurácia e Perda ao longo das 15 épocas no próprio notebook.

> [!NOTE]
> No ambiente do Google Colab, a concatenação de caminhos foi implementada de forma simplificada por strings (`SOURCE + filename`), aproveitando a estrutura de diretórios estática e padrão do Linux.

---

## 🏠 2. Script Local: `aplicando_transfer_learning.py`

O script Python local foi adaptado para ser **cross-platform** e rodar de forma robusta em qualquer máquina local (inclusive sistemas Windows).

### Principais melhorias e características da versão local:
* **Caminhos Dinâmicos:** Usa `os.path.dirname(os.path.abspath(__file__))` para obter o diretório atual do script e cria uma pasta `./tmp` localmente. Isso evita conflitos e erros de diretório não encontrado.
* **Download Inteligente e Seguro:**
  * O download só é iniciado se o arquivo `.zip` ainda não existir na máquina, economizando tempo e banda (o dataset possui ~786 MB).
  * Utiliza a biblioteca `urllib.request` juntamente com configurações do módulo `ssl` para evitar erros de certificação HTTPS comuns no ambiente Windows.
* **Robustez de Caminhos:** Substitui as concatenações manuais de strings por `os.path.join()`, garantindo que as barras de caminho sejam renderizadas de acordo com o sistema operacional (barras invertidas `\` no Windows, normais `/` no Linux).
* **Geração Automática de Relatório:** Ao final do treinamento, o script compila o histórico do modelo e as estatísticas dos diretórios, exportando tudo de forma estruturada para o arquivo [resultado.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Programa%C3%A7%C3%A3o%20para%20Machine%20Learning/Aplicando%20Transfer%20Learning%20na%20Pr%C3%A1tica/resultado.txt).

---

## 📊 3. Resumo dos Resultados do Treinamento

A execução do modelo em ambiente local gerou os seguintes resultados documentados em [resultado.txt](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Programa%C3%A7%C3%A3o%20para%20Machine%20Learning/Aplicando%20Transfer%20Learning%20na%20Pr%C3%A1tica/resultado.txt):

### Divisão dos Dados:
| Classe | Imagens de Treino | Imagens de Validação (Teste) |
| :--- | :---: | :---: |
| 🐱 **Gatos** | 12.488 | 3.354 |
| 🐶 **Cães** | 12.390 | 2.366 |

### Arquitetura do Modelo:
* **Camadas Convolucionais:** 3 camadas `Conv2D` intercaladas com `MaxPooling2D`
* **Parâmetros Totais:** 18.989.124 (sendo 9.494.561 treináveis)
* **Função de Perda:** `binary_crossentropy`
* **Otimizador:** `RMSprop` com taxa de aprendizado de 0.001

### Histórico de Métricas Finais:
* **Melhor Acurácia de Treino:** `80.09%` (Época 15)
* **Melhor Acurácia de Validação:** `83.00%` (Época 13)
* **Menor Perda (Loss) de Treino:** `0.4337` (Época 15)
* **Menor Perda (Loss) de Validação:** `0.3804` (Época 15)

> [!TIP]
> Os gráficos gerados ao final do treinamento mostram uma leve divergência a partir da época 10, sugerindo uma tendência ao *overfitting* se o número de épocas for muito ampliado sem técnicas adicionais como *dropout* ou *data augmentation* no gerador de imagens.
