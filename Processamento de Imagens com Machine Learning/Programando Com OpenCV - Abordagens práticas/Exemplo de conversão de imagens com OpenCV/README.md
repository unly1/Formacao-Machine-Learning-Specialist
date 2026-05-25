# Conversão de Imagens: RGB para Tons de Cinza com OpenCV

Este diretório contém um exemplo prático de como baixar uma imagem da internet, carregá-la utilizando o OpenCV, convertê-la de colorida (RGB/BGR) para escala de cinza e exibir/salvar a comparação de visualização utilizando o Matplotlib.

## Conteúdo do Diretório

* **[imagens_rbg_cinza.py](file:///c:/Users/moliv/Documents/Formacao-Machine-Learning-Specialist/Processamento%20de%20Imagens%20com%20Machine%20Learning/Programando%20Com%20OpenCV%20-%20Abordagens%20pr%C3%A1ticas/Exemplo%20de%20convers%C3%A3o%20de%20imagens%20com%20OpenCV/imagens_rbg_cinza.py)**: Script Python principal contendo toda a lógica do exemplo.
* **`imagens/`**: Pasta contendo a imagem de entrada baixada (`Lenna.png`).
* **`resultados/`**: Pasta gerada com as saídas geradas pelo processamento:
  * `lenna_cinza.png`: Imagem da Lenna convertida para tons de cinza.
  * `comparativo_rgb_cinza.png`: Gráfico comparativo gerado via Matplotlib mostrando o antes (colorida) e depois (cinza).

---

## Passos Realizados no Script

### 1. Download de Imagem
Utiliza a biblioteca padrão `urllib.request` com um cabeçalho `User-Agent` simulando um navegador web para evitar erros de bloqueio de requisição (`HTTP Error 403: Forbidden`). A imagem é baixada de:
```python
url_lenna = "https://pns2019.github.io/images/Lenna.png"
```

### 2. Leitura com OpenCV (Espaço BGR)
O OpenCV carrega imagens no padrão **BGR** (Blue, Green, Red). Para que a plotagem com o Matplotlib (que espera o padrão **RGB**) fique correta, foi realizada a conversão de canais:
```python
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

### 3. Conversão para Escala de Cinza
Foi aplicada a conversão de canais de cor de BGR para escala de cinza (`COLOR_BGR2GRAY`):
```python
img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

### 4. Geração de Comparativo
O script plota lado a lado a imagem em formato RGB e a imagem em tons de cinza usando subplots do Matplotlib, e exporta o resultado final como figura.

---

## Como Executar

Para rodar o exemplo, execute o seguinte comando utilizando seu interpretador Python:

```powershell
python imagens_rbg_cinza.py
```
