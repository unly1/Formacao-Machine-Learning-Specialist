import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request

def baixar_imagem(url, pasta_destino, nome_arquivo):
    """
    Cria a pasta de destino caso não exista e realiza o download da imagem a partir da URL.
    """
    # Criar a pasta se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    caminho_completo = os.path.join(pasta_destino, nome_arquivo)
    
    if not os.path.exists(caminho_completo):
        print(f"Baixando imagem de: {url}...")
        try:
            # Usar um cabeçalho User-Agent para evitar problemas de requisição bloqueada
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                with open(caminho_completo, 'wb') as f:
                    f.write(response.read())
            print(f"Imagem baixada e salva com sucesso em: '{caminho_completo}'")
        except Exception as e:
            print(f"Erro ao baixar a imagem: {e}")
            return None
    else:
        print(f"Imagem já existente em: '{caminho_completo}'")
        
    return caminho_completo

def criar_imagem_sintetica():
    """
    Fallback caso o download falhe: Cria uma imagem sintética com formas geométricas simples.
    """
    img = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
    cv2.circle(img, (280, 280), 70, 255, -1)
    pts = np.array([[300, 50], [250, 150], [350, 150]], np.int32)
    cv2.polylines(img, [pts], True, 255, 3)
    cv2.fillPoly(img, [pts], 255)
    return img

def main():
    # Definição dos caminhos e URL
    url_lenna = "https://pns2019.github.io/images/Lenna.png"
    pasta_imagens = "imagens"
    nome_arquivo = "Lenna.png"
    
    # Baixar a imagem
    caminho_imagem = baixar_imagem(url_lenna, pasta_imagens, nome_arquivo)
    
    # Carregar imagem
    img_cinza = None
    if caminho_imagem and os.path.exists(caminho_imagem):
        print(f"Carregando a imagem: '{caminho_imagem}'")
        img_original = cv2.imread(caminho_imagem)
        if img_original is not None:
            # Converter para escala de cinza
            img_cinza = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        else:
            print("Falha ao abrir a imagem baixada. Usando imagem sintética...")
            
    if img_cinza is None:
        print("Usando imagem sintética como fallback...")
        img_cinza = criar_imagem_sintetica()

    # Aplicar o filtro de Sobel no eixo X (detecta bordas verticais)
    # cv2.CV_64F é usado para evitar estouro de representação de dados (valores negativos de gradiente)
    sobel_x = cv2.Sobel(img_cinza, cv2.CV_64F, 1, 0, ksize=3)
    
    # Aplicar o filtro de Sobel no eixo Y (detecta bordas horizontais)
    sobel_y = cv2.Sobel(img_cinza, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcular a magnitude do gradiente (combinação de Sobel X e Y)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    
    # Converter de volta para uint8 (8 bits, 0-255) para exibição e salvamento
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_magnitude_abs = cv2.convertScaleAbs(sobel_magnitude)
    
    # Criar pasta para salvar resultados se não existir
    pasta_resultados = "resultados"
    os.makedirs(pasta_resultados, exist_ok=True)
    
    # Salvar os resultados em disco
    cv2.imwrite(os.path.join(pasta_resultados, 'sobel_x.png'), sobel_x_abs)
    cv2.imwrite(os.path.join(pasta_resultados, 'sobel_y.png'), sobel_y_abs)
    cv2.imwrite(os.path.join(pasta_resultados, 'sobel_magnitude.png'), sobel_magnitude_abs)
    print(f"Imagens resultantes salvas na pasta '{pasta_resultados}'.")

    # Configurar exibição usando Matplotlib
    plt.figure(figsize=(12, 10))
    
    # 1. Imagem Original / Cinza
    plt.subplot(2, 2, 1)
    plt.imshow(img_cinza, cmap='gray')
    plt.title('Imagem Original (Escala de Cinza)')
    plt.axis('off')
    
    # 2. Sobel X
    plt.subplot(2, 2, 2)
    plt.imshow(sobel_x_abs, cmap='gray')
    plt.title('Sobel X (Bordas Verticais)')
    plt.axis('off')
    
    # 3. Sobel Y
    plt.subplot(2, 2, 3)
    plt.imshow(sobel_y_abs, cmap='gray')
    plt.title('Sobel Y (Bordas Horizontais)')
    plt.axis('off')
    
    # 4. Magnitude de Sobel
    plt.subplot(2, 2, 4)
    plt.imshow(sobel_magnitude_abs, cmap='gray')
    plt.title('Sobel Magnitude (Bordas Totais)')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Salvar a figura comparativa
    caminho_comparativo = os.path.join(pasta_resultados, 'comparativo_sobel.png')
    plt.savefig(caminho_comparativo, dpi=300)
    print(f"Figura comparativa salva como '{caminho_comparativo}'.")
    
    # Exibir a janela (se houver interface gráfica disponível)
    try:
        plt.show()
    except Exception as e:
        print(f"Não foi possível abrir a janela do Matplotlib: {e}")

if __name__ == "__main__":
    main()

