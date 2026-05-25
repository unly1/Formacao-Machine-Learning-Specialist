import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import urllib.request


def baixar_imagem(url, pasta_destino, nome_arquivo):
    """
    Cria a pasta de destino caso não exista e realiza o download da imagem a partir da URL.
    """
    os.makedirs(pasta_destino, exist_ok=True)
    caminho_completo = os.path.join(pasta_destino, nome_arquivo)

    if not os.path.exists(caminho_completo):
        print(f"Baixando imagem de: {url}...")
        try:
            # Usar um cabeçalho User-Agent para evitar problemas de requisição bloqueada
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                with open(caminho_completo, 'wb') as f:
                    f.write(response.read())
            print(
                f"Imagem baixada e salva com sucesso em: '{caminho_completo}'")
        except Exception as e:
            print(f"Erro ao baixar a imagem: {e}")
            return None
    else:
        print(f"Imagem já existente em: '{caminho_completo}'")

    return caminho_completo


def main():
    # Obter o diretório do script para que as pastas de imagens e resultados fiquem salvas dentro dele
    diretorio_script = os.path.dirname(os.path.abspath(__file__))

    # URL da imagem
    url = "https://cdn.assinafy.com.br/website/base/9b8/619/251/contrato-uso-imagem.jpg"
    pasta_imagens = os.path.join(diretorio_script, "imagens")
    nome_arquivo = "uso-imagem.png"

    # Baixar a imagem
    caminho_imagem = baixar_imagem(url, pasta_imagens, nome_arquivo)

    if caminho_imagem and os.path.exists(caminho_imagem):
        # Carregar a imagem com OpenCV (formato BGR padrão, suportando caracteres Unicode no caminho)
        img_bgr = cv2.imdecode(np.fromfile(
            caminho_imagem, dtype=np.uint8), cv2.IMREAD_COLOR)

        if img_bgr is not None:
            # 1. Converter de BGR (OpenCV) para RGB (Matplotlib) para exibição correta
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # 2. Converter a imagem de BGR para Escala de Cinza
            img_cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Criar pasta para salvar os resultados
            pasta_resultados = os.path.join(diretorio_script, "resultados")
            os.makedirs(pasta_resultados, exist_ok=True)

            # Salvar a imagem em escala de cinza no disco (suportando caracteres Unicode no caminho)
            caminho_cinza = os.path.join(pasta_resultados, "uso-imagem.png")
            sucesso_cinza, buffer_cinza = cv2.imencode(".png", img_cinza)
            if sucesso_cinza:
                buffer_cinza.tofile(caminho_cinza)
                print(
                    f"Imagem em escala de cinza salva com sucesso em: '{caminho_cinza}'")
            else:
                print(
                    f"Erro ao salvar a imagem em escala de cinza em: '{caminho_cinza}'")

            # --- FILTRO SOBEL ---
            # Aplicar o filtro de Sobel no eixo X (detecta bordas verticais)
            sobel_x = cv2.Sobel(img_cinza, cv2.CV_64F, 1, 0, ksize=3)

            # Aplicar o filtro de Sobel no eixo Y (detecta bordas horizontais)
            sobel_y = cv2.Sobel(img_cinza, cv2.CV_64F, 0, 1, ksize=3)

            # Calcular a magnitude do gradiente (combinação de Sobel X e Y)
            sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

            # Converter de volta para uint8 (8 bits, 0-255) para exibição e salvamento
            sobel_x_abs = cv2.convertScaleAbs(sobel_x)
            sobel_y_abs = cv2.convertScaleAbs(sobel_y)
            sobel_magnitude_abs = cv2.convertScaleAbs(sobel_magnitude)

            # Salvar os resultados do Sobel no disco (suportando caracteres Unicode no caminho)
            caminhos_sobel = {
                "sobel_x.png": sobel_x_abs,
                "sobel_y.png": sobel_y_abs,
                "sobel_magnitude.png": sobel_magnitude_abs
            }

            for nome, img_filt in caminhos_sobel.items():
                caminho_filt = os.path.join(pasta_resultados, nome)
                sucesso_filt, buffer_filt = cv2.imencode(".png", img_filt)
                if sucesso_filt:
                    buffer_filt.tofile(caminho_filt)
                    print(
                        f"Resultado do filtro Sobel '{nome}' salvo em: '{caminho_filt}'")
                else:
                    print(
                        f"Erro ao salvar o resultado '{nome}' em: '{caminho_filt}'")

            # Configurar exibição comparativa usando Matplotlib (RGB vs Cinza)
            plt.figure(figsize=(10, 5))

            # Exibir Imagem Colorida (RGB)
            plt.subplot(1, 2, 1)
            plt.imshow(img_rgb)
            plt.title("Imagem Colorida (RGB)")
            plt.axis("off")

            # Exibir Imagem em Escala de Cinza
            plt.subplot(1, 2, 2)
            plt.imshow(img_cinza, cmap="gray")
            plt.title("Imagem em Escala de Cinza")
            plt.axis("off")

            plt.tight_layout()

            # Salvar o gráfico comparativo RGB vs Cinza
            caminho_comparativo = os.path.join(
                pasta_resultados, "comparativo_rgb_cinza.png")
            plt.savefig(caminho_comparativo, dpi=300)
            print(
                f"Gráfico comparativo RGB vs Cinza salvo em: '{caminho_comparativo}'")

            # Configurar exibição comparativa do filtro Sobel
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

            # Salvar a figura comparativa Sobel
            caminho_comparativo_sobel = os.path.join(
                pasta_resultados, 'comparativo_sobel.png')
            plt.savefig(caminho_comparativo_sobel, dpi=300)
            print(
                f"Figura comparativa Sobel salva em: '{caminho_comparativo_sobel}'")

            # Exibir a janela gráfica se possível
            try:
                plt.show()
            except Exception as e:
                print(
                    f"Não foi possível abrir a janela do Matplotlib (provavelmente por falta de ambiente gráfico): {e}")
        else:
            print("Erro ao decodificar a imagem carregada com OpenCV.")
    else:
        print("Erro: Não foi possível obter ou carregar a imagem da Lenna.")


if __name__ == "__main__":
    main()
