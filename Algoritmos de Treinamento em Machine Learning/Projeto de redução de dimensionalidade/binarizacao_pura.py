"""
=============================================================
  BINARIZAÇÃO DE IMAGEM — SEM BIBLIOTECAS EXTERNAS
=============================================================
Converte uma imagem colorida (PPM/PGM) para:
  1. Níveis de cinza  (0–255)
  2. Binária          (0 ou 255) — preto e branco

Usa APENAS a biblioteca padrão do Python:
  - struct  : leitura de bytes
  - os      : manipulação de arquivos
  - sys     : argumentos

Formato suportado na leitura : PPM (P6) — RGB binário
Formato de saída             : PGM (P5) — cinza binário
                               PBM (P4) — bitmap 1-bit  ← binarizada

Como converter sua imagem para PPM (sem instalar nada extra):
  Windows : Paint → Salvar como BMP, depois use IrfanView para PPM
  Linux   : convert foto.jpg foto.ppm   (ImageMagick)
  Python  : from PIL import Image; Image.open("foto.jpg").save("foto.ppm")

Uso:
  python binarizacao_pura.py entrada.ppm [threshold]
  threshold padrão = 128  (0–255)
=============================================================
"""

import sys
import os
import struct

# Configura stdout para UTF-8 para evitar erros de codificação no console Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass



# ─────────────────────────────────────────────────────────────
# 1. LEITURA DE ARQUIVO PPM (P6 — RGB binário)
# ─────────────────────────────────────────────────────────────
def ler_ppm(caminho: str):
    """
    Lê um arquivo PPM binário (P6) e retorna:
      largura, altura, max_val, pixels
    pixels: lista de listas de tuplas (R, G, B)
    """
    with open(caminho, "rb") as f:
        dados = f.read()

    pos = 0

    def ler_linha():
        nonlocal pos
        linha = b""
        while pos < len(dados):
            c = dados[pos:pos+1]
            pos += 1
            if c == b"\n":
                break
            linha += c
        return linha.decode("ascii").strip()

    def pular_comentarios():
        nonlocal pos
        while pos < len(dados) and dados[pos:pos+1] == b"#":
            while pos < len(dados) and dados[pos:pos+1] != b"\n":
                pos += 1
            pos += 1  # pula o \n

    # Cabeçalho
    magic = ler_linha()
    if magic != "P6":
        raise ValueError(f"Formato não suportado: '{magic}'. Esperado P6 (PPM binário).")

    pular_comentarios()

    dimensoes = ler_linha()
    while dimensoes.startswith("#"):
        dimensoes = ler_linha()
    largura, altura = map(int, dimensoes.split())

    max_val_str = ler_linha()
    while max_val_str.startswith("#"):
        max_val_str = ler_linha()
    max_val = int(max_val_str)

    # Dados de pixel (3 bytes por pixel: R, G, B)
    total_bytes = largura * altura * 3
    raw = dados[pos: pos + total_bytes]

    if len(raw) < total_bytes:
        raise ValueError("Arquivo PPM truncado ou corrompido.")

    # Constrói matriz de pixels
    pixels = []
    idx = 0
    for _ in range(altura):
        linha_pixels = []
        for _ in range(largura):
            r = raw[idx]
            g = raw[idx + 1]
            b = raw[idx + 2]
            idx += 3
            linha_pixels.append((r, g, b))
        pixels.append(linha_pixels)

    return largura, altura, max_val, pixels


# ─────────────────────────────────────────────────────────────
# 2. CONVERSÃO PARA ESCALA DE CINZA
# ─────────────────────────────────────────────────────────────
def rgb_para_cinza(pixels, largura, altura):
    """
    Fórmula padrão de luminância (BT.601):
      Y = 0.299·R + 0.587·G + 0.114·B

    Retorna matriz 2D de inteiros (0–255).
    """
    cinza = []
    for linha in pixels:
        linha_cinza = []
        for (r, g, b) in linha:
            # Multiplicação inteira para evitar float puro
            # round() arredonda para o inteiro mais próximo
            y = round(0.299 * r + 0.587 * g + 0.114 * b)
            y = max(0, min(255, y))   # garante intervalo válido
            linha_cinza.append(y)
        cinza.append(linha_cinza)
    return cinza


# ─────────────────────────────────────────────────────────────
# 3. BINARIZAÇÃO (limiarização / thresholding)
# ─────────────────────────────────────────────────────────────
def binarizar(cinza, largura, altura, threshold=128):
    """
    Para cada pixel:
      se valor >= threshold  →  255 (branco)
      caso contrário         →    0 (preto)

    Retorna matriz 2D de 0 ou 255.
    """
    binaria = []
    for linha in cinza:
        linha_bin = []
        for valor in linha:
            linha_bin.append(255 if valor >= threshold else 0)
        binaria.append(linha_bin)
    return binaria


# ─────────────────────────────────────────────────────────────
# 4. THRESHOLD AUTOMÁTICO (Otsu — sem libs)
# ─────────────────────────────────────────────────────────────
def otsu_threshold(cinza, largura, altura):
    """
    Calcula o limiar ótimo pelo método de Otsu:
    Maximiza a variância ENTRE as classes (fundo vs. objeto).
    Retorna o valor de threshold ótimo.
    """
    # Histograma (256 níveis)
    hist = [0] * 256
    total = largura * altura
    for linha in cinza:
        for v in linha:
            hist[v] += 1

    # Probabilidade de cada nível
    prob = [h / total for h in hist]

    melhor_t   = 0
    melhor_var = -1.0

    soma_total = sum(i * prob[i] for i in range(256))

    soma_fundo = 0.0
    peso_fundo = 0.0

    for t in range(256):
        peso_fundo += prob[t]
        if peso_fundo == 0:
            continue

        peso_objeto = 1.0 - peso_fundo
        if peso_objeto == 0:
            break

        soma_fundo += t * prob[t]
        media_fundo  = soma_fundo / peso_fundo
        media_objeto = (soma_total - soma_fundo) / peso_objeto

        # Variância inter-classe
        var = peso_fundo * peso_objeto * (media_fundo - media_objeto) ** 2

        if var > melhor_var:
            melhor_var = var
            melhor_t   = t

    return melhor_t


# ─────────────────────────────────────────────────────────────
# 5. ESCRITA — PGM (cinza) e PPM binarizado
# ─────────────────────────────────────────────────────────────
def salvar_pgm(caminho, cinza, largura, altura):
    """Salva imagem em escala de cinza como PGM P5 (binário)."""
    with open(caminho, "wb") as f:
        header = f"P5\n{largura} {altura}\n255\n"
        f.write(header.encode("ascii"))
        for linha in cinza:
            f.write(bytes(linha))
    print(f"  [OK] Cinza salvo em: {caminho}")


def salvar_ppm_binario(caminho, binaria, largura, altura):
    """
    Salva imagem binarizada como PPM P6 (visualizável em qualquer viewer).
    Pixels 255 → branco (255,255,255) | 0 → preto (0,0,0)
    """
    with open(caminho, "wb") as f:
        header = f"P6\n{largura} {altura}\n255\n"
        f.write(header.encode("ascii"))
        for linha in binaria:
            row = bytearray()
            for v in linha:
                row += bytes([v, v, v])   # R=G=B → tons de cinza / PB
            f.write(row)
    print(f"  [OK] Binária salva em: {caminho}")


def salvar_imagem_pillow(caminho, matriz, largura, altura):
    """
    Salva uma matriz 2D (cinza ou binária) usando Pillow.
    """
    from PIL import Image
    dados_planos = [pixel for linha in matriz for pixel in linha]
    img = Image.new("L", (largura, altura))
    img.putdata(dados_planos)
    img.save(caminho)
    print(f"  [OK] Imagem salva em: {caminho}")


# ─────────────────────────────────────────────────────────────
# 6. ESTATÍSTICAS (sem libs)
# ─────────────────────────────────────────────────────────────
def estatisticas(cinza, largura, altura):
    """Calcula e exibe estatísticas básicas da imagem em cinza."""
    total  = largura * altura
    soma   = 0
    minimo = 255
    maximo = 0
    hist   = [0] * 256

    for linha in cinza:
        for v in linha:
            soma   += v
            hist[v] += 1
            if v < minimo: minimo = v
            if v > maximo: maximo = v

    media = soma / total

    # Variância
    var = sum((v - media) ** 2 * hist[v] for v in range(256)) / total
    desvio = var ** 0.5

    # Mediana
    acumulado = 0
    mediana   = 0
    for v in range(256):
        acumulado += hist[v]
        if acumulado >= total / 2:
            mediana = v
            break

    print("\n  ── Estatísticas da imagem em cinza ──")
    print(f"     Dimensões : {largura} x {altura} px  ({total:,} pixels)")
    print(f"     Mínimo    : {minimo}")
    print(f"     Máximo    : {maximo}")
    print(f"     Média     : {media:.2f}")
    print(f"     Desvio    : {desvio:.2f}")
    print(f"     Mediana   : {mediana}")
    return hist


def imprimir_histograma_ascii(hist, largura_barra=40):
    """Histograma ASCII simplificado (16 faixas)."""
    print("\n  ── Histograma ASCII (16 faixas) ──")
    faixas = 16
    por_faixa = 256 // faixas
    max_count = 0
    grupos = []
    for f in range(faixas):
        inicio = f * por_faixa
        fim    = inicio + por_faixa
        cnt    = sum(hist[inicio:fim])
        grupos.append((inicio, fim - 1, cnt))
        if cnt > max_count:
            max_count = cnt

    for (ini, fim, cnt) in grupos:
        barra = int(cnt / max_count * largura_barra) if max_count > 0 else 0
        print(f"  {ini:>3}–{fim:<3} │{'█' * barra:<{largura_barra}}│ {cnt:>7}")


# ─────────────────────────────────────────────────────────────
# 7. FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BINARIZAÇÃO DE IMAGEM — SEM BIBLIOTECAS EXTERNAS")
    print("=" * 60)

    # ── Parâmetros de entrada
    caminho_entrada = None
    usar_otsu = False
    threshold_val = 128

    if len(sys.argv) < 2:
        # Verifica se o Sonic.png padrão existe no diretório 'imagem' relativo ao script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        caminho_sonic = os.path.join(script_dir, "imagem", "Sonic.png")
        
        if os.path.exists(caminho_sonic):
            caminho_entrada = caminho_sonic
            usar_otsu = True
            threshold_val = None
            print(f"\nNenhum argumento fornecido. Utilizando imagem padrão: {caminho_entrada} (com Otsu)")
        else:
            print("\nUso: python binarizacao_pura.py entrada.ppm [threshold|auto]")
            print("  threshold : inteiro 0–255  (padrão: 128)")
            print("  auto      : usa Otsu para calcular threshold automaticamente")

            # Demo: cria uma imagem PPM sintética colorida para teste
            print("\n[Demo] Criando imagem sintética 64x64 para demonstração...")
            caminho_entrada = "imagem_demo.ppm"
            larg, alt = 64, 64
            with open(caminho_entrada, "wb") as f:
                f.write(f"P6\n{larg} {alt}\n255\n".encode())
                for y in range(alt):
                    for x in range(larg):
                        # Gradiente colorido
                        r = int(x / larg * 255)
                        g = int(y / alt * 255)
                        b = 128
                        f.write(bytes([r, g, b]))
            threshold_val = 128
            usar_otsu = False
            print(f"  Imagem demo criada: {caminho_entrada}")
    else:
        caminho_entrada = sys.argv[1]
        if len(sys.argv) >= 3:
            arg3 = sys.argv[2].lower()
            if arg3 == "auto":
                usar_otsu = True
                threshold_val = None
            else:
                usar_otsu = False
                threshold_val = int(arg3)
                if not (0 <= threshold_val <= 255):
                    raise ValueError("Threshold deve estar entre 0 e 255.")
        else:
            usar_otsu = False
            threshold_val = 128

    if not os.path.exists(caminho_entrada):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_entrada}")

    base = os.path.splitext(caminho_entrada)[0]
    ext = os.path.splitext(caminho_entrada)[1].lower()
    is_pillow_format = ext in [".png", ".jpg", ".jpeg", ".bmp"]

    if is_pillow_format:
        caminho_cinza   = base + "_cinza" + ext
        caminho_binaria = base + "_binaria" + ext
    else:
        caminho_cinza   = base + "_cinza.pgm"
        caminho_binaria = base + "_binaria.ppm"

    # ── Passo 1: Leitura
    print(f"\n[1/4] Lendo imagem: {caminho_entrada}")
    if is_pillow_format:
        from PIL import Image
        img = Image.open(caminho_entrada)
        img = img.convert("RGB")
        largura, altura = img.size
        max_val = 255
        
        dados_raw = list(img.getdata())
        pixels = []
        for y in range(altura):
            linha_pixels = dados_raw[y * largura : (y + 1) * largura]
            pixels.append(linha_pixels)
    else:
        largura, altura, max_val, pixels = ler_ppm(caminho_entrada)
        
    print(f"      Dimensões : {largura} x {altura}")
    print(f"      Max valor  : {max_val}")

    # ── Passo 2: Cinza
    print("\n[2/4] Convertendo para escala de cinza (BT.601)...")
    cinza = rgb_para_cinza(pixels, largura, altura)

    hist = estatisticas(cinza, largura, altura)
    imprimir_histograma_ascii(hist)

    if is_pillow_format:
        salvar_imagem_pillow(caminho_cinza, cinza, largura, altura)
    else:
        salvar_pgm(caminho_cinza, cinza, largura, altura)

    # ── Passo 3: Threshold
    if usar_otsu:
        print("\n[3/4] Calculando threshold automático (Otsu)...")
        threshold_val = otsu_threshold(cinza, largura, altura)
        print(f"      Threshold de Otsu: {threshold_val}")
    else:
        print(f"\n[3/4] Usando threshold manual: {threshold_val}")

    # ── Passo 4: Binarização
    print("\n[4/4] Binarizando imagem...")
    binaria = binarizar(cinza, largura, altura, threshold_val)

    # Contagem de pixels pretos e brancos
    total    = largura * altura
    brancos  = sum(1 for linha in binaria for v in linha if v == 255)
    pretos   = total - brancos
    print(f"      Pixels brancos : {brancos:>8,}  ({100*brancos/total:.1f}%)")
    print(f"      Pixels pretos  : {pretos:>8,}  ({100*pretos/total:.1f}%)")

    if is_pillow_format:
        salvar_imagem_pillow(caminho_binaria, binaria, largura, altura)
    else:
        salvar_ppm_binario(caminho_binaria, binaria, largura, altura)

    print("\n" + "=" * 60)
    print("  CONCLUÍDO")
    print("=" * 60)
    print(f"  Entrada  : {caminho_entrada}")
    print(f"  Cinza    : {caminho_cinza}")
    print(f"  Binária  : {caminho_binaria}")
    print(f"  Threshold: {threshold_val}")
    if is_pillow_format:
        print(f"\nAbra os arquivos correspondentes em qualquer visualizador de imagens.")
    else:
        print("\nAbra os arquivos .pgm/.ppm em qualquer visualizador de imagens.")


if __name__ == "__main__":
    main()
