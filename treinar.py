import os
import shutil
import cv2
import numpy as np

PASTA_TREINO = "treino"
PASTA_GABARITO = "gabarito"

if os.path.exists(PASTA_GABARITO):
    shutil.rmtree(PASTA_GABARITO)
os.makedirs(PASTA_GABARITO)


def extrair_recorte_perfeito(slot_bgr):
    """Encontra o contorno real do número usando brilho e retorna um corte

    ajustado e centralizado dele (em escala de cinza/brilho).
    """
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    v_blur = cv2.GaussianBlur(v, (3, 3), 0)

    # Cria uma máscara binária apenas para achar a localização do texto
    _, th = cv2.threshold(v_blur, 130, 255, cv2.THRESH_BINARY)

    contornos, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    caixas = []
    for c in contornos:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > 2 and ch > 5:  # Filtra sujeiras pequenas
            caixas.append((cx, cy, cx + cw, cy + ch))

    if caixas:
        # Agrupa todos os contornos encontrados (une o primeiro e o segundo dígito se houver)
        x_min = min([b[0] for b in caixas])
        y_min = min([b[1] for b in caixas])
        x_max = max([b[2] for b in caixas])
        y_max = max([b[3] for b in caixas])

        # Adiciona uma margem de segurança de 2 pixels
        y1 = max(0, y_min - 2)
        y2 = min(slot_bgr.shape[0], y_max + 2)
        x1 = max(0, x_min - 2)
        x2 = min(slot_bgr.shape[1], x_max + 2)

        # Retorna o recorte focado na imagem de brilho (V)
        recorte_num = v[y1:y2, x1:x2]
        # Padroniza para um tamanho fixo para o template matching funcionar perfeitamente
        return cv2.resize(recorte_num, (30, 20), interpolation=cv2.INTER_CUBIC)

    # Se falhar totalmente em achar contornos, devolve o centro padrão redimensionado
    return cv2.resize(v, (30, 20), interpolation=cv2.INTER_CUBIC)


print("Gerando gabarito adaptativo inteligente...")
contadores = {str(n): 0 for n in range(37)}

for nome_arquivo in os.listdir(PASTA_TREINO):
    if not nome_arquivo.endswith((".png", ".jpg", ".jpeg")):
        continue

    caminho_img = os.path.join(PASTA_TREINO, nome_arquivo)
    img = cv2.imread(caminho_img)
    h, w, _ = img.shape

    resultados_esperados = nome_arquivo.split(".")[0].split("-")
    if len(resultados_esperados) != 13:
        print(f"Ignorado: {nome_arquivo} precisa ter 13 números.")
        continue

    slot1_w = int(w * 0.095)
    resto_w = w - slot1_w
    outros_slots_w = resto_w // 12

    for i in range(13):
        numero_alvo = resultados_esperados[i].strip()
        if numero_alvo == "":
            continue

        if i == 0:
            x1, x2 = 0, slot1_w
        else:
            x1 = slot1_w + (i - 1) * outros_slots_w
            x2 = slot1_w + i * outros_slots_w if i < 12 else w

        # Recorta a coluna teórica aproximada com uma folga lateral para não perder nada
        margem_busca = 3
        xa = max(0, x1 - margem_busca)
        xb = min(w, x2 + margem_busca)
        slot_largo = img[:, xa:xb]

        # Deixa o contorno achar o número exato dentro dessa coluna
        molde_perfeito = extrair_recorte_perfeito(slot_largo)

        idx = contadores.get(numero_alvo, 0)
        caminho_salvar = os.path.join(
            PASTA_GABARITO, f"{numero_alvo}_{idx}.png"
        )
        cv2.imwrite(caminho_salvar, molde_perfeito)
        contadores[numero_alvo] = idx + 1

print("Treinamento concluído com sucesso!")