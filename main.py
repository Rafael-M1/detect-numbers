import os
import shutil
import cv2
import mss
import numpy as np

PASTA_PRINTS = "prints_roleta"
PASTA_GABARITO = "gabarito"


def limpar_e_preparar_pasta():
    if os.path.exists(PASTA_PRINTS):
        shutil.rmtree(PASTA_PRINTS)
    os.makedirs(PASTA_PRINTS)


REGIAO = {"top": 120, "left": 1390, "width": 505, "height": 22}

GABARITO = []
if os.path.exists(PASTA_GABARITO):
    for arquivo in os.listdir(PASTA_GABARITO):
        if arquivo.endswith(".png"):
            num_verdadeiro = arquivo.split("_")[0]
            img_gab = cv2.imread(
                os.path.join(PASTA_GABARITO, arquivo), cv2.IMREAD_GRAYSCALE
            )
            GABARITO.append((num_verdadeiro, img_gab))


def extrair_recorte_perfeito(slot_bgr):
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    v_blur = cv2.GaussianBlur(v, (3, 3), 0)

    _, th = cv2.threshold(v_blur, 130, 255, cv2.THRESH_BINARY)
    contornos, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    caixas = []
    for c in contornos:
        cx, cy, cw, ch = cv2.boundingRect(c)
        if cw > 2 and ch > 5:
            caixas.append((cx, cy, cx + cw, cy + ch))

    if caixas:
        x_min = min([b[0] for b in caixas])
        y_min = min([b[1] for b in caixas])
        x_max = max([b[2] for b in caixas])
        y_max = max([b[3] for b in caixas])

        y1 = max(0, y_min - 2)
        y2 = min(slot_bgr.shape[0], y_max + 2)
        x1 = max(0, x_min - 2)
        x2 = min(slot_bgr.shape[1], x_max + 2)

        recorte_num = v[y1:y2, x1:x2]
        return cv2.resize(recorte_num, (30, 20), interpolation=cv2.INTER_CUBIC)

    return cv2.resize(v, (30, 20), interpolation=cv2.INTER_CUBIC)


def reconhecer_por_gabarito(molde_atual):
    if not GABARITO:
        return ""

    melhor_numero = ""
    maior_score = -1.0

    for numero_verdadeiro, img_gab in GABARITO:
        res = cv2.matchTemplate(molde_atual, img_gab, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)

        if max_val > maior_score:
            maior_score = max_val
            melhor_numero = numero_verdadeiro

    # Como agora as imagens estão perfeitamente alinhadas, exigimos alta fidelidade
    return melhor_numero if maior_score > 0.70 else ""


# --- EXECUÇÃO ---
limpar_e_preparar_pasta()

with mss.MSS() as sct:
    screenshot = sct.grab(REGIAO)

img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
cv2.imwrite(os.path.join(PASTA_PRINTS, "captura_total.png"), img)

h, w, _ = img.shape
slot1_w = int(w * 0.095)
resto_w = w - slot1_w
outros_slots_w = resto_w // 12

resultado = []

for i in range(13):
    if i == 0:
        x1, x2 = 0, slot1_w
    else:
        x1 = slot1_w + (i - 1) * outros_slots_w
        x2 = slot1_w + i * outros_slots_w if i < 12 else w

    # Captura com uma folga lateral estratégica de 3 pixels
    xa = max(0, x1 - 3)
    xb = min(w, x2 + 3)
    slot_largo = img[:, xa:xb]

    # Cria o molde focado e centralizado do número atual
    molde_atual = extrair_recorte_perfeito(slot_largo)

    # Salva para monitoramento visual
    cv2.imwrite(
        os.path.join(PASTA_PRINTS, f"slot_{i}_miolo.png"), molde_atual
    )

    numero = reconhecer_por_gabarito(molde_atual)
    resultado.append(numero)

print("Resultado final por Gabarito Dinâmico Corrigido:")
print(resultado)