import cv2
import mss
import numpy as np
import pytesseract


def criar_regiao(x1, y1, x2, y2):
    return {
        "top": min(y1, y2),
        "left": min(x1, x2),
        "width": abs(x2 - x1),
        "height": abs(y2 - y1)
    }


REGIAO = criar_regiao(1390, 120, 1895, 142)

NUM_SLOTS = 13


def preprocess(slot):
    gray = slot

    # leve blur só para remover ruído de captura
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # binarização mais estável (invertido ajuda em números claros)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # garante fundo branco / número preto consistente
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)

    # MORPH para juntar falhas de pixel
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    return th


def extrair_numero(slot_img):
    img = preprocess(slot_img)

    config = (
        "--psm 7 -c tessedit_char_whitelist=0123456789"
    )

    txt = pytesseract.image_to_string(img, config=config)
    txt = ''.join(filter(str.isdigit, txt))

    if txt == "":
        return ""

    # segurança roleta
    try:
        val = int(txt)

        if val > 36:
            # evita erro tipo 710 virar 10 ou 0
            txt = str(val % 100)

        if val > 36:
            return ""  # ainda inválido

    except:
        return ""

    return txt


with mss.MSS() as sct:
    screenshot = sct.grab(REGIAO)

img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

h, w = gray.shape
slot_w = w // NUM_SLOTS

resultado = []

for i in range(NUM_SLOTS):

    x1 = i * slot_w
    x2 = (i + 1) * slot_w if i < NUM_SLOTS - 1 else w

    slot = gray[:, x1:x2]

    numero = extrair_numero(slot)

    resultado.append(numero)

print("Resultado final:")
print(resultado)