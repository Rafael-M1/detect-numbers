import cv2
import numpy as np
import mss
import os
import uuid
from datetime import datetime

# Região da tela
REGIAO = {
    "top": 120,
    "left": 1390,
    "width": 505,
    "height": 22
}

PASTA_SAIDA = "novas"
os.makedirs(PASTA_SAIDA, exist_ok=True)


def capturar_uma_vez():
    with mss.MSS() as sct:

        screenshot = sct.grab(REGIAO)

        img = np.array(screenshot)

        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        return img


def gerar_nome_unico():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_id = uuid.uuid4().hex[:6]

    return f"captura_{timestamp}_{random_id}.png"


if __name__ == "__main__":

    frame = capturar_uma_vez()

    nome_arquivo = gerar_nome_unico()

    caminho = os.path.join(PASTA_SAIDA, nome_arquivo)

    cv2.imwrite(caminho, frame)

    print(f"[OK] Captura salva em: {caminho}")