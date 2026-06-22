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

# Carrega as amostras coloridas na memória
GABARITO = []
if os.path.exists(PASTA_GABARITO):
    for arquivo in os.listdir(PASTA_GABARITO):
        if arquivo.endswith(".png"):
            num_verdadeiro = arquivo.split("_")[0]
            img_gab = cv2.imread(os.path.join(PASTA_GABARITO, arquivo))
            GABARITO.append((num_verdadeiro, img_gab))

def reconhecer_por_gabarito(slot_miolo):
    if not GABARITO:
        return ""
        
    melhor_numero = ""
    maior_score = -1.0
    
    # Compara a imagem colorida atual com o gabarito colorido
    for numero_verdadeiro, img_gab in GABARITO:
        res = cv2.matchTemplate(slot_miolo, img_gab, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > maior_score:
            maior_score = max_val
            melhor_numero = numero_verdadeiro
            
    return melhor_numero if maior_score > 0.40 else ""

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
        
    # Pega o mesmo miolo de 20x20 pixels
    centro_x = x1 + ((x2 - x1) // 2)
    centro_y = h // 2
    slot_miolo = img[centro_y-10:centro_y+10, centro_x-10:centro_x+10]
    
    # Salva para você monitorar visualmente
    cv2.imwrite(os.path.join(PASTA_PRINTS, f"slot_{i}_miolo.png"), slot_miolo)
    
    numero = reconhecer_por_gabarito(slot_miolo)
    resultado.append(numero)

print("Resultado final por Gabarito Colorido Estável:")
print(resultado)