import os
import shutil
import cv2
import numpy as np

PASTA_TREINO = "treino"
PASTA_GABARITO = "gabarito"

# Limpa treinos antigos errados
if os.path.exists(PASTA_GABARITO):
    shutil.rmtree(PASTA_GABARITO)
os.makedirs(PASTA_GABARITO)

print("Gerando gabarito visual colorido...")
contadores = {str(n): 0 for n in range(37)}

for nome_arquivo in os.listdir(PASTA_TREINO):
    if not nome_arquivo.endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    caminho_img = os.path.join(PASTA_TREINO, nome_arquivo)
    img = cv2.imread(caminho_img)
    h, w, _ = img.shape
    
    resultados_esperados = nome_arquivo.split('.')[0].split('-')
    if len(resultados_esperados) != 13:
        print(f"Ignorado: {nome_arquivo} precisa ter 13 números separados por '-'")
        continue

    # Divisão fixa ajustada da roleta
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
            
        # Pega o meio exato do slot (evita pegar as bordas/linhas divisórias)
        centro_x = x1 + ((x2 - x1) // 2)
        centro_y = h // 2
        
        # Recorta um quadrado fixo de 20x20 pixels ao redor do centro do número
        # Como o número fica centralizado, isso pega o dígito puro com cores originais
        slot_recortado = img[centro_y-10:centro_y+10, centro_x-10:centro_x+10]
        
        # Salva o arquivo colorido
        idx = contadores.get(numero_alvo, 0)
        caminho_salvar = os.path.join(PASTA_GABARITO, f"{numero_alvo}_{idx}.png")
        cv2.imwrite(caminho_salvar, slot_recortado)
        contadores[numero_alvo] = idx + 1

print("Treinamento concluído! Verifique a pasta 'gabarito'.")