"""
Avaliação de Performance sobre os Dados de Treino
===================================================
Validação cruzada "deixa-um-de-fora": cada imagem de 'treino/' é reconhecida
usando um modelo montado com TODAS AS OUTRAS imagens, e o resultado é comparado
com o gabarito do nome do arquivo. Isso mede a precisão real esperada em
imagens novas (e não apenas se o modelo "decorou" o treino).

Uso:
    python reconhecer_dados_treino.py
    python reconhecer_dados_treino.py --detalhes      (mostra todas as imagens)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from reconhecer import (
    Modelo, extrair_amostras, reconhecer_img, rotulo_do_nome,
    NUM_REGIOES, PASTA_TREINO,
)


def avaliar(pasta_treino: str, mostrar_detalhes: bool) -> float:
    imagens = sorted(Path(pasta_treino).glob("*.png"))
    if not imagens:
        print(f"[ERRO] Nenhuma imagem em '{pasta_treino}/'.")
        sys.exit(1)

    # Extrai os dígitos de cada imagem uma única vez
    dados = {}
    for caminho in imagens:
        nums = rotulo_do_nome(caminho)
        img = cv2.imread(str(caminho))
        if nums is None or img is None:
            print(f"  [ignorado] {caminho.name}")
            continue
        X, Y, puladas = extrair_amostras(img, nums)
        if puladas:
            print(f"  [aviso] {caminho.name}: {puladas} casa(s) não segmentadas")
        dados[caminho] = (img, nums, X, Y)

    total = acertos = perfeitas = 0
    erros_por_num = {}

    for caminho, (img, gabarito, _, _) in dados.items():
        X = [x for c, d in dados.items() if c != caminho for x in d[2]]
        Y = [y for c, d in dados.items() if c != caminho for y in d[3]]
        obtido = reconhecer_img(img, Modelo(X, Y))

        erros = [(i + 1, e, o) for i, (e, o) in enumerate(zip(gabarito, obtido)) if e != o]
        total += NUM_REGIOES
        acertos += NUM_REGIOES - len(erros)
        perfeitas += not erros
        for _, real, pred in erros:
            erros_por_num.setdefault(real, []).append(pred)

        if erros:
            print(f"[ERRO] {caminho.name}")
            print(f"   Esperado : {gabarito}")
            print(f"   Obtido   : {obtido}")
        elif mostrar_detalhes:
            print(f"[OK]   {caminho.name}")

    if erros_por_num:
        print("\nNúmeros com erro (real: obtidos):")
        for num, preds in sorted(erros_por_num.items(), key=lambda kv: -len(kv[1])):
            print(f"  {num:>2}: {preds}")

    taxa = 100 * acertos / total if total else 0.0
    print("\n" + "=" * 50)
    print(f"  Imagens avaliadas : {len(dados)}")
    print(f"  Imagens perfeitas : {perfeitas} ({100 * perfeitas / len(dados):.1f}%)")
    print(f"  Casas corretas    : {acertos}/{total} ({taxa:.2f}%)")
    print("=" * 50)
    return taxa


def main():
    parser = argparse.ArgumentParser(description="Avalia o reconhecedor sobre os dados de treino.")
    parser.add_argument("--treino", default=PASTA_TREINO,
                        help=f"Pasta de treino (padrão: {PASTA_TREINO})")
    parser.add_argument("--detalhes", action="store_true", help="Mostrar resultado por imagem")
    args = parser.parse_args()
    avaliar(args.treino, args.detalhes)


if __name__ == "__main__":
    main()
