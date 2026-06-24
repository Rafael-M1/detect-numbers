"""
Avaliação de Performance sobre os Dados de Treino
===================================================
Executa o reconhecimento sobre todas as imagens da pasta 'treino/'
e compara com os valores reais (extraídos do nome do arquivo).

Uso:
    python reconhecer_dados_treino.py
    python reconhecer_dados_treino.py --treino treino --templates templates
    python reconhecer_dados_treino.py --detalhes      (mostra erros por imagem)
"""

import cv2
import sys
import argparse
from pathlib import Path
from reconhecer import (
    segmentar_regioes, binarizar, ocr_regiao,
    BancoTemplates, reconhecer_imagem,
    NUM_REGIOES, VALIDOS,
    PASTA_TREINO, PASTA_TEMPLATES
)

# ─── Cores para o terminal (Windows suporta via ANSI no Python 3.12+) ─────────
VERDE   = "\033[92m"
VERMELHO= "\033[91m"
AMARELO = "\033[93m"
RESET   = "\033[0m"
NEGRITO = "\033[1m"

def colorir(texto, cor):
    return f"{cor}{texto}{RESET}"

# ─── Avaliação ────────────────────────────────────────────────────────────────

def avaliar(pasta_treino: str, pasta_templates: str, mostrar_detalhes: bool):
    pasta = Path(pasta_treino)
    if not pasta.exists():
        print(f"[ERRO] Pasta '{pasta_treino}' não encontrada.")
        sys.exit(1)

    imagens = sorted(list(pasta.glob("*.png")) + list(pasta.glob("*.jpg")))
    if not imagens:
        print(f"[AVISO] Nenhuma imagem encontrada em '{pasta_treino}'.")
        sys.exit(0)

    banco = BancoTemplates(pasta_templates)
    if banco.vazio():
        print(f"[AVISO] Banco de templates vazio. Execute primeiro:")
        print(f"        python reconhecer.py --indexar")

    # Contadores globais
    total_regioes  = 0
    total_acertos  = 0
    erros_por_num  = {}   # {numero_real: [numero_obtido, ...]}
    imgs_perfeitas = 0
    imgs_com_erro  = 0

    resultados = []

    for caminho in imagens:
        # Extrair gabarito do nome do arquivo
        partes = caminho.stem.split("-")
        try:
            gabarito = [int(p) for p in partes]
        except ValueError:
            print(f"  [ignorado] nome inválido: {caminho.name}")
            continue

        if len(gabarito) != NUM_REGIOES:
            print(f"  [ignorado] {caminho.name} → {len(gabarito)} números (esperado {NUM_REGIOES})")
            continue

        img = cv2.imread(str(caminho))
        if img is None:
            print(f"  [ignorado] não foi possível ler: {caminho.name}")
            continue

        obtido = reconhecer_imagem(str(caminho), banco)

        erros = [
            (i + 1, esp, obt)
            for i, (esp, obt) in enumerate(zip(gabarito, obtido))
            if esp != obt
        ]

        acertos = NUM_REGIOES - len(erros)
        total_regioes += NUM_REGIOES
        total_acertos += acertos

        for _, real, pred in erros:
            erros_por_num.setdefault(real, []).append(pred)

        if erros:
            imgs_com_erro += 1
        else:
            imgs_perfeitas += 1

        resultados.append((caminho.name, gabarito, obtido, erros))

    # ─── Relatório por imagem ──────────────────────────────────────────────
    if mostrar_detalhes:
        print(f"\n{'─'*60}")
        print(f" DETALHES POR IMAGEM")
        print(f"{'─'*60}")
        for nome, gabarito, obtido, erros in resultados:
            if erros:
                print(f"\n{colorir('❌', VERMELHO)} {nome}")
                print(f"   Esperado : {gabarito}")
                print(f"   Obtido   : {obtido}")
                for reg, real, pred in erros:
                    print(f"   {colorir(f'Região {reg:2d}', AMARELO)}: esperado={real}  obtido={pred}")
            else:
                print(f"{colorir('✅', VERDE)} {nome}")

    # ─── Resumo de erros por número ───────────────────────────────────────
    if erros_por_num:
        print(f"\n{'─'*60}")
        print(f" NÚMEROS COM MAIS ERROS")
        print(f"{'─'*60}")
        print(f"  {'Número':>8}  {'Erros':>6}  {'Confundido com'}")
        print(f"  {'─'*8}  {'─'*6}  {'─'*20}")
        for num in sorted(erros_por_num, key=lambda n: -len(erros_por_num[n])):
            preds = erros_por_num[num]
            contagem = {}
            for p in preds:
                contagem[p] = contagem.get(p, 0) + 1
            confusoes = ", ".join(
                f"{p}×{c}" for p, c in sorted(contagem.items(), key=lambda x: -x[1])
            )
            print(f"  {num:>8}  {len(preds):>6}  {confusoes}")

    # ─── Resumo global ────────────────────────────────────────────────────
    taxa_acerto = 100 * total_acertos / total_regioes if total_regioes > 0 else 0
    taxa_img    = 100 * imgs_perfeitas / len(resultados) if resultados else 0

    cor_taxa = VERDE if taxa_acerto >= 95 else (AMARELO if taxa_acerto >= 80 else VERMELHO)

    print(f"\n{'═'*60}")
    print(f" RESULTADO FINAL")
    print(f"{'═'*60}")
    print(f"  Imagens avaliadas : {len(resultados)}")
    print(f"  Imagens perfeitas : {colorir(str(imgs_perfeitas), VERDE)}  "
          f"({taxa_img:.1f}%)")
    print(f"  Imagens com erro  : {colorir(str(imgs_com_erro), VERMELHO) if imgs_com_erro else '0'}")
    print(f"  Regiões corretas  : {total_acertos}/{total_regioes}  "
          f"({colorir(f'{taxa_acerto:.1f}%', cor_taxa)})")
    print(f"{'═'*60}\n")

    return taxa_acerto

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Avalia performance do reconhecedor sobre os dados de treino."
    )
    parser.add_argument("--treino", default=PASTA_TREINO,
                        help=f"Pasta de treino (padrão: {PASTA_TREINO})")
    parser.add_argument("--templates", default=PASTA_TEMPLATES,
                        help=f"Pasta de templates (padrão: {PASTA_TEMPLATES})")
    parser.add_argument("--detalhes", action="store_true",
                        help="Mostrar resultado por imagem")
    args = parser.parse_args()

    avaliar(args.treino, args.templates, args.detalhes)


if __name__ == "__main__":
    main()
