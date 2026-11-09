"""
Gera um gráfico explicando como o Vizinho Mais Próximo (1-NN) reconhece um dígito.

Uso (na raiz do projeto):
    python tools/grafico_knn.py
    python tools/grafico_knn.py --imagem novas/captura_xxx.png --casa 3 --digito 2

Saída: docs/knn.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reconhecer import (  # noqa: E402
    ALT_GLIFO, LARG_GLIFO, PASTA_TREINO, Modelo, mapas_das_casas, separar_digitos, vetorizar,
)

# Cores (texto em tinta neutra; a cor só marca o "vencedor")
SUPERFICIE = "#fcfcfb"
TEXTO      = "#0b0b0b"
TEXTO_2    = "#52514e"
GRADE      = "#e4e3df"
NEUTRO     = "#bdbcb6"
DESTAQUE   = "#2a78d6"


def miniatura(vetor):
    return vetor[:-1].reshape(ALT_GLIFO, LARG_GLIFO)


def eixo_limpo(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO_2, length=0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagem", default="novas/captura_20260624_010522_cbdd78.png")
    p.add_argument("--casa", type=int, default=3, help="casa (1-13) de onde tirar o dígito")
    p.add_argument("--digito", type=int, default=2, help="1º ou 2º dígito da casa")
    p.add_argument("--saida", default="docs/knn.png")
    args = p.parse_args()

    # Se a imagem estiver no treino, ela fica de fora (senão se compara consigo mesma)
    caminho = Path(args.imagem).resolve()
    no_treino = caminho.parent == Path(PASTA_TREINO).resolve()
    modelo = Modelo.do_treino(ignorar=caminho.name if no_treino else None)
    img = cv2.imread(args.imagem)
    if img is None:
        sys.exit(f"[ERRO] não foi possível ler {args.imagem}")
    glifos = separar_digitos(mapas_das_casas(img)[args.casa - 1])
    if args.digito > len(glifos):
        sys.exit(f"[ERRO] a casa {args.casa} tem só {len(glifos)} dígito(s)")
    q = vetorizar(glifos[args.digito - 1])

    dist = np.sqrt(((modelo.X - q) ** 2).sum(axis=1))
    ordem = np.argsort(dist)
    vencedor = int(modelo.Y[ordem[0]])
    menor_por_digito = np.array([dist[modelo.Y == d].min() for d in range(10)])

    # Projeção 2D (PCA) só para visualizar — o 1-NN usa os 225 valores completos
    media = modelo.X.mean(axis=0)
    _, _, vt = np.linalg.svd(modelo.X - media, full_matrices=False)
    proj = (modelo.X - media) @ vt[:2].T
    proj_q = (q - media) @ vt[:2].T

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXTO})
    fig = plt.figure(figsize=(12, 8.2), facecolor=SUPERFICIE)
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 2.4], hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)
    fig.suptitle(f"Como o 1-NN reconhece um dígito  —  resposta: {vencedor}",
                 fontsize=15, fontweight="bold", x=0.06, ha="left")
    fig.text(0.06, 0.925,
             f"Dígito {args.digito} da casa {args.casa} de {Path(args.imagem).name}, "
             f"comparado com {len(modelo.Y)} dígitos de treino",
             color=TEXTO_2, fontsize=10)

    # 1) Dígito consultado + 5 vizinhos mais próximos
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(miniatura(q), cmap="gray_r", vmin=0, vmax=1)
    ax.set_title("Dígito lido", fontsize=10, fontweight="bold")
    for s in ax.spines.values():
        s.set_color(DESTAQUE); s.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    for i in range(5):
        k = ordem[i]
        ax = fig.add_subplot(gs[0, i + 2])
        ax.imshow(miniatura(modelo.X[k]), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"vizinho {i + 1}: \"{modelo.Y[k]}\"", fontsize=9, color=TEXTO)
        ax.set_xlabel(f"dist = {dist[k]:.2f}", fontsize=9, color=TEXTO_2)
        for s in ax.spines.values():
            s.set_color(GRADE)
        ax.set_xticks([]); ax.set_yticks([])
    fig.text(0.215, 0.80, "→", fontsize=26, color=TEXTO_2, ha="center", va="center")

    # 2) Menor distância para cada dígito
    ax = fig.add_subplot(gs[1, :3])
    eixo_limpo(ax)
    cores = [DESTAQUE if d == vencedor else NEUTRO for d in range(10)]
    ax.bar(range(10), menor_por_digito, color=cores, width=0.7, edgecolor=SUPERFICIE, linewidth=2)
    ax.set_xticks(range(10))
    ax.set_xlabel("Dígito", color=TEXTO_2)
    ax.set_ylabel("Distância até o exemplo mais parecido", color=TEXTO_2)
    ax.yaxis.grid(True, color=GRADE, linewidth=0.8); ax.set_axisbelow(True)
    # Marca na base: a barra vencedora pode ter altura ~0
    ax.plot([vencedor - 0.35, vencedor + 0.35], [0, 0], color=DESTAQUE, linewidth=5,
            solid_capstyle="round", clip_on=False, zorder=5)
    ax.get_xticklabels()[vencedor].set_fontweight("bold")
    ax.get_xticklabels()[vencedor].set_color(TEXTO)
    segundo = np.argsort(menor_por_digito)[1]
    for d in (vencedor, segundo):
        ax.annotate(f"{menor_por_digito[d]:.2f}", (d, menor_por_digito[d]),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=9, color=TEXTO, fontweight="bold" if d == vencedor else "normal")
    ax.set_title("Menor distância para cada dígito\n(a menor barra vence)",
                 fontsize=11, fontweight="bold", loc="left")

    # 3) Mapa 2D dos exemplos de treino
    ax = fig.add_subplot(gs[1, 3:])
    eixo_limpo(ax)
    # Cada exemplo é desenhado como o próprio dígito (pontos idênticos são agrupados)
    vistos = set()
    for (x, y), d in zip(proj, modelo.Y):
        chave = (round(x * 4), round(y * 4), int(d))
        if chave in vistos:
            continue
        vistos.add(chave)
        ax.text(x, y, str(d), fontsize=8, color=TEXTO_2, alpha=0.55,
                ha="center", va="center")
    viz = proj[ordem[0]]
    ax.plot([proj_q[0], viz[0]], [proj_q[1], viz[1]], color=DESTAQUE, linewidth=2, zorder=4)
    ax.scatter(proj_q[0], proj_q[1], s=220, marker="*", color=DESTAQUE,
               edgecolors=SUPERFICIE, linewidths=2, zorder=5)
    folga = 0.06 * (proj.max(axis=0) - proj.min(axis=0))
    ax.set_xlim(proj[:, 0].min() - folga[0], proj[:, 0].max() + folga[0])
    ax.set_ylim(proj[:, 1].min() - folga[1], proj[:, 1].max() + folga[1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Mapa dos {len(modelo.Y)} exemplos de treino\n"
                 "★ = dígito lido   — = ligação ao vizinho mais próximo",
                 fontsize=11, fontweight="bold", loc="left")

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=110, facecolor=SUPERFICIE)
    print(f"[OK] gráfico salvo em {saida}")


if __name__ == "__main__":
    main()
