"""
Reconhecedor de Números de Roleta
==================================
Lê uma imagem (22x505) com 13 números em linha e retorna um array com os valores.

Como funciona:
  1. Corta a imagem nas 13 casas (posições fixas: a 1ª é a caixa grande,
     as demais têm ~38,33 px de espaçamento).
  2. Em cada casa, estima a cor do fundo linha a linha pela margem esquerda
     (onde nunca há dígito) e calcula o contraste de cada pixel contra esse
     fundo — funciona com texto branco/vermelho/verde e fundo escuro, dourado
     ou colorido.
  3. Separa os dígitos individuais (componentes conectados; se dois dígitos
     estiverem encostados, divide na coluna de menor tinta).
  4. Classifica cada dígito (0–9) pelo vizinho mais próximo entre os dígitos
     extraídos das imagens de treino. O rótulo vem do nome do arquivo.

Uso:
    python reconhecer.py imagem.png
    python reconhecer.py imagem.png --debug
    python reconhecer.py pasta_com_imagens/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── Configurações ────────────────────────────────────────────────────────────

PASTA_TREINO = "treino"
NUM_REGIOES  = 13
LARGURA_IMG  = 505
ALTURA_IMG   = 22
PASSO_CASA   = 38.33     # distância entre centros das casas
LIMIAR       = 90        # contraste mínimo para considerar pixel como dígito
ALT_GLIFO    = 16        # tamanho normalizado do dígito
LARG_GLIFO   = 14
DIST_ALERTA  = 5.0       # distância acima disso → reconhecimento duvidoso
                         # (no treino: mesmo dígito ≤ 5,8; dígitos diferentes ≥ 2,2)
VALIDOS      = set(range(0, 37))

# ─── Segmentação ──────────────────────────────────────────────────────────────

def _contraste(casa: np.ndarray, cols_fundo: slice, x_min: int, x_max: int) -> np.ndarray:
    """Distância de cor de cada pixel ao fundo (estimado por linha)."""
    fundo = np.median(casa[:, cols_fundo], axis=1, keepdims=True)
    d = np.sqrt(((casa - fundo) ** 2).sum(axis=2))
    d[:, :x_min] = 0
    d[:, x_max:] = 0
    return d


def mapas_das_casas(img: np.ndarray) -> list:
    """Retorna, para cada uma das 13 casas, o mapa de contraste (float)."""
    if img.shape[:2] != (ALTURA_IMG, LARGURA_IMG):
        img = cv2.resize(img, (LARGURA_IMG, ALTURA_IMG), interpolation=cv2.INTER_AREA)
    f = img.astype(np.float32)

    # Casa 0: caixa grande (x 0..40), fonte maior
    mapas = [_contraste(f[:, 0:40], slice(1, 5), 5, 38)]

    for i in range(1, NUM_REGIOES):
        x1 = int(round(20 + i * PASSO_CASA - 19))
        mapas.append(_contraste(f[:, x1:x1 + 38], slice(4, 10), 8, 36))
    return mapas


def _dividir(d: np.ndarray, a: int, b: int) -> list:
    """Divide um bloco com dois dígitos encostados na coluna de menor tinta."""
    tinta = (d[:, a:b] > LIMIAR).sum(axis=0)
    meio = (b - a) // 2
    lo, hi = max(3, meio - 5), min(b - a - 3, meio + 6)
    k = lo + int(np.argmin(tinta[lo:hi]))
    return [(a, a + k), (a + k, b)]


def separar_digitos(d: np.ndarray) -> list:
    """Recorta cada dígito da casa. Retorna lista de mapas (esquerda → direita)."""
    m = (d > LIMIAR).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    validos = [i for i in range(1, n) if st[i, 3] >= 8]   # ignora ruído baixo
    blocos = sorted((st[i, 0], st[i, 0] + st[i, 2]) for i in validos)

    unidos = []
    for a, b in blocos:
        if unidos and a < unidos[-1][1]:
            unidos[-1] = (unidos[-1][0], max(b, unidos[-1][1]))
        else:
            unidos.append((a, b))

    altura = max((st[i, 3] for i in validos), default=14)
    faixas = []
    for a, b in unidos:
        if len(unidos) == 1 and (b - a) > altura * 0.95:
            faixas += _dividir(d, a, b)
        else:
            faixas.append((a, b))

    digitos = []
    for a, b in faixas:
        sub = d[:, a:b]
        ys = np.where((sub > LIMIAR).any(axis=1))[0]
        if len(ys):
            digitos.append(sub[ys[0]:ys[-1] + 1])
    return digitos


def vetorizar(g: np.ndarray) -> np.ndarray:
    """Normaliza o dígito para altura fixa (mantendo proporção) e achata."""
    g = np.clip(g / 255.0, 0, 1).astype(np.float32)
    h, w = g.shape
    nw = max(1, min(LARG_GLIFO, int(round(w * ALT_GLIFO / h))))
    r = cv2.resize(g, (nw, ALT_GLIFO), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((ALT_GLIFO, LARG_GLIFO), np.float32)
    off = (LARG_GLIFO - nw) // 2
    canvas[:, off:off + nw] = r
    return np.concatenate([canvas.ravel(), [w / h * 2]])

# ─── Modelo (dígitos de treino) ───────────────────────────────────────────────

def rotulo_do_nome(caminho: Path) -> Optional[list]:
    try:
        nums = [int(p) for p in caminho.stem.split("-")]
    except ValueError:
        return None
    if len(nums) != NUM_REGIOES or any(n not in VALIDOS for n in nums):
        return None
    return nums


def extrair_amostras(img: np.ndarray, numeros: list):
    """Extrai (vetores, dígitos) de uma imagem rotulada. Casas inconsistentes são puladas."""
    X, Y, puladas = [], [], 0
    for d, n in zip(mapas_das_casas(img), numeros):
        gs = separar_digitos(d)
        if len(gs) != len(str(n)):
            puladas += 1
            continue
        for g, ch in zip(gs, str(n)):
            X.append(vetorizar(g))
            Y.append(int(ch))
    return X, Y, puladas


class Modelo:
    def __init__(self, X, Y):
        self.X = np.asarray(X, np.float32)
        self.Y = np.asarray(Y, np.int32)

    @classmethod
    def do_treino(cls, pasta: str = PASTA_TREINO, ignorar: Optional[str] = None,
                  verbose: bool = False) -> "Modelo":
        X, Y = [], []
        for caminho in sorted(Path(pasta).glob("*.png")):
            if ignorar and caminho.name == ignorar:
                continue
            nums = rotulo_do_nome(caminho)
            img = cv2.imread(str(caminho))
            if nums is None or img is None:
                if verbose:
                    print(f"  [ignorado] {caminho.name}")
                continue
            x, y, puladas = extrair_amostras(img, nums)
            if puladas and verbose:
                print(f"  [aviso] {caminho.name}: {puladas} casa(s) não segmentadas")
            X += x
            Y += y
        return cls(X, Y)

    def vazio(self) -> bool:
        return len(self.Y) == 0

    def classificar(self, g: np.ndarray):
        dist = ((self.X - vetorizar(g)) ** 2).sum(axis=1)
        k = int(np.argmin(dist))
        return int(self.Y[k]), float(np.sqrt(dist[k]))

# ─── Reconhecimento ───────────────────────────────────────────────────────────

def reconhecer_img(img: np.ndarray, modelo: Modelo, detalhes: bool = False):
    """
    Retorna a lista com os 13 números (-1 = falha).
    Com detalhes=True retorna também a lista de distâncias (confiança) por casa.
    """
    numeros, dists = [], []
    for d in mapas_das_casas(img):
        gs = separar_digitos(d)
        if not 1 <= len(gs) <= 2:
            numeros.append(-1)
            dists.append(float("inf"))
            continue
        txt, pior = "", 0.0
        for g in gs:
            dig, dist = modelo.classificar(g)
            txt += str(dig)
            pior = max(pior, dist)
        n = int(txt)
        numeros.append(n if n in VALIDOS else -1)
        dists.append(pior)
    return (numeros, dists) if detalhes else numeros


def reconhecer_imagem(caminho: str, modelo: Modelo, debug: bool = False) -> list:
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")
    numeros, dists = reconhecer_img(img, modelo, detalhes=True)
    if debug:
        for i, (n, dist) in enumerate(zip(numeros, dists)):
            alerta = "  <-- DUVIDOSO" if n == -1 or dist > DIST_ALERTA else ""
            print(f"  Casa {i + 1:2d}: {n:3d}   dist={dist:6.2f}{alerta}")
    return numeros

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reconhecedor de números de roleta em imagens.")
    parser.add_argument("imagem", nargs="?", help="Imagem (ou pasta de imagens) a reconhecer")
    parser.add_argument("--debug", action="store_true", help="Mostrar confiança por casa")
    parser.add_argument("--treino", default=PASTA_TREINO,
                        help=f"Pasta de treino (padrão: {PASTA_TREINO})")
    args = parser.parse_args()

    if not args.imagem:
        parser.print_help()
        return

    if not os.path.exists(args.imagem):
        print(f"[ERRO] Arquivo não encontrado: {args.imagem}")
        sys.exit(1)

    modelo = Modelo.do_treino(args.treino)
    if modelo.vazio():
        print(f"[ERRO] Nenhum dígito de treino em '{args.treino}/'.")
        sys.exit(1)

    alvo = Path(args.imagem)
    arquivos = sorted(alvo.glob("*.png")) if alvo.is_dir() else [alvo]
    for arq in arquivos:
        if args.debug or len(arquivos) > 1:
            print(f"{arq.name}:")
        print(reconhecer_imagem(str(arq), modelo, debug=args.debug))


if __name__ == "__main__":
    main()
