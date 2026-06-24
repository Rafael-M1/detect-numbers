"""
Reconhecedor de Números de Roleta
==================================
Lê uma imagem com 13 números em linha e retorna um array com os valores.
Usa as imagens da pasta 'treino/' como base de templates.

Uso:
    python reconhecer.py imagem.png
    python reconhecer.py imagem.png --debug
    python reconhecer.py --indexar
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# ─── Configurações ────────────────────────────────────────────────────────────

PASTA_TREINO    = "treino"
PASTA_TEMPLATES = "templates"
NUM_REGIOES     = 13
VALIDOS         = set(range(0, 37))

# ─── Segmentação ──────────────────────────────────────────────────────────────

def segmentar_regioes(img: np.ndarray, n: int = NUM_REGIOES) -> list:
    h, w = img.shape[:2]
    largura = w // n
    regioes = []
    for i in range(n):
        x1 = i * largura
        x2 = x1 + largura if i < n - 1 else w
        regioes.append(img[:, x1:x2])
    return regioes

# ─── Pré-processamento ────────────────────────────────────────────────────────

def detectar_tipo_fundo(regiao_bgr: np.ndarray) -> str:
    gray = cv2.cvtColor(regiao_bgr, cv2.COLOR_BGR2GRAY)
    mediana = float(np.median(gray))
    return "escuro" if mediana < 80 else "colorido"

def binarizar(regiao_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(regiao_bgr, cv2.COLOR_BGR2GRAY)
    mediana = float(np.median(gray))

    if mediana < 30:
        _, bw = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    else:
        diff = np.abs(gray.astype(np.float32) - mediana)
        bw = (diff > 50).astype(np.uint8) * 255

    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return bw

# ─── Análise de componentes ───────────────────────────────────────────────────

def calcular_ratio(bw: np.ndarray):
    """Retorna (ratio largura/altura, n_componentes, stats)."""
    _, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    n_comp = int(labels.max())
    if n_comp == 0:
        return 0.0, 0, []

    comp_stats = [stats[j] for j in range(1, n_comp + 1)]
    xs  = [s[0] for s in comp_stats]
    x2s = [s[0] + s[2] for s in comp_stats]
    total_w = max(x2s) - min(xs)
    max_h   = max(s[3] for s in comp_stats)
    ratio   = total_w / max_h if max_h > 0 else 0.0
    return ratio, n_comp, comp_stats

# ─── OCR via Tesseract ────────────────────────────────────────────────────────

def _tesseract_disponivel() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

TESSERACT_OK = _tesseract_disponivel()

def _ocr_bw(bw: np.ndarray) -> Optional[int]:
    """OCR em uma imagem binarizada. Retorna int ou None."""
    if not TESSERACT_OK:
        return None
    try:
        import pytesseract
        h, w = bw.shape[:2]
        escala = max(4, 120 // max(h, 1))
        ampliado = cv2.resize(bw, None, fx=escala, fy=escala,
                              interpolation=cv2.INTER_CUBIC)
        margem = 10
        com_margem = cv2.copyMakeBorder(ampliado, margem, margem, margem, margem,
                                        cv2.BORDER_CONSTANT, value=0)
        config = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
        texto = pytesseract.image_to_string(com_margem, config=config).strip()
        num = int(texto)
        if num in VALIDOS:
            return num
    except Exception:
        pass
    return None

def ocr_regiao(bw: np.ndarray) -> Optional[int]:
    """
    OCR com detecção de 'dígito estreito' para evitar confundir
    números de 1 dígito estreito (1) com 2 dígitos (11).

    Lógica:
    - ratio < 0.9 → dígito estreito (provavelmente 1 ou 7) → OCR direto
    - ratio >= 1.0 → dois dígitos → OCR direto (Tesseract lida bem com 2 dígitos
      quando há pixels suficientes). Se retornar 1 dígito, retorna None
      para forçar o fallback de Template Matching, que tem exemplos do 11.
    """
    ratio, n_comp, comp_stats = calcular_ratio(bw)
    resultado = _ocr_bw(bw)

    # Se ratio indica 2 dígitos mas OCR retornou apenas 1 dígito (< 10)
    # → não confiar no OCR, deixar para template matching
    if ratio >= 1.0 and resultado is not None and resultado < 10:
        return None

    return resultado

# ─── Template Matching ────────────────────────────────────────────────────────

class BancoTemplates:
    def __init__(self, pasta: str = PASTA_TEMPLATES):
        self.pasta = Path(pasta)
        self.banco: dict = {}
        self._carregar()

    def _carregar(self):
        if not self.pasta.exists():
            return
        for dir_tipo in self.pasta.iterdir():
            if not dir_tipo.is_dir():
                continue
            tipo = dir_tipo.name
            self.banco[tipo] = {}
            for dir_num in dir_tipo.iterdir():
                if not dir_num.is_dir():
                    continue
                try:
                    num = int(dir_num.name)
                except ValueError:
                    continue
                imgs = []
                for arq in dir_num.glob("*.png"):
                    img = cv2.imread(str(arq), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        imgs.append(img)
                if imgs:
                    self.banco[tipo][num] = imgs

    def vazio(self) -> bool:
        return len(self.banco) == 0

    def reconhecer(self, bw: np.ndarray, tipo_fundo: str) -> Optional[int]:
        candidatos = {}
        if tipo_fundo in self.banco:
            candidatos.update(self.banco[tipo_fundo])
        if not candidatos:
            for tipo in self.banco:
                candidatos.update(self.banco[tipo])
        if not candidatos:
            return None

        melhor_num   = None
        melhor_score = float("inf")
        h_ref, w_ref = bw.shape[:2]

        for num, lista_imgs in candidatos.items():
            for tmpl in lista_imgs:
                tmpl_r = cv2.resize(tmpl, (w_ref, h_ref), interpolation=cv2.INTER_AREA)
                diff   = cv2.absdiff(bw.astype(np.float32), tmpl_r.astype(np.float32))
                mse    = float(np.mean(diff ** 2))
                if mse < melhor_score:
                    melhor_score = mse
                    melhor_num   = num

        return melhor_num

# ─── Indexação dos templates de treino ───────────────────────────────────────

def indexar_treino(pasta_treino: str = PASTA_TREINO,
                   pasta_templates: str = PASTA_TEMPLATES):
    pasta = Path(pasta_treino)
    if not pasta.exists():
        print(f"[ERRO] Pasta '{pasta_treino}' não encontrada.")
        return

    imagens = list(pasta.glob("*.png")) + list(pasta.glob("*.jpg"))
    if not imagens:
        print(f"[AVISO] Nenhuma imagem encontrada em '{pasta_treino}'.")
        return

    print(f"Indexando {len(imagens)} imagem(ns) de treino…")
    contadores: dict = {}

    for caminho in imagens:
        stem   = caminho.stem
        partes = stem.split("-")
        try:
            numeros = [int(p) for p in partes]
        except ValueError:
            print(f"  [ignorado] nome inválido: {caminho.name}")
            continue

        if len(numeros) != NUM_REGIOES:
            print(f"  [ignorado] {caminho.name} → {len(numeros)} números (esperado {NUM_REGIOES})")
            continue

        img = cv2.imread(str(caminho))
        if img is None:
            print(f"  [ignorado] não foi possível ler: {caminho.name}")
            continue

        regioes = segmentar_regioes(img)

        for regiao, num in zip(regioes, numeros):
            if num not in VALIDOS:
                continue
            tipo = detectar_tipo_fundo(regiao)
            bw   = binarizar(regiao)

            dest = Path(pasta_templates) / tipo / str(num)
            dest.mkdir(parents=True, exist_ok=True)

            contadores.setdefault(tipo, {}).setdefault(num, 0)
            contadores[tipo][num] += 1
            idx = contadores[tipo][num]

            cv2.imwrite(str(dest / f"img_{idx:04d}.png"), bw)

    total = sum(v for c in contadores.values() for v in c.values())
    print(f"Indexação concluída: {total} templates salvos em '{pasta_templates}/'")
    for tipo, nums in sorted(contadores.items()):
        print(f"  Tipo '{tipo}': {sum(nums.values())} templates — números: {sorted(nums.keys())}")

# ─── Reconhecimento principal ─────────────────────────────────────────────────

def reconhecer_imagem(caminho: str,
                      banco: BancoTemplates,
                      debug: bool = False) -> list:
    img = cv2.imread(caminho)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {caminho}")

    regioes   = segmentar_regioes(img)
    resultado = []

    for i, regiao in enumerate(regioes):
        tipo = detectar_tipo_fundo(regiao)
        bw   = binarizar(regiao)

        num    = ocr_regiao(bw)
        metodo = "OCR"

        if num is None:
            num    = banco.reconhecer(bw, tipo)
            metodo = "Template"

        if num is None:
            num    = -1
            metodo = "FALHA"

        resultado.append(num)

        if debug:
            ratio, n_comp, _ = calcular_ratio(bw)
            print(f"  Região {i+1:2d} | fundo={tipo:8s} | ratio={ratio:.2f} | "
                  f"comp={n_comp} | num={str(num):3s} | via {metodo}")

    if debug:
        _salvar_debug(img, resultado, "debug_resultado.png")
        print("  → debug_resultado.png salvo")

    return resultado

def _salvar_debug(img: np.ndarray, numeros: list, saida: str):
    escala = 6
    out    = cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)
    h, w   = img.shape[:2]
    larg   = w // NUM_REGIOES

    for i, num in enumerate(numeros):
        x1      = i * larg * escala
        x2      = (x1 + larg * escala) if i < NUM_REGIOES - 1 else out.shape[1]
        cor_box = (0, 255, 0) if num != -1 else (0, 0, 255)
        cv2.rectangle(out, (x1, 0), (x2 - 1, out.shape[0] - 1), cor_box, 1)
        cv2.putText(out, str(num), (x1 + 2, out.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imwrite(saida, out)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reconhecedor de números de roleta em imagens."
    )
    parser.add_argument("imagem", nargs="?",
                        help="Caminho da imagem a reconhecer")
    parser.add_argument("--indexar", action="store_true",
                        help="Reindexar templates a partir da pasta de treino")
    parser.add_argument("--debug", action="store_true",
                        help="Mostrar detalhes e salvar imagem anotada")
    parser.add_argument("--treino", default=PASTA_TREINO,
                        help=f"Pasta de treino (padrão: {PASTA_TREINO})")
    parser.add_argument("--templates", default=PASTA_TEMPLATES,
                        help=f"Pasta de templates (padrão: {PASTA_TEMPLATES})")
    args = parser.parse_args()

    if args.indexar:
        indexar_treino(args.treino, args.templates)
        return

    if not args.imagem:
        parser.print_help()
        return

    if not os.path.exists(args.imagem):
        print(f"[ERRO] Arquivo não encontrado: {args.imagem}")
        sys.exit(1)

    if not TESSERACT_OK:
        print("[AVISO] Tesseract não encontrado — usando apenas Template Matching.")

    banco = BancoTemplates(args.templates)
    if banco.vazio():
        print("[AVISO] Banco de templates vazio. Execute primeiro:")
        print("        python reconhecer.py --indexar")

    if args.debug:
        print(f"Processando: {args.imagem}")

    numeros = reconhecer_imagem(args.imagem, banco, debug=args.debug)
    print(numeros)
    return numeros


if __name__ == "__main__":
    main()