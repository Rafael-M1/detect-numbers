@echo off
echo ============================================
echo  Instalacao - Reconhecedor de Roleta
echo ============================================
echo.

echo [1/3] Instalando dependencias Python...
pip install opencv-python numpy pytesseract pillow

echo.
echo [2/3] Verificando Tesseract...
where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ATENCAO: Tesseract NAO encontrado no PATH.
    echo  Baixe e instale em:
    echo  https://github.com/UB-Mannheim/tesseract/wiki
    echo.
    echo  Durante a instalacao, marque a opcao de adicionar ao PATH.
    echo  Versao recomendada: tesseract-ocr-w64-setup-5.x.x.exe
    echo.
    echo  Sem Tesseract, o sistema usara apenas Template Matching.
) else (
    echo  Tesseract encontrado: OK
)

echo.
echo [3/3] Criando pastas necessarias...
if not exist "treino" mkdir treino
if not exist "templates" mkdir templates

echo.
echo ============================================
echo  Instalacao concluida!
echo.
echo  PROXIMOS PASSOS:
echo  1. Coloque suas imagens de treino em: treino\
echo     Nome das imagens: 1-36-4-12-3-0-17-22-5-11-8-25-14.png
echo     (13 numeros separados por traco)
echo.
echo  2. Execute para indexar:
echo     python reconhecer.py --indexar
echo.
echo  3. Para reconhecer uma nova imagem:
echo     python reconhecer.py sua_imagem.png
echo.
echo  4. Para ver detalhes do reconhecimento:
echo     python reconhecer.py sua_imagem.png --debug
echo ============================================
pause