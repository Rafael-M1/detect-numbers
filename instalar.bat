@echo off
echo ============================================
echo  Instalacao - Reconhecedor de Roleta
echo ============================================
echo.

echo [1/2] Instalando dependencias Python...
pip install -r requirements.txt

echo.
echo [2/2] Criando pastas necessarias...
if not exist "treino" mkdir treino
if not exist "novas" mkdir novas

echo.
echo ============================================
echo  Instalacao concluida!
echo.
echo  PROXIMOS PASSOS:
echo  1. Coloque suas imagens de treino em: treino\
echo     Nome das imagens: 1-36-4-12-3-0-17-22-5-11-8-25-14.png
echo     (13 numeros separados por traco)
echo.
echo  2. Para reconhecer uma nova imagem:
echo     python reconhecer.py sua_imagem.png
echo.
echo  3. Para ver a confianca de cada casa:
echo     python reconhecer.py sua_imagem.png --debug
echo.
echo  4. Para medir a precisao sobre o treino:
echo     python reconhecer_dados_treino.py
echo ============================================
pause
