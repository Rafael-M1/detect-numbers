import pyautogui
import time
from pynput import mouse

print("Posicione o mouse sobre a região desejada")
print("Clique com o botão esquerdo para capturar a coordenada")

def on_click(x, y, button, pressed):
    if pressed and button == mouse.Button.left:
        print(f"\nCLIQUE CAPTURADO -> X={x} Y={y}")

# Thread que fica mostrando posição em tempo real
def monitor_position():
    while True:
        x, y = pyautogui.position()
        print(f"X={x} Y={y}", end="\r")
        time.sleep(0.1)

# Listener de clique
listener = mouse.Listener(on_click=on_click)
listener.start()

# Loop principal
monitor_position()