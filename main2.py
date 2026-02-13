import time
import keyboard

print("Bot iniciando em 5 segundos...")
time.sleep(5)
print("Executando: F1 → F12 | Pressione ESC para parar.")

try:
    while True:
        # Se ESC for pressionado, interrompe o loop
        if keyboard.is_pressed("esc"):
            print("ESC pressionado. Encerrando o bot.")
            break

        # Pressiona F1 até F12
        for i in range(1, 13):
            if keyboard.is_pressed("esc"):
                print("ESC pressionado. Encerrando o bot.")
                raise KeyboardInterrupt

            tecla = f"f{i}"
            keyboard.press_and_release(tecla)
            print(f"Tecla {tecla.upper()} pressionada")
            time.sleep(0.1)  # pequeno delay entre teclas

except KeyboardInterrupt:
    pass

print("Bot finalizado.")
