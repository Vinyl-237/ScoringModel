# src/timer.py
import time
import threading
import sys
from contextlib import contextmanager

@contextmanager
def timer(message: str):
    spinner_chars = ['|', '/', '-', '\\']
    stop_spinner = False

    def spinner():
        i = 0
        while not stop_spinner:
            sys.stdout.write(f'\r[{spinner_chars[i % len(spinner_chars)]}] {message}...')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

    # Démarre le spinner dans un thread séparé
    thread = threading.Thread(target=spinner, daemon=True)
    thread.start()

    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        stop_spinner = True
        thread.join()  # Attend que le thread se termine
        sys.stdout.write('\r' + ' ' * (len(message) + 10) + '\r')  # Efface la ligne du spinner
        print(f"[--] {message} terminé en {end - start:.2f} sec [--]\n")
