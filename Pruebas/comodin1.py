import cv2
import numpy as np
import easyocr

"""
Script local para leer una placa usando: modelo pre entrenado de EasyOCR + OpenCV.

1. OpenCV → detección aproximada de la placa (ROI por contornos)
2. EasyOCR → OCR sobre la región detectada (o la imagen completa si no detecta nada)

La imagen se llama '1.jpg' y debe estar en el mismo directorio.
"""

# ---------------------- CONFIG ----------------------

# Idiomas de EasyOCR
EASYOCR_LANGS = ["en"]  # puedes probar ['en', 'es']

# Inicializamos EasyOCR una sola vez (tarda un poco al inicio)
reader = easyocr.Reader(EASYOCR_LANGS)


# ---------------------- FUNCIONES ----------------------

def find_plate_roi(image: np.ndarray):
    """
    Intenta detectar la placa mediante contornos y devolver un recorte (ROI).
    Si no detecta nada razonable, devuelve None.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    edges = cv2.Canny(gray, 30, 200)

    contours_info = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # buscamos un rectángulo
            x, y, w, h = cv2.boundingRect(approx)
            if w == 0 or h == 0:
                continue

            aspect_ratio = w / float(h)

            # Heurísticas simples para placas
            if 1.8 < aspect_ratio < 6.5 and w > 80 and h > 25:
                roi = image[y:y + h, x:x + w]
                return roi

    return None


def clean_plate_text(text: str) -> str:
    """
    Normaliza texto de placa:
    - Mayúsculas
    - Quita espacios y guiones
    - Solo letras y números
    """
    text = text.upper().replace(" ", "").replace("-", "")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    filtered = "".join(ch for ch in text if ch in allowed)
    return filtered


def is_plate_like(text: str) -> bool:
    """
    Heurística: texto que se parece a una placa.
    - Longitud entre 5 y 8 caracteres (ajusta si quieres)
    - Debe tener al menos una letra y al menos un dígito
    """
    if not (5 <= len(text) <= 8):
        return False

    has_letter = any(c.isalpha() for c in text)
    has_digit = any(c.isdigit() for c in text)

    return has_letter and has_digit


def choose_best_plate(results):
    """
    results = [ [bbox, text, conf], ... ] de easyocr.readtext

    1. Primero buscamos candidatos que se parezcan a placas (mezcla letras+números).
    2. De esos, elegimos el de mayor confianza.
    3. Si no hay ninguno, como fallback tomamos el texto de mayor confianza general.
    """
    best_plate = ""
    best_conf = 0.0

    # 1) candidatos tipo placa
    for bbox, text, conf in results:
        cleaned = clean_plate_text(text)
        if is_plate_like(cleaned) and conf > best_conf:
            best_conf = conf
            best_plate = cleaned

    if best_plate:
        return best_plate

    # 2) fallback: mejor OCR general, pero normalizado
    if results:
        _, text, conf = max(results, key=lambda r: r[2])
        return clean_plate_text(text)

    return ""


def read_plate(image: np.ndarray) -> str:
    """
    1. Detecta ROI de placa (si puede).
    2. Lee texto con EasyOCR.
    3. Selecciona el texto que más se parece a una placa.
    """
    roi = find_plate_roi(image)
    target = roi if roi is not None else image

    results = reader.readtext(target)  # [ [bbox, text, conf], ... ]
    plate_text = choose_best_plate(results)
    return plate_text


# ---------------------- MAIN ----------------------

def main():
    filename = "../7.jpg"

    print(f"[INFO] Leyendo imagen local '{filename}' ...")
    image = cv2.imread(filename)

    if image is None:
        print("[ERROR] No se encontró la imagen '1.jpg' en el root.")
        return

    plate = read_plate(image)

    print("----------------------------------------")
    print(f"[RESULTADO] Placa detectada (local): {plate if plate else '(sin lectura)'}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()

