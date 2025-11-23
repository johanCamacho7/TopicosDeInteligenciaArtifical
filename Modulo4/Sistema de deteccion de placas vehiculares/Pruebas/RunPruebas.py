from easyocr import read_plate
import pytesseract

# Ruta del ejecutable de Tesseract en tu Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ruta = "1.jpg"  # Tu imagen en la misma carpeta del script

plate = read_plate(ruta, debug=True)

if plate is None:
    print("❌ No se pudo detectar una matrícula válida.")
else:
    print(f"✅ Matrícula detectada: {plate}")
