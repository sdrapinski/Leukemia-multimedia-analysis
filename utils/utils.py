import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte

def preprocess_image(image):
    """
    Konwersja obrazu do skali szarości oraz poprawa kontrastu metodą CLAHE.
    """
    # Jeśli obraz jest kolorowy, konwertuj do skali szarości
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Zastosowanie CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    return enhanced_gray

def calculate_texture_map(gray_image, disk_size=3):
    """
    Wyznaczenie mapy entropii lokalnej dla obrazu w skali szarości.
    Używany jest promień dysku określający sąsiedztwo.
    """
    img_uint8 = img_as_ubyte(gray_image)
    texture_map = entropy(img_uint8, disk(disk_size))
    # Normalizacja do zakresu 0-255
    texture_map_norm = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX)
    return texture_map_norm.astype(np.uint8)

def generate_binary_mask(texture_map):
    """
    Binaryzacja mapy tekstury metodą Otsu.
    """
    _, binary_mask = cv2.threshold(texture_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def clean_mask(binary_mask):
    """
    Morfologiczne czyszczenie maski: zamknięcie i otwarcie (kernel 5x5).
    """
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned

def extract_features(original_image, mask):
    """
    Ekstrakcja cech morfologicznych, kolorystycznych i teksturalnych z obrazu na podstawie maski.
    """
    features = {}
    # Znajdź kontury w masce
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Największy kontur traktowany jako obiekt docelowy
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter != 0 else 0
    features['area'] = area
    features['perimeter'] = perimeter
    features['circularity'] = circularity

    # Średnie wartości HSV wewnątrz maski
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    mean_color = cv2.mean(hsv, mask=mask)
    features['mean_hue'] = mean_color[0]
    features['mean_saturation'] = mean_color[1]
    features['mean_value'] = mean_color[2]

    # Odchylenie standardowe jasności (tekstura) wewnątrz maski
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    cell_pixels = gray[mask == 255]
    if len(cell_pixels) > 0:
        features['texture_std'] = np.std(cell_pixels) # Odchylenie standardowe jasności
    else:
        features['texture_std'] = 0

    return features