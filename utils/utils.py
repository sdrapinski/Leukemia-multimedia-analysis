import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte

def preprocess_image(image):
    """
    KROK 1: Wstępne przetwarzanie i poprawa kontrastu (CLAHE).
    Zamienia obraz na odcienie szarości i wyrównuje histogram, 
    aby uwydatnić detale wewnątrz komórki przed analizą tekstury.
    """
    # Konwersja na szarość
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # ClipLimit zapobiega nadmiernemu wzmocnieniu szumu na czarnym tle
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    return enhanced_gray

def calculate_texture_map(gray_image, disk_size=3):
    """
    KROK 2: Obliczenie mapy tekstury (Entropia).
    Miejsca gładkie (czarne tło) będą miały niską wartość.
    Miejsca złożone (wnętrze komórki) będą miały wysoką wartość.
    """
    # Entropia wymaga formatu uint8
    img_uint8 = img_as_ubyte(gray_image)
    
    # Obliczenie entropii lokalnej
    # Disk(3) oznacza, że patrzymy na otoczenie o promieniu 3 pikseli
    texture_map = entropy(img_uint8, disk(disk_size))
    
    # Normalizacja wyniku do zakresu 0-255 (żeby można było go progować jak obraz)
    texture_map_norm = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX)
    return texture_map_norm.astype(np.uint8)

def generate_binary_mask(texture_map):
    """
    KROK 3: Binaryzacja na podstawie tekstury.
    Oddzielamy "coś co ma strukturę" od "pustego tła".
    """
    # Używamy Otsu do automatycznego znalezienia progu
    _, binary_mask = cv2.threshold(texture_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask

def clean_mask(binary_mask):
    """
    KROK 4: Operacje morfologiczne.
    Wygładzamy krawędzie maski i wypełniamy ewentualne dziury w środku komórki.
    """
    kernel = np.ones((5,5), np.uint8)
    
    # Zamknięcie (Closing) - wypełnia małe czarne dziury wewnątrz białego obszaru
    cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Otwarcie (Opening) - usuwa małe białe kropki z tła (szum)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def extract_features(original_image, mask):
    """
    KROK 5: Ekstrakcja cech numerycznych.
    Mierzymy kształt z maski oraz kolor z oryginalnego obrazu (tam gdzie maska > 0).
    """
    features = {}
    
    # --- Analiza Kształtu ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None # Nie znaleziono komórki
        
    # Zakładamy, że największy kontur to komórka
    cnt = max(contours, key=cv2.contourArea)
    
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    if perimeter == 0:
        circularity = 0
    else:
        circularity = (4 * np.pi * area) / (perimeter**2)
        
    features['area'] = area
    features['perimeter'] = perimeter
    features['circularity'] = circularity
    
    # --- Analiza Koloru (tylko wewnątrz maski) ---
    # Konwersja na HSV dla lepszej analizy medycznej
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # cv2.mean oblicza średnią tylko tam, gdzie maska jest różna od zera
    mean_color = cv2.mean(hsv, mask=mask)
    
    features['mean_hue'] = mean_color[0]
    features['mean_saturation'] = mean_color[1]
    features['mean_value'] = mean_color[2]
    
    # --- Analiza Tekstury (Wariancja jasności wewnątrz komórki) ---
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # Wybieramy piksele należące do komórki
    cell_pixels = gray[mask == 255]
    if len(cell_pixels) > 0:
        features['texture_std'] = np.std(cell_pixels) # Odchylenie standardowe jasności
    else:
        features['texture_std'] = 0

    return features