import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte

class ImagePreprocessor:
    """
    Klasa do wstępnego przetwarzania obrazów, zawierająca metody konwersji do skali szarości,
    poprawy kontrastu i jasności oraz generowania masek na podstawie entropii lub progu Otsu.
    """
    def __init__(self):
        pass

    def to_gray(self, image):
        """
        Konwertuje obraz kolorowy do skali szarości.
        Jeśli obraz jest już w skali szarości, zwraca go bez zmian.
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def enhance_contrast_clahe(self, gray_image):
        """
        Zwiększa kontrast obrazu w skali szarości za pomocą algorytmu CLAHE.
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def enhance_brightness_simple(self, gray_image, beta=30):
        """
        Prosto zwiększa jasność obrazu w skali szarości przez dodanie wartości beta.
        """
        return cv2.convertScaleAbs(gray_image, alpha=1.0, beta=beta)

    def generate_mask_entropy(self, gray_image, disk_size=3):
        """
        Generuje maskę na podstawie mapy entropii lokalnej obrazu w skali szarości.
        Używa progu Otsu po normalizacji mapy entropii.
        """
        img_uint8 = img_as_ubyte(gray_image)
        texture_map = entropy(img_uint8, disk(disk_size))
        texture_map_norm = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(texture_map_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._clean_mask(mask)

    def generate_mask_simple(self, gray_image):
        """
        Generuje maskę binarną obrazu w skali szarości za pomocą progu Otsu.
        """
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def _clean_mask(self, mask):
        """
        Czyści maskę binarną za pomocą operacji morfologicznych (zamykanie i otwieranie).
        """
        kernel_close = np.ones((7, 7), np.uint8) 
        kernel_open = np.ones((5, 5), np.uint8) 
        
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
        return cleaned