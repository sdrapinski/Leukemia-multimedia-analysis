import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte

class ImagePreprocessor:
    def __init__(self):
        pass

    def to_gray(self, image):
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def enhance_contrast_clahe(self, gray_image):
        """Metoda z utils.py (lepsza do tekstur)"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray_image)

    def enhance_brightness_simple(self, gray_image, beta=30):
        """Metoda z ImgProcessing.ipynb (prostsza)"""
        return cv2.convertScaleAbs(gray_image, alpha=1.0, beta=beta)

    def generate_mask_entropy(self, gray_image, disk_size=3):
        """Metoda z utils.py - detekcja na podstawie tekstury"""
        img_uint8 = img_as_ubyte(gray_image)
        texture_map = entropy(img_uint8, disk(disk_size))
        texture_map_norm = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, mask = cv2.threshold(texture_map_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._clean_mask(mask)

    def generate_mask_simple(self, gray_image):
        """Metoda z ImgProcessing.ipynb - detekcja na podstawie jasności"""
        _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def _clean_mask(self, mask):
        """Czyszczenie dziur i szumów"""
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        return cleaned