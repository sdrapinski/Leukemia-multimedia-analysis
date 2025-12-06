import cv2
import numpy as np
from scipy.stats import skew

class FeatureExtractor:
    @staticmethod
    def extract(original_image, mask):
        features = {}
        
        # Znajdź kontury na masce
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None 
            
        # Zakładamy, że największy kontur to komórka
        cnt = max(contours, key=cv2.contourArea)
        
        # Geometria
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: return None
        circularity = (4 * np.pi * area) / (perimeter**2)
        
        features['area'] = area
        features['perimeter'] = perimeter
        features['circularity'] = circularity

        mask_bool = mask == 255
        # Kolor (HSV)
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        hue_pixels = hsv[:, :, 0][mask_bool]        # Odcień
        sat_pixels = hsv[:, :, 1][mask_bool]        # Nasycenie
        val_pixels = hsv[:, :, 2][mask_bool]

        if len(val_pixels) == 0: return None

        mean_color = cv2.mean(hsv, mask=mask)
        
        features['mean_hue'] = np.mean(hue_pixels)
        features['mean_saturation'] = np.mean(sat_pixels)
        features['mean_value'] = np.mean(val_pixels)

        features['std_saturation'] = np.std(sat_pixels)
        features['std_value'] = np.std(val_pixels)
        
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_pixels = gray[mask_bool]

        # 1. Ziarnistość (Tekstura) - Standard Deviation
        features['texture_std'] = np.std(gray_pixels)
        
        # 2. Wykrywanie "jasnych plam" (Anomalie jasności)
        # Różnica między najjaśniejszym punktem a średnią.
        # Duża różnica = prawdopodobnie jasne przebarwienie/wakuola.
        features['max_intensity_diff'] = np.max(gray_pixels) - np.mean(gray_pixels)
        
        # 3. Skośność (Skewness)
        # Mierzy asymetrię rozkładu jasności.
        # Jeśli komórka jest ciemna, ale ma jasną plamę, skośność się zmieni.
        features['skewness'] = skew(gray_pixels)

        return features, cnt