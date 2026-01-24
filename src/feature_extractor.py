import cv2
import numpy as np
from scipy.stats import skew

class FeatureExtractor:
    """
    Ekstraktor cech morfologicznych, kolorystycznych i teksturalnych z obrazu na podstawie maski.
    """
    @staticmethod
    def extract(original_image, mask):
        """
        Ekstrahuje cechy z podanego obrazu i maski.
        Zwraca słownik cech oraz największy kontur.
        """
        features = {}
        
        # Znajdź kontury w masce
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None 
            
        # Największy kontur traktowany jako obiekt docelowy
        cnt = max(contours, key=cv2.contourArea)
        
        # Cechy morfologiczne
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: return None
        circularity = (4 * np.pi * area) / (perimeter**2)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        features['area'] = area
        features['perimeter'] = perimeter
        features['circularity'] = circularity
        features['solidity'] = solidity
        features['aspect_ratio'] = aspect_ratio

        # Maska logiczna dla pikseli należących do obiektu
        mask_bool = mask == 255
        
        # Konwersja do przestrzeni HSV
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        # Konwersja do przestrzeni Lab
        lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Ekstrakcja pikseli HSV i Lab z obszaru maski
        hue_pixels = hsv[:, :, 0][mask_bool]        
        sat_pixels = hsv[:, :, 1][mask_bool]        
        val_pixels = hsv[:, :, 2][mask_bool]

        a_pixels = a_channel[mask_bool]
        b_pixels = b_channel[mask_bool]

        if len(val_pixels) == 0: return None

        # Średnie wartości HSV
        mean_color = cv2.mean(hsv, mask=mask)
        
        features['mean_hue'] = np.mean(hue_pixels)
        features['mean_saturation'] = np.mean(sat_pixels)
        features['mean_value'] = np.mean(val_pixels)

        # Odchylenia standardowe i heterogeniczność koloru
        features['std_saturation'] = np.std(sat_pixels)
        features['std_value'] = np.std(val_pixels)
        features['std_a'] = np.std(a_pixels)
        features['std_b'] = np.std(b_pixels)
        features['color_heterogeneity'] = np.sqrt(features['std_a']**2 + features['std_b']**2)
        
        # Cechy teksturalne (jasność w skali szarości)
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_pixels = gray[mask_bool]

        features['texture_std'] = np.std(gray_pixels)
        features['max_intensity_diff'] = np.max(gray_pixels) - np.mean(gray_pixels)
        features['skewness'] = skew(gray_pixels)

        # Hu Moments (cechy niezmiennicze względem transformacji geometrycznych)
        moments = cv2.moments(cnt)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            val = -1 * np.sign(huMoments[i][0]) * np.log10(np.abs(huMoments[i][0])) if huMoments[i][0] != 0 else 0
            features[f'hu_moment_{i}'] = val
        
        # Szacowanie stosunku jądro/cytoplazma na podstawie binaryzacji Otsu wewnątrz maski
        gray_masked = gray[mask_bool]
        if len(gray_masked) > 0:
            thresh_val, _ = cv2.threshold(gray_masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            nucleus_pixels = np.sum(gray_masked < thresh_val)
            total_pixels = len(gray_masked)
            features['nc_ratio'] = nucleus_pixels / total_pixels if total_pixels > 0 else 0
        else:
            features['nc_ratio'] = 0

        return features, cnt