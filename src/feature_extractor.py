import cv2
import numpy as np
from scipy.stats import skew

class FeatureExtractor:
    @staticmethod
    def extract(original_image, mask):
        features = {}
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None 
            
       
        cnt = max(contours, key=cv2.contourArea)
        
      
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: return None
        circularity = (4 * np.pi * area) / (perimeter**2)
        
        features['area'] = area
        features['perimeter'] = perimeter
        features['circularity'] = circularity

        mask_bool = mask == 255
        
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        hue_pixels = hsv[:, :, 0][mask_bool]        
        sat_pixels = hsv[:, :, 1][mask_bool]        
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

        
        features['texture_std'] = np.std(gray_pixels)
        
        features['max_intensity_diff'] = np.max(gray_pixels) - np.mean(gray_pixels)
        
        
        features['skewness'] = skew(gray_pixels)

        return features, cnt