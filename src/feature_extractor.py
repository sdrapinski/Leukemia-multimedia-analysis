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

        mask_bool = mask == 255
        
        hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        lab_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        hue_pixels = hsv[:, :, 0][mask_bool]        
        sat_pixels = hsv[:, :, 1][mask_bool]        
        val_pixels = hsv[:, :, 2][mask_bool]

        a_pixels = a_channel[mask_bool]
        b_pixels = b_channel[mask_bool]

        if len(val_pixels) == 0: return None

        mean_color = cv2.mean(hsv, mask=mask)
        
        features['mean_hue'] = np.mean(hue_pixels)
        features['mean_saturation'] = np.mean(sat_pixels)
        features['mean_value'] = np.mean(val_pixels)

        features['std_saturation'] = np.std(sat_pixels)
        features['std_value'] = np.std(val_pixels)
        features['std_a'] = np.std(a_pixels)
        features['std_b'] = np.std(b_pixels)
        features['color_heterogeneity'] = np.sqrt(features['std_a']**2 + features['std_b']**2)
        
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_pixels = gray[mask_bool]

        
        features['texture_std'] = np.std(gray_pixels)
        
        features['max_intensity_diff'] = np.max(gray_pixels) - np.mean(gray_pixels)
        
        
        features['skewness'] = skew(gray_pixels)

        moments = cv2.moments(cnt)
        huMoments = cv2.HuMoments(moments)
        
        for i in range(0, 7):
            val = -1 * np.sign(huMoments[i][0]) * np.log10(np.abs(huMoments[i][0])) if huMoments[i][0] != 0 else 0
            features[f'hu_moment_{i}'] = val
        
        gray_masked = gray[mask_bool]
        if len(gray_masked) > 0:
            thresh_val, _ = cv2.threshold(gray_masked, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            nucleus_pixels = np.sum(gray_masked < thresh_val)
            total_pixels = len(gray_masked)
            
            features['nc_ratio'] = nucleus_pixels / total_pixels if total_pixels > 0 else 0
        else:
            features['nc_ratio'] = 0

        return features, cnt