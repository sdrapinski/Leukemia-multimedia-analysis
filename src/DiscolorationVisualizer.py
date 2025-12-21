import cv2
import numpy as np

class DiscolorationVisualizer:
    """Klasa odpowiedzialna za wizualizację anomalii kolorystycznych."""
    
    @staticmethod
    def create_heatmap(original_image, mask):
        """
        Tworzy mapę cieplną pokazującą odchylenia od średniego koloru komórki.
        Miejsca 'gorące' (czerwone/żółte) to potencjalne przebarwienia.
        """
       
        lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2Lab)
        
        mean_color = cv2.mean(lab, mask=mask)[:3] # (L, a, b)
        
        mean_img = np.zeros_like(lab, dtype=np.uint8)
        mean_img[:] = mean_color
        
        diff = cv2.absdiff(lab, mean_img)
        diff_l, diff_a, diff_b = cv2.split(diff)
        
        color_diff = cv2.addWeighted(diff_a, 0.5, diff_b, 0.5, 0)
        
        color_diff = cv2.bitwise_and(color_diff, color_diff, mask=mask)
        
        norm_diff = cv2.normalize(color_diff, None, 0, 255, cv2.NORM_MINMAX)
        
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
        
        heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=mask)
        
        return heatmap_masked