import cv2
import os
import glob

class ImageLoader:
    def __init__(self, input_dir):
        self.input_dir = input_dir

    def load_images(self, extensions=("*.bmp", "*.jpg", "*.png")):
        """Generator zwracający ścieżkę, nazwę pliku i wczytany obraz."""
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.input_dir, ext)))
        
        for filepath in files:
            filename = os.path.basename(filepath)
            img = cv2.imread(filepath)
            if img is not None:
                yield filename, img

    @staticmethod
    def save_image(output_dir, filename, image, suffix=""):
        """Zapisuje obraz z odpowiednim przyrostkiem."""
        os.makedirs(output_dir, exist_ok=True)
        name, ext = os.path.splitext(filename)
        new_name = f"{name}_{suffix}{ext}"
        cv2.imwrite(os.path.join(output_dir, new_name), image)