import cv2
import numpy as np
from PIL import Image

def get_resolution(image_path):
    # Получает разрешение изображения.
    try:
        img = Image.open(image_path)
        width, height = img.size
        return width, height
    except FileNotFoundError:
        return None, None

def get_brightness_contrast(image_path):
    # Оценивает яркость и контраст.
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        brightness = np.mean(img)
        contrast = np.std(img)
        return brightness, contrast
    except FileNotFoundError:
        return None, None

def calculate_noise(image_path):
    # Рассчитывает шум (стандартное отклонение).
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        noise = np.std(img)
        return noise
    except FileNotFoundError:
        return None

def calculate_laplacian_variance(image_path):
    # Рассчитывает четкость (дисперсия лапласиана).
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = np.var(laplacian)
        return variance
    except FileNotFoundError:
        return None