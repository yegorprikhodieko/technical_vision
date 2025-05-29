import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.util import random_noise
from math import log10

# Функция для вычисления MSE
def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

# Функция для вычисления PSNR
def psnr(img1, img2):
    mse_val = mse(img1, img2)
    if mse_val == 0:
        return float('inf')
    return 20 * log10(255.0 / np.sqrt(mse_val))

# Функция для вычисления MLEP (Maximum Likelihood Estimation Poisson)
def mlep_poisson(original, noisy):
    original = original.astype(np.float64) + 1e-10  # избегаем деления на ноль
    noisy = noisy.astype(np.float64) + 1e-10
    return np.mean(noisy * np.log(noisy / original) - noisy + original)

# Основная функция программы
def evaluate_noise_metrics(image_files):
    results = {}

    for file in image_files:
        # Загрузка изображения в оттенках серого
        original = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        if original is None:
            print(f"Ошибка загрузки изображения {file}. Проверьте путь и формат файла.")
            continue

        # Добавление Гауссовского шума
        gaussian_noisy = random_noise(original, mode='gaussian', var=0.01)
        gaussian_noisy = np.array(255 * gaussian_noisy, dtype=np.uint8)

        # Добавление Пуассоновского шума
        poisson_noisy = random_noise(original, mode='poisson')
        poisson_noisy = np.array(255 * poisson_noisy, dtype=np.uint8)

        # Расчёт метрик для Гауссовского шума
        mse_gauss = mse(original, gaussian_noisy)
        psnr_gauss = psnr(original, gaussian_noisy)
        ssim_gauss = ssim(original, gaussian_noisy, data_range=255)

        # Расчёт метрик для Пуассоновского шума
        ssim_poisson = ssim(original, poisson_noisy, data_range=255)
        mlep_poisson_val = mlep_poisson(original, poisson_noisy)

        # Сохранение результатов
        results[file] = {
            'Gaussian': {
                'MSE': mse_gauss,
                'PSNR': psnr_gauss,
                'SSIM': ssim_gauss
            },
            'Poisson': {
                'SSIM': ssim_poisson,
                'MLEP': mlep_poisson_val
            }
        }

    return results

# Список изображений для анализа
image_files = ['user_photo_{message.from_user.id}.jpg']

# Выполнение оценки шумности
results = evaluate_noise_metrics(image_files)

# Сохранение результатов
for img, metrics in results.items():
    MSE = metrics['Gaussian']['MSE'] # Метод Гаусса
    PSNR = metrics['Gaussian']['PSNR'] # Метод Гаусса db
    SSIM = metrics['Gaussian']['SSIM'] # Метод Гаусса
    SSIM = metrics['Poisson']['SSIM'] # Метод Пуассона
    MLEP = metrics['Poisson']['MLEP'] # Метод Пуассона
