import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def detect_and_remove_noise(image_path, threshold_ratio=2.0, blur_kernel_size=3):
    """
    Обнаруживает шумы в изображении и применяет размытие ко всему изображению при их наличии.

    Args:
        image_path: Путь к изображению
        threshold_ratio: Пороговое отношение яркости пикселя к среднему значению соседей
        blur_kernel_size: Начальный размер ядра размытия

    Returns:
        Изображение с устраненным шумом
    """
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None

    # Размеры изображения
    height, width = img.shape

    # Счетчик шумных пикселей
    noise_count = 0

    # Проходим по всем пикселям (кроме границы)
    for y in range(1, height-1):
        for x in range(1, width-1):
            # Получаем значение текущего пикселя
            pixel_value = float(img[y, x])

            # Получаем значения 8 соседних пикселей
            neighbors = [
                img[y-1, x-1], img[y-1, x], img[y-1, x+1],
                img[y, x-1],                img[y, x+1],
                img[y+1, x-1], img[y+1, x], img[y+1, x+1]
            ]

            # Вычисляем среднее значение соседей
            neighbors_mean = np.mean(neighbors)

            # Если значение пикселя значительно выше среднего соседей
            if neighbors_mean > 0 and pixel_value / neighbors_mean > threshold_ratio:
                noise_count += 1

    print(f"Обнаружено {noise_count} шумных пикселей в {image_path}")

    # Если шумных пикселей нет, возвращаем исходное изображение
    if noise_count == 0:
        return img

    # Применяем размытие ко всему изображению
    current_kernel_size = blur_kernel_size
    max_iterations = 10  # Максимальное количество итераций

    # Создаем копию для результата
    result_img = img.copy()

    for iteration in range(max_iterations):
        # Применяем размытие по Гауссу ко всему изображению
        result_img = cv2.GaussianBlur(result_img, (current_kernel_size, current_kernel_size), 0)

        # Проверяем, устранен ли шум
        noise_remaining = 0

        for y in range(1, height-1):
            for x in range(1, width-1):
                pixel_value = float(result_img[y, x])

                neighbors = [
                    result_img[y-1, x-1], result_img[y-1, x], result_img[y-1, x+1],
                    result_img[y, x-1],                      result_img[y, x+1],
                    result_img[y+1, x-1], result_img[y+1, x], result_img[y+1, x+1]
                ]

                neighbors_mean = np.mean(neighbors)

                if neighbors_mean > 0 and pixel_value / neighbors_mean > threshold_ratio:
                    noise_remaining += 1

        print(f"Итерация {iteration+1}: осталось {noise_remaining} шумных пикселей")

        # Если шум устранен или значительно уменьшен, прекращаем
        if noise_remaining == 0 or noise_remaining < noise_count * 0.1:
            break

        # Увеличиваем размер ядра размытия
        current_kernel_size += 2
        if current_kernel_size > 15:  # Ограничиваем максимальный размер ядра
            current_kernel_size = 15

    return result_img

def calculate_metrics(original, processed):
    """Вычисляет метрики для оценки качества обработки"""
    # Проверяем размеры изображений и при необходимости изменяем размер
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))

    # MSE
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)

    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))

    # SSIM (упрощенная версия)
    # Константы для стабильности
    K1 = 0.01
    K2 = 0.03
    L = 255  # Динамический диапазон

    # Вычисляем среднее
    mu_x = np.mean(original)
    mu_y = np.mean(processed)

    # Вычисляем дисперсию и ковариацию
    sigma_x = np.std(original)
    sigma_y = np.std(processed)
    sigma_xy = np.mean((original - mu_x) * (processed - mu_y))

    # Вычисляем SSIM
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2)

    ssim = numerator / denominator

    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }

def calculate_mlep(original, noisy):
    """Вычисляет MLEP (Maximum Likelihood Estimation Performance)"""
    # Преобразуем изображения в одномерные массивы
    orig_flat = original.flatten() + 1e-10  # Добавляем малое значение, чтобы избежать log(0)
    noisy_flat = noisy.flatten() + 1e-10

    # Вычисляем логарифмическое правдоподобие для модели Пуассона
    log_likelihood = np.sum(noisy_flat * np.log(orig_flat) - orig_flat)

    # Нормализуем по количеству пикселей
    mlep = log_likelihood / len(orig_flat)
    return mlep

def save_processed_image(image, original_path, suffix="_processed_full"):
    """Сохраняет обработанное изображение с новым именем"""
    # Получаем имя файла без расширения
    filename, ext = os.path.splitext(original_path)
    # Создаем новое имя файла
    new_filename = f"{filename}{suffix}{ext}"
    # Сохраняем изображение
    cv2.imwrite(new_filename, image)
    print(f"Изображение сохранено как {new_filename}")
    return new_filename

def evaluate_noise_metrics(original_path, noisy_path):
    """Оценивает шумность изображения с использованием различных метрик"""
    # Загрузка изображений
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)

    # Проверка, что изображения имеют одинаковый размер
    if original.shape != noisy.shape:
        # Изменяем размер шумного изображения, чтобы соответствовать оригиналу
        noisy = cv2.resize(noisy, (original.shape[1], original.shape[0]))

    # Метрики для шума Гаусса (предполагаем, что шум Гауссовский)
    mse_value = calculate_metrics(original, noisy)['MSE']
    psnr_value = calculate_metrics(original, noisy)['PSNR']
    ssim_value = calculate_metrics(original, noisy)['SSIM']

    # Метрики для шума Пуассона
    mlep_value = calculate_mlep(original, noisy)

    return {
        'Gaussian Noise': {
            'MSE': mse_value,
            'PSNR': psnr_value,
            'SSIM': ssim_value
        },
        'Poisson Noise': {
            'SSIM': ssim_value,  # Используем тот же SSIM
            'MLEP': mlep_value
        }
    }

def compare_images(original_path, processed_paths):
    """Сравнивает оригинальное изображение с обработанными"""
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

    results = {}
    for path in processed_paths:
        processed = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        metrics = calculate_metrics(original, processed)
        mlep = calculate_mlep(original, processed)

        results[path] = {
            'MSE': metrics['MSE'],
            'PSNR': metrics['PSNR'],
            'SSIM': metrics['SSIM'],
            'MLEP': mlep
        }

    return results

def visualize_results(original_path, noisy_paths, processed_paths):
    """Визуализирует результаты обработки"""
    # Загружаем изображения
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

    noisy_images = []
    for path in noisy_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        noisy_images.append(img)

    processed_images = []
    for path in processed_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        processed_images.append(img)

    # Создаем фигуру для визуализации
    fig, axes = plt.subplots(1, 1 + len(noisy_images) + len(processed_images), figsize=(15, 5))

    # Отображаем оригинальное изображение
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Оригинал')
    axes[0].axis('off')

    # Отображаем шумные изображения
    for i, img in enumerate(noisy_images):
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f'Шумное {i+1}')
        axes[i+1].axis('off')

    # Отображаем обработанные изображения
    for i, img in enumerate(processed_images):
        axes[i+1+len(noisy_images)].imshow(img, cmap='gray')
        axes[i+1+len(noisy_images)].set_title(f'Обработанное {i+1}')
        axes[i+1+len(noisy_images)].axis('off')

    plt.tight_layout()
    plt.savefig('comparison.png')
    print("Визуализация сохранена как comparison.png")

# Основная функция программы
def main():
    # Обрабатываем все изображения
    image_paths = ['image.png', 'image2.png', 'image3.png']
    processed_images = {}
    metrics_results = {}
    saved_filenames = {}

    # Получаем размеры всех изображений
    image_shapes = {}
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_shapes[path] = img.shape
        print(f"Размер изображения {path}: {img.shape}")

    for path in image_paths:
        # Обработка изображения
        processed = detect_and_remove_noise(path, threshold_ratio=2.0)
        processed_images[path] = processed

        # Сохранение обработанного изображения
        saved_filename = save_processed_image(processed, path)
        saved_filenames[path] = saved_filename

        # Вычисление метрик (сравниваем с оригиналом)
        original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        metrics = calculate_metrics(original, processed)
        metrics_results[path] = metrics

    # Вывод результатов
    for path, metrics in metrics_results.items():
        print(f"\nМетрики для обработанного {path}:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  PSNR: {metrics['PSNR']:.4f}")
        print(f"  SSIM: {metrics['SSIM']:.4f}")

    # Сравниваем обработанные изображения с оригиналом (image.png)
    original = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

    for path in ['image2.png', 'image3.png']:
        processed = processed_images[path]
        metrics = calculate_metrics(original, processed)
        mlep = calculate_mlep(original, processed)
        print(f"\nСравнение обработанного {path} с оригиналом image.png:")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  PSNR: {metrics['PSNR']:.4f}")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
        print(f"  MLEP: {mlep:.4f}")

    # Сравниваем обработанные изображения с их исходными версиями
    print("\nСравнение обработанных изображений с их исходными версиями:")
    for path in ['image2.png', 'image3.png']:
        original_noisy = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        processed = processed_images[path]
        metrics = calculate_metrics(original_noisy, processed)
        mlep = calculate_mlep(original_noisy, processed)
        print(f"\n{path} (исходное с шумом vs обработанное):")
        print(f"  MSE: {metrics['MSE']:.4f}")
        print(f"  PSNR: {metrics['PSNR']:.4f}")
        print(f"  SSIM: {metrics['SSIM']:.4f}")
        print(f"  MLEP: {mlep:.4f}")

    # Выводим список сохраненных файлов
    print("\nСписок сохраненных файлов:")
    for original_path, saved_path in saved_filenames.items():
        print(f"  {original_path} -> {saved_path}")

    # Оценка шумности для каждого изображения с шумом относительно оригинала
    print("\nОценка шумности относительно оригинала (image.png):")
    for path in ['image2.png', 'image3.png']:
        metrics = evaluate_noise_metrics('image.png', path)
        print(f"\nРезультаты для {path}:")
        print("Метрики для шума Гаусса:")
        print(f"  MSE: {metrics['Gaussian Noise']['MSE']:.4f}")
        print(f"  PSNR: {metrics['Gaussian Noise']['PSNR']:.4f}")
        print(f"  SSIM: {metrics['Gaussian Noise']['SSIM']:.4f}")
        print("Метрики для шума Пуассона:")
        print(f"  SSIM: {metrics['Poisson Noise']['SSIM']:.4f}")
        print(f"  MLEP: {metrics['Poisson Noise']['MLEP']:.4f}")

    # Визуализация результатов
    visualize_results('image.png', 
                     ['image2.png', 'image3.png'], 
                     [saved_filenames['image2.png'], saved_filenames['image3.png']])

# Запуск программы
if __name__ == "__main__":
    main()
