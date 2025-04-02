# Technical vision. Camera Quality Assessment Tool (CQAT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/your-org/your-repo/graphs/commit-activity)

## Подключение к репозиторию (Git)

Следуйте этим шагам, чтобы клонировать репозиторий и начать работу с проектом:

1.  **Клонируйте репозиторий:**
    ```bash
    cd (your_repo)
    git clone https://github.com/yegorprikhodieko/technical_vision.git
    ```
2.  **Установите python на своё устройство:**
     https://www.python.org/

4.  **Установите модули:**
    ```bash
    pip install opencv-python
    pip install pillow
    pip install numpy
    ```
3.  **Запустите проект:**
    ```bash
    cd (your_repo)/source
    python start.py
    ```
    
## Описание проекта

Camera Quality Assessment Tool (CQAT) - это инструмент на Python, предназначенный для автоматизированной оценки качества изображений, получаемых с камер мобильных телефонов и других устройств.  Проект использует библиотеки компьютерного зрения и методы обработки изображений для объективного измерения различных параметров, определяющих качество снимков.

**Основные цели проекта:**

*   **Автоматизация:** Снижение трудозатрат на ручное тестирование камер.
*   **Объективность:** Получение количественных оценок качества изображений на основе измеримых характеристик.
*   **Воспроизводимость:** Обеспечение возможности повторного тестирования в контролируемых условиях для сопоставления результатов.
*   **Анализ:** Предоставление данных для анализа сильных и слабых сторон камер.

**Функциональность:**
*   **Анализ изображения:** Автоматическое измерение следующих параметров:
    *   Разрешение
    *   Яркость и контраст
    *   Уровень шума
    *   Четкость (резкость)
    *   Точность цветопередачи (Delta E)
*   **Настраиваемые тесты:** Возможность добавления новых тестов и метрик для анализа изображений.
*   **Сохранение результатов:** Сохранение результатов тестирования в формате CSV для дальнейшего анализа и сравнения.
*   **(Планируется)** Графический интерфейс пользователя (GUI) для упрощения управления тестами и просмотра результатов.
*   **(Планируется)** Использование машинного обучения для более сложного анализа изображений.

**Технологии:**

*   Python 3.7+
*   OpenCV (cv2)
*   NumPy
*   Pillow (PIL)
*   Scikit-image
*   Matplotlib

## Участники проекта

| Имя                   | Роль в проекте                | Контакты                      |
| --------------------- | ------------------------------- | ----------------------------- |
| [Приходько Егор Анатольевич]           | Менеджмент, разработка | [yaga.pri99@mail.ru]                  |
| [Лабнер Юлия] | UI/UX дизайн, документация | [Email]           |
| [Вихляев Александр Сергеевич] | Разработка, тестирование | [Email]           |
| [Макаров Евгений Владиславоdbx] | Разработка, тестирование | [Email]                           |

**Руководитель:**

 Верещагин Владислав Юрьевич.

## Contribution

Мы приветствуем вклад в развитие проекта!
