import cv2
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import pandas as pd
from datetime import datetime
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
import aiofiles
import image_analysis

bot = Bot(token="7972736557:AAFoqecWUvjD6hHtQ4kpB4IM7jmeYwL0zhQ")
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('camera_rank.db')
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS photos
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      phone_model TEXT,
                      file_path TEXT,
                      sharpness REAL,
                      contrast REAL,
                      dynamic_range REAL,
                      upload_date TIMESTAMP)''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS ratings
                     (phone_model TEXT PRIMARY KEY,
                      avg_sharpness REAL,
                      avg_contrast REAL,
                      total_photos INTEGER)''')
    
    conn.commit()
    conn.close()

init_db()

async def save_to_db(user_id, phone_model, file_path, sharpness,contrast):
    conn = sqlite3.connect('camera_rank.db')
    cursor = conn.cursor()
    
    # Сохраняем фото и данные анализа
    cursor.execute('''INSERT INTO photos 
                     (user_id, phone_model, file_path, sharpness, contrast, upload_date)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (user_id, phone_model, file_path, sharpness, contrast, datetime.now()))
    
    # Обновляем агрегированный рейтинг
    cursor.execute('''INSERT OR REPLACE INTO ratings
                     (phone_model, avg_sharpness, avg_contrast, total_photos)
                     SELECT 
                         phone_model,
                         AVG(sharpness),
                         0,  -- Заглушка для шумов (можно добавить реальный анализ)
                         COUNT(*)
                     FROM photos
                     WHERE phone_model = ?
                     GROUP BY phone_model''', (phone_model,))
    
    conn.commit()
    conn.close()

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "📷 Пришлите фото, снятое на ваш смартфон, и укажите модель "
        "(например: 'Samsung S23 Ultra')."
    )

@dp.message(lambda msg: msg.photo and msg.caption)
async def handle_photo(message: Message):
    try:
        # Получаем файл фото
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        
        # Скачиваем фото с использованием aiofiles
        async with aiofiles.open(f"user_photo_{message.from_user.id}.jpg", "wb") as f:
            photo_bytes = await bot.download_file(file.file_path)
            await f.write(photo_bytes.read())

        # Анализ изображения
        image = cv2.imread("user_photo.jpg", cv2.IMREAD_GRAYSCALE)
        if image is None:
            await message.answer("❌ Не удалось обработать фото. Отправьте его снова.")
            return

        sharpness = image_analysis.calculate_laplacian_variance(image)
        brightness,contrast = image_analysis.get_brightness_contrast(image)
        # sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
        
        await save_to_db(
            user_id=message.from_user.id,
            phone_model=message.caption,
            file_path=file.file_path,
            sharpness=sharpness,
            contrast = contrast,
            # brightness = brightness
        )

        # Получаем данные рейтинга
        conn = sqlite3.connect('camera_rank.db')
        cursor = conn.cursor()
        
        # 1. Сначала получаем все модели для ранжирования
        cursor.execute('''
            SELECT phone_model, avg_sharpness 
            FROM ratings 
            WHERE total_photos >= 3
            ORDER BY avg_sharpness DESC
        ''')
        all_models = cursor.fetchall()
        
        # 2. Находим позицию текущей модели
        current_rank = None
        avg_sharpness = None
        total_photos = None
        print(all_models)
        for rank, (model, score) in enumerate(all_models, start=1):
            if model == message.caption:
                current_rank = rank
                print(current_rank)
                avg_sharpness = score
                # Получаем общее количество оценок для этой модели
                cursor.execute('SELECT total_photos FROM ratings WHERE phone_model = ?', 
                             (message.caption,))
                total_photos = cursor.fetchone()[0]
                break
        
        # 3. Получаем топ-5 моделей
        top_5 = all_models[:5]
        
        # 4. Персональная статистика
        cursor.execute('''
            SELECT COUNT(*), AVG(sharpness) 
            FROM photos 
            WHERE user_id = ?
        ''', (message.from_user.id,))
        user_stats = cursor.fetchone()
        
        conn.close()

        # Формируем ответ
        response = [
            f"📊 Анализ снимка ({message.caption}):",
            f"• Резкость: {sharpness:.1f}/100",
            f"• Контраст: {contrast:.1f}"
        ]

        if current_rank is not None:
            response.extend([
                f"",
                f"🏆 Позиция в рейтинге: #{current_rank}",
                f"• Средняя резкость модели: {avg_sharpness:.1f}",
                f"• Всего оценок модели: {total_photos}",
                f"",
                f"📈 Ваших оценок: {user_stats[0]} (средняя: {user_stats[1] if user_stats[1] else 0:.1f})",
                f"",
                f"Топ-5 камер:"
            ])
            
            for i, (model, score) in enumerate(top_5, 1):
                response.append(f"{i}. {model}: {score:.1f}")
        else:
            response.append("\nℹ️ Для включения в рейтинг нужно минимум 3 оценки этой модели")

        await message.answer("\n".join(response))

    except Exception as e:
        print(e)
        await message.answer(f"❌ Ошибка: {str(e)}")
    
    async def generate_rating_plot():
        conn = sqlite3.connect('camera_rank.db')
        df = pd.read_sql('SELECT phone_model, avg_sharpness FROM ratings ORDER BY avg_sharpness DESC LIMIT 10', conn)
        conn.close()
        
        plt.figure(figsize=(10, 6))
        plt.barh(df['phone_model'], df['avg_sharpness'], color='skyblue')
        plt.xlabel('Средняя резкость')
        plt.title('Топ-10 камер смартфонов')
        plt.tight_layout()
        plt.savefig('top10.png')
        return 'top10.png'

    # В handle_photo:
    plot_path = await generate_rating_plot()
    await message.answer_photo(types.FSInputFile(plot_path))
if __name__ == '__main__':
    dp.run_polling(bot)