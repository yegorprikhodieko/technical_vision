import cv2
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
import aiofiles
import image_analysis

bot = Bot(token="7972736557:AAFoqecWUvjD6hHtQ4kpB4IM7jmeYwL0zhQ")
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

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
        async with aiofiles.open("user_photo.jpg", "wb") as f:
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

        await message.answer(
            f"📊 Анализ снимка ({message.caption}):\n"
            f"• Резкость: {sharpness:.1f}/100\n"
            f"• Яркость: {brightness:.1f} , {contrast:.1f}\n"
            f"• Примерный рейтинг: {min(sharpness / 20, 5.0):.1f}/5"
        )

    except Exception as e:
        await message.answer(f"❌ Ошибка: {str(e)}")

if __name__ == '__main__':
    dp.run_polling(bot)