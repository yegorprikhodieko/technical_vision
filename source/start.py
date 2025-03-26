import cv2
import image_analysis

def take_snapshot_and_display(filename="snapshot.jpg"):
    # Делает снимок с веб-камеры, сохраняет его и отображает на экране.
    try:
        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise IOError("Не удалось открыть веб-камеру")

        ret, frame = camera.read()

        if not ret:
            raise IOError("Не удалось сделать снимок")

        cv2.imwrite(filename, frame)
        print(f"Снимок успешно сохранен в файл: {filename}")

        cv2.imshow("Снимок с веб-камеры", frame)
        print("яркость и контраст",image_analysis.get_brightness_contrast('snapshot.jpg'))
        print("шум (стандартное отклонение).",image_analysis.calculate_noise('snapshot.jpg'))
        print("четкость (дисперсия лапласиана).",image_analysis.calculate_laplacian_variance('snapshot.jpg'))
        
        cv2.waitKey(0)

    except IOError as e:
        print(f"Ошибка: {e}")

    finally:
        if 'camera' in locals() and camera.isOpened():
            camera.release()
        cv2.destroyAllWindows()

take_snapshot_and_display()