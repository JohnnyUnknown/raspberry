import cv2
import numpy as np
from typing import List, Tuple

"""! В этом модуле определены функции для аффинного преобразования изображений. """


def affine_transform(img: np.ndarray, tilt: int, rotate: int) -> np.ndarray:
    """! Функция применяет аффинное преобразование к изображению с заданными углами наклона и поворота.
        @param img: Исходное изображение в формате NumPy.
        @param tilt: Угол наклона изображения в градусах.
        @param rotate: Угол поворота изображения в градусах.
        @return: Преобразованное изображение с применёнными наклоном и поворотом. """
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, rotate, 1.0)

    # Вычисляем косинус и синус из матрицы поворота
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    # Вычисляем новые размеры холста после поворота
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Корректируем матрицу поворота с учетом смещения центра
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    # Применяем поворот к изображению с учетом новых размеров
    img_rot = cv2.warpAffine(img, matrix, (new_w, new_h))

    # Преобразование перспективы для моделирования наклона
    height, width = img_rot.shape[:2]

    # Параметры изменения перспективы
    d = tilt * 10  # Фактор масштабирования для наклона

    # Определяем исходные и целевые точки для перспективного преобразования
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    dst_points = np.float32([[d, 0], [width - 1 - d, 0], [0, height - 1], [width - 1, height - 1]])

    # Получаем матрицу перспективного преобразования
    matrix_transform = cv2.getPerspectiveTransform(src_points, dst_points)
    img_transformed = cv2.warpPerspective(img_rot, matrix_transform, (width, height))

    return img_transformed


def asift_detect_and_compute(img1: np.ndarray, sift: cv2.SIFT) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """! Функция выполняет обнаружение и вычисление ключевых точек и дескрипторов для заданного изображения
        с использованием SIFT. Она применяет аффинные преобразования с различными углами наклона и поворота.
        @param img1: Исходное изображение в формате NumPy.
        @param sift: Объект SIFT для обнаружения ключевых точек и вычисления дескрипторов.
        @return: Список ключевых точек и массив дескрипторов. """
    tilt_angles = [0, 5, 10]  # Наклоны
    rotate_angles = [0, 180]  # Повороты

    kp_asift, des_asift = [], []
    for tilt in tilt_angles:
        for rotate in rotate_angles:
            img_transformed = affine_transform(img1, tilt, rotate)
            kp, des = sift.detectAndCompute(img_transformed, None)
            if kp is not None and des is not None:
                kp_asift.extend(kp)
                des_asift.extend(des)
    des_asift = np.array(des_asift)

    return kp_asift, des_asift
