import cv2 as cv
import numpy as np

"""! В этом модуле определены и применяются функции для искажений изображений. """

# Определение индексов типов искажений изображений для вывода в протокол
augment = {
    0: "Без искажений",
    1: "Поворот ",
    2: "Поворот ",
    3: "Поворот ",
    4: "Яркость ",
    5: "Яркость ",
    6: "Шумы",
    7: "Блюр (5, 5)"
}


def rotate_image(img: np.ndarray, degrees: int) -> np.ndarray:
    """! Поворачивает изображение на заданный угол.
        @param img: Исходное изображение.
        @param degrees: Угол поворота в градусах.
        @return: Повёрнутое изображение. """
    height, width = img.shape[:2]
    center_x, center_y = (width / 2, height / 2)
    matrix = cv.getRotationMatrix2D((center_x, center_y), degrees, 1.0)
    out_image = cv.warpAffine(img, matrix, (width, height))
    return out_image


def brightness(img: np.ndarray, value: int) -> np.ndarray:
    """! Изменяет яркость изображения.
        @param img: Исходное изображение.
        @param value: Значение изменения яркости (положительное - увеличение, отрицательное - уменьшение) [-255: 255].
        @return: Изображение с изменённой яркостью."""
    color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    hsv = cv.cvtColor(color, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    if 0 < value < 256:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    elif -256 < value < 0:
        v[v > abs(value)] -= abs(value)
        v[v <= abs(value)] = 0
    else:
        return img
    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img


def add_noise(img: np.ndarray) -> np.ndarray:
    """! Добавляет шум к изображению.
        @param img: Исходное изображение.
        @return: Изображение с добавленным шумом."""
    noise = np.zeros(img.shape, np.uint8)
    cv.randn(noise, 0, 20)
    img_n = cv.add(img, noise)
    return img_n


def add_blur(img: np.ndarray) -> np.ndarray:
    """! Применяет размытие к изображению.
        @param img: Исходное изображение.
        @return: Размытое изображение."""
    img_bl = cv.blur(img, (5, 5))
    return img_bl


def augmentation(img: np.ndarray, augment_index: int) -> np.ndarray:
    """! Применяет аугментацию к изображению в зависимости от индекса. Значение по ключу соответствующему индексу
        аугментации в словаре 'augment' будет изменено в соответствии с переданными значениями для изменений.
        @param img: Исходное изображение.
        @param augment_index: Индекс аугментации (0-7).
        @return: Изменённое изображение."""
    out_img = img.copy()
    match augment_index:
        case 1:
            degrees = 60
            out_img = rotate_image(img, degrees)
            augment[1] = augment[1].split()[0] + f" {degrees}\u00b0"
        case 2:
            degrees = -120
            out_img = rotate_image(img, degrees)
            augment[2] = augment[2].split()[0] + f" {degrees}\u00b0"
        case 3:
            degrees = 180
            out_img = rotate_image(img, degrees)
            augment[3] = augment[3].split()[0] + f" {degrees}\u00b0"
        case 4:
            delta_brightness = 30
            out_img = brightness(img, delta_brightness)
            augment[4] = augment[4].split()[0] + f" +{delta_brightness}"
        case 5:
            delta_brightness = -30
            out_img = brightness(img, delta_brightness)
            augment[5] = augment[5].split()[0] + f" {delta_brightness}"
        case 6:
            out_img = add_noise(img)
        case 7:
            out_img = add_blur(img)
    return out_img


def resize_img(img: np.ndarray, new_width: int) -> np.ndarray:
    """! Изменяет размер изображения с сохранением пропорций.
        @param img: Исходное изображение.
        @param new_width: Новая ширина изображения.
        @return: Изображение с изменённым размером."""
    new_height = int(img.shape[0] * (new_width / img.shape[1]))
    resized_image = cv.resize(img, (new_width, new_height))
    return resized_image


def definition_of_blur(height: int, altitude: int) -> tuple:
    """! Определяет параметры размытия в зависимости от разности высот изображений для приведения изображений
        к оптимальному виду для сравнения.
        @param height: Высота первого изображения.
        @param altitude: Высота второго изображения.
        @return: Значение ядра размытия."""
    diff = int(height / altitude)
    if diff <= 5:
        return 5, 5
    elif diff % 2 == 1:
        return diff, diff
    else:
        return diff + 1, diff + 1
