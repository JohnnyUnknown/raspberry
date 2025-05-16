import math
from typing import List, Tuple

"""! В этом модуле определён класс Determ_coord для перевода относительных координат 
    изображения в географические координаты. """


class DetermCoord:
    """! Класс Determ_coord устанавливает соответствие пикселей изображения с географическими координатами. """
    point1, point2, point3, point4 = None, None, None, None
    point_pixel1, point_pixel2, point_pixel3, point_pixel4 = None, None, None, None
    target_point: List[float] = [0, 0]
    target_pixel_point = [0, 0]
    angle = 0  # Угол отклонения ориентации изображения от севера
    count_rotate = 0  # Количество поворотов изображения по часовой стрелке на 90 градусов
    old_begin = [0, 0]  # Относительные координаты левого верхнего угла

    def __init__(self,
                 point1: Tuple[float, float],
                 point2: Tuple[float, float],
                 point3: Tuple[float, float],
                 image_size: Tuple[int, ...]
                 ):
        """! Инициализация класса Determ_coord.
            @param point1: Координаты левого верхнего угла изображения (широта, долгота).
            @param point2: Координаты правого верхнего угла изображения (широта, долгота).
            @param point3: Координаты правого нижнего угла изображения (широта, долгота).
            @param image_size: Размер изображения в пикселях (ширина, высота)."""
        point_pixel1 = [0, 0]
        point_pixel2 = [0, image_size[1]]
        point_pixel3 = [image_size[0], image_size[1]]
        point_pixel4 = [image_size[0], 0]

        point4 = self.__find_p4(point1, point2, point3)
        self.point1, self.point2, self.point3, self.point4 = self.__new_points(point1, point2, point3, point4)
        self.point_pixel1, self.point_pixel2, self.point_pixel3, self.point_pixel4 = (
            self.__new_pixel_points(point_pixel1, point_pixel2, point_pixel3, point_pixel4))

        meters_in_a_degree_of_longitude = round((6378137 * math.cos(float((self.point1[0] + self.point4[0]) / 2) *
                                                                    math.pi / 180) * 2 * math.pi) / 360, 3)
        meters_in_a_degree_of_latitude = 111100
        self.coef = meters_in_a_degree_of_longitude / meters_in_a_degree_of_latitude

        self.angle = self.__find_angle()

    def calculate(self, pixel_center: List[int]) -> List[float]:
        """! Вычисляет целевую географическую координату на основе относительных координат.
            @param pixel_center: Пиксельные координаты центра (y, x).
            @return: Целевая географическая координата (широта, долгота). """
        self.target_pixel_point[0] = pixel_center[1]
        self.target_pixel_point[1] = pixel_center[0]
        if self.angle != 0:
            self.__new_center()
            self.target_point[0] = self.__find_latitude()
            self.target_point[1] = self.__find_longitude()
        else:
            pixel_price_x = round((self.point2[1] - self.point1[1]) / self.point_pixel2[1], 10)
            pixel_price_y = round((self.point2[0] - self.point3[0]) / self.point_pixel3[0], 10)
            self.target_point[0] = round(pixel_center[0] * abs(pixel_price_y) + self.point1[0], 6)
            self.target_point[1] = round(pixel_center[1] * abs(pixel_price_x) + self.point1[1], 6)
        target = self.target_point.copy()
        return target

    @staticmethod
    def __inversion(point_pixel1: List[int],
                    point_pixel2: List[int],
                    point_pixel3: List[int],
                    point_pixel4: List[int]
                    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        """! Инвертирует пиксельные координаты.
            @param point_pixel1: Пиксельные координаты первой точки.
            @param point_pixel2: Пиксельные координаты второй точки.
            @param point_pixel3: Пиксельные координаты третьей точки.
            @param point_pixel4: Пиксельные координаты четвёртой точки.
            @return: Инвертированные пиксельные координаты. """
        point_pixel1[0], point_pixel1[1] = point_pixel1[1], point_pixel1[0]
        point_pixel2[0], point_pixel2[1] = point_pixel2[1], point_pixel2[0]
        point_pixel3[0], point_pixel3[1] = point_pixel3[1], point_pixel3[0]
        point_pixel4[0], point_pixel4[1] = point_pixel4[1], point_pixel4[0]
        return point_pixel1, point_pixel2, point_pixel3, point_pixel4

    @staticmethod
    def __find_p4(point1: Tuple[float, float],
                  point2: Tuple[float, float],
                  point3: Tuple[float, float]
                  ) -> Tuple[float, float]:
        """! Находит четвёртую точку на основе трёх заданных.
            @param point1: Координаты левого верхнего угла изображения (широта, долгота).
            @param point2: Координаты правого верхнего угла изображения (широта, долгота).
            @param point3: Координаты правого нижнего угла изображения (широта, долгота).
            @return: Координаты четвёртого угла. """
        x_point = round(point1[1] - (point2[1] - point3[1]), 6)
        y_point = round(point1[0] - (point2[0] - point3[0]), 6)
        return y_point, x_point

    def __new_points(self,
                     point1: Tuple[float, float],
                     point2: Tuple[float, float],
                     point3: Tuple[float, float],
                     point4: Tuple[float, float]
                     ) -> List[Tuple[float, float]]:
        """! Метод смещает последовательность координат таким образом, при которой
            изображение располагается с севером наверху с некоторым углом отклонения.
            Количество поворотов изображения по часовой стрелке сохраняется в переменную
            'self.count_rotate'.
            @param point1: Координаты левого верхнего угла изображения (широта, долгота).
            @param point2: Координаты правого верхнего угла изображения (широта, долгота).
            @param point3: Координаты правого нижнего угла изображения (широта, долгота).
            @param point4: Координаты левого нижнего угла изображения (широта, долгота).
            @return: Новый порядок точек. """
        array = [point1, point2, point3, point4]
        array_out = []
        temp = min(array, key=lambda x: x[1])
        for i in range(4):
            if temp == array[i] and i != 0:
                self.count_rotate = i
                for j in range(4):
                    array_out.append(array[i])
                    i = i + 1 if i < 3 else 0
                break
            elif temp == array[i] and i == 0:
                return array
        return array_out

    def __new_pixel_points(self,
                           point1: List[int],
                           point2: List[int],
                           point3: List[int],
                           point4: List[int]
                           ) -> List[List[int]]:
        """! Метод смещает последовательность координат таким образом, при которой
            изображение располагается с севером наверху с некоторым углом отклонения.
            Также метод сохраняет новые относительные координаты угла с координатами при старом
            расположении изображения - [0, 0].
            @param point1: Пиксельные координаты первой точки.
            @param point2: Пиксельные координаты второй точки.
            @param point3: Пиксельные координаты третьей точки.
            @param point4: Пиксельные координаты четвёртой точки.
            @return: Новый порядок точек. """
        array = [point1, point2, point3, point4]
        array_out = []
        for i in range(self.count_rotate):
            k = 0
            array[0], array[1] = array[1], array[0]
            array[2], array[3] = array[3], array[2]
            array[0], array[1], array[2], array[3] = self.__inversion(array[0], array[1], array[2], array[3])
            for j in range(4):
                k = (j + 1) if k < 3 else 0
                array_out.append(array[k])
            array = array_out.copy()
            array_out.clear()
        self.old_begin = array[4 - self.count_rotate] if self.count_rotate > 0 else array[0]
        return array

    def __new_center(self):
        """! Метод находит новые относительные координаты пикселя с учётом поворота изображения. """
        if self.count_rotate % 2 == 0:
            self.target_pixel_point[0] = abs(self.old_begin[0] - self.target_pixel_point[0])
            self.target_pixel_point[1] = abs(self.old_begin[1] - self.target_pixel_point[1])
        else:
            self.target_pixel_point[0] = abs(self.old_begin[1] - self.target_pixel_point[0])
            self.target_pixel_point[1] = abs(self.old_begin[0] - self.target_pixel_point[1])
            self.target_pixel_point[0], self.target_pixel_point[1] = (
                self.target_pixel_point[1], self.target_pixel_point[0])

    def __find_angle(self) -> float:
        """! Метод определяет угол отклонения положения изображения от истинного севера.
            @return: Найденный угол. """
        size1 = round((self.point1[1] - self.point4[1]) * self.coef, 6)
        size2 = round(self.point1[0] - self.point4[0], 6)
        hypotenuse = math.sqrt(size1 * size1 + size2 * size2)
        sin_angle = abs(size1 / hypotenuse)
        angle_rad = round(math.asin(sin_angle), 6)
        return round(angle_rad, 3)

    def __find_latitude(self) -> float:
        """! Метод определяет координаты точки по широте.
            @return: Широта точки. """
        size_1 = abs((self.point_pixel2[1] - self.target_pixel_point[1]) * math.tan(self.angle))
        size_2 = size_1 + self.target_pixel_point[0]
        size_3 = round((self.point_pixel3[0] - size_2), 2)
        height = size_3 * math.cos(self.angle)
        height_pixel_price = abs((self.point2[0] - self.point3[0]) / (self.point_pixel3[0] * math.cos(self.angle)))
        height_orig = height_pixel_price * height
        return round(self.point3[0] + height_orig, 6)

    def __find_longitude(self) -> float:
        """! Метод определяет координаты точки по долготе.
            @return: Долгота точки. """
        size_1 = abs(self.target_pixel_point[1] + self.target_pixel_point[0] * math.tan(self.angle))
        size_2 = round((self.point_pixel2[1] - size_1), 2)
        width = size_2 * math.cos(self.angle)
        width_pixel_price = abs((self.point2[1] - self.point1[1]) / (self.point_pixel2[1] * math.cos(self.angle)))
        width_orig = (self.point_pixel2[1] * math.cos(self.angle) - width) * width_pixel_price
        return round(self.point1[1] + width_orig, 6)
