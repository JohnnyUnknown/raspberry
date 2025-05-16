import cv2 as cv
import numpy as np
#import torch
#import demo_superpoint
import AffineTransform as Affine
#from superpoint_superglue_deployment import Matcher
from typing import Tuple, List
from sys import path

"""! В этом модуле определён класс Method для выбора способа нахождения контрольных точек и их дескрипторов. """


class Method:
    """! Класс Method предназначен для поиска ключевых точек и их сопоставления с использованием различных
        алгоритмов. Он поддерживает методы SIFT, ORB, AKAZE, ASIFT и SuperPoint. """
    search_model = None
    matcher = None

    def __init__(self, method_index: int = 1, dist_kf: float = 0.5):
        """! Инициализация класса Method для поиска ключевых точек и их сопоставления. Запускает метод
            определения способа извлечения КТ и Д.
            @param method_index: Индекс метода поиска (по умолчанию 1 для SIFT).
            @param dist_kf: Коэффициент расстояния для фильтрации совпадений (по умолчанию 0.5). """
        self.method_index = method_index
        self.__set_method()
        self.dist_kf = dist_kf

    def __set_method(self):
        """! Устанавливает метод поиска ключевых точек в зависимости от заданного индекса метода. """
        match self.method_index:
            case 2:
                self.search_model = cv.AKAZE_create()
            case 3:
                self.search_model = cv.ORB_create(nfeatures=60000)
            #case 5:
            #    self.search_model = demo_superpoint.SuperPointNet()
            #    self.search_model.load_state_dict(torch.load(path[0] + '\\superpoint_v1.pth', weights_only=True))
            case _:
                self.search_model = cv.SIFT_create()  # nOctaveLayers=3, contrastThreshold=0.03, edgeThreshold=10

    def get_kp_and_des(self, img: np.ndarray) -> Tuple[List[cv.KeyPoint], np.ndarray] | Tuple[None, None]:
        """! Находит ключевые точки и дескрипторы для заданного изображения.
            @param img: Изображение для обработки.
            @return: Ключевые точки и дескрипторы (или None, если метод не поддерживается)."""
        if self.method_index == 4:
            kp, des = Affine.asift_detect_and_compute(img, self.search_model)
        # elif self.method_index == 5:
        #     return None, None
        else:
            kp, des = self.search_model.detectAndCompute(img, None)

        return list(kp), des

    def __set_matcher(self):
        """! Метод определяет способ сравнения дескрипторов изображений или самих изображений в случае
            с SuperPoint (method_index == 5). """
        if self.method_index == 3:
            self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        elif self.method_index == 4:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            self.matcher = cv.FlannBasedMatcher(index_params, search_params)
        # elif self.method_index == 5:
        #     self.matcher = Matcher(
        #         {
        #             "superpoint": {
        #                 "input_shape": (-1, -1),
        #                 "keypoint_threshold": 0.003,
        #             },
        #             "superglue": {
        #                 "match_threshold": self.dist_kf,
        #             },
        #             "use_gpu": False,
        #         }
        #     )
        else:
            self.matcher = cv.BFMatcher()

    def find_and_get_matches(self,
                             *,
                             img1: np.ndarray | None = None,
                             img2: np.ndarray | None = None,
                             des1: np.ndarray | None = None,
                             des2: np.ndarray | None = None
                             ) -> (Tuple[None, None, List[cv.DMatch]]
                                   | Tuple[List[cv.KeyPoint], List[cv.KeyPoint], List[cv.DMatch]]
                                   | Tuple[List[cv.KeyPoint], List[cv.KeyPoint], None]):
        """! Находит общие ключевые точки между двумя изображениями и возвращает их совпадения.
            @param img1: Первое изображение.
            @param img2: Второе изображение.
            @param des1: Дескрипторы первого изображения.
            @param des2: Дескрипторы второго изображения.
            @return: Ключевые точки первого и второго изображения (или None, None)
                     и список хороших совпадений (или None)."""
        self.__set_matcher()
        key_points1, key_points2 = None, None
        if self.method_index == 3:
            matches = self.matcher.match(des1, des2)
            good = [m for m in matches if m.distance < self.dist_kf * matches[-1].distance]
        elif self.method_index == 4:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.dist_kf * n.distance]
        # elif self.method_index == 5:
        #     key_points1, key_points2, _, _, good = self.matcher.match(img1, img2)
        else:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < self.dist_kf * n.distance]

        return key_points1, key_points2, good if len(good) >= 3 else None
