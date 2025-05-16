import cv2
import cv2 as cv
import Preprocessing
import os
import DetermCoord
import numpy as np


class Compare:
    good_match = 0
    filter_matches = 0
    center = None
    center_location = None
    key_1 = 0  # Кол-во контрольных точек основного изображения
    key_2 = 0  # Кол-во контрольных точек области видимости
    method = None  # Объект класса Method
    coordinates_map = [[48.245954, 46.164273],
                       [48.238956, 46.166415],
                       [48.238237, 46.160394]]

    def __init__(self, *, main_img, kp_main, des_main, height_main, img_2, altitude, method):
        self.img1 = main_img  # print_map
        self.kp1 = kp_main
        self.des1 = des_main
        self.height_map = height_main
        self.img_size = main_img.shape
        self.gray = img_2
        self.flight_altitude = altitude
        self.method = method
        self.my_map = DetermCoord.DetermCoord(*self.coordinates_map, self.img_size)

    # Поиск списка координат общих КТ на главном изображении
    def find_area(self, good_matches, kp1):
        matches = []
        for i in range(len(good_matches)):
            dmatch = good_matches[i]
            # Поиск найденных КТ для обеих изображений в списке КТ главного изображения
            large_image_KP = list(kp1[dmatch.queryIdx].pt)
            large_image_KP[0] = int(large_image_KP[0])
            large_image_KP[1] = int(large_image_KP[1])
            # Добавление в список КТ главного изображения, совпадающих с КТ искомого
            matches.append(large_image_KP)
        return matches

    # Поиск списка координат общих КТ на кадре после pixel_mask
    def location_images_2(self, good_matches, kp, matches_index):
        matches = []
        for i in range(len(good_matches)):
            if i in matches_index:
                large_image_KP = list(kp[good_matches[i].trainIdx].pt)
                large_image_KP[0] = int(large_image_KP[0])
                large_image_KP[1] = int(large_image_KP[1])
                matches.append(large_image_KP)
        return matches

    # Отображение местоположения дрона на главном изображении
    def print_map(self):
        # Отрисовка найденного центра на опороном изображении
        color = (0, 255, 0)
        temp_main_img = self.img1.copy()
        main_img = cv.circle(temp_main_img, self.center, radius=3, color=color, thickness=20)
        center2 = [round(self.gray.shape[1] / 2), round(self.gray.shape[0] / 2)]
        crop_img = cv.circle(self.gray, center2, radius=3, color=color, thickness=6)

        cv.imshow("Main image", Preprocessing.resize_img(main_img, 1024))
        if crop_img.shape[1] > 1024:
            new_width = 1024
        else:
            new_width = crop_img.shape[1]
        cv.imshow("Crop image", Preprocessing.resize_img(crop_img, new_width))
        cv.waitKey(0)
        cv.destroyAllWindows()
 
    # Удаление одинаковых точек
    def deleting_identical_points(self, main_matches, crop_matches):
        matches_1, matches_2 = [], []
        for i in range(len(main_matches)):
            flag = True
            for j in range(len(matches_1)):
                if main_matches[i] == matches_1[j] and crop_matches[i] == matches_2[j]:
                    flag = False
                    break
            if flag:
                matches_1.append(main_matches[i])
                matches_2.append(crop_matches[i])

        if len(matches_1) > 3:  # and cnt_2 > 3 and cnt_3 > 3
            # print(f"Общих точек: до {len(crop_matches)}, после отсеивания {len(matches_1)}")
            return matches_1, matches_2
        else:
            # print(f"Общих точек: до {len(crop_matches)}, после отсеивания 0")
            return [], []

    # Маска проверки найденных КТ на карте
    def pixel_mask(self, matches):  # принимаются координаты КТ главного изображения
        correct_matches = []
        correct_matches_index = []
        mask_correction = 1
        match_x = sorted(matches)
        match_y = sorted(matches, key=lambda i: i[1])

        if len(matches) % 2 == 0:
            indx1 = int(len(matches) / 2 - 1)
            indx2 = int(len(matches) / 2)
            median_y = (match_y[indx1][1] + match_y[indx2][1]) / 2
            median_x = (match_x[indx1][0] + match_x[indx2][0]) / 2
        else:
            indx = int((len(matches) - 1) / 2)
            median_y = match_y[indx][1]
            median_x = match_x[indx][0]

        # Нахождение коэффициента разницы высот полета и главного снимка для маски
        height_coefficient = round(self.height_map / self.flight_altitude, 2)

        for i in range(len(matches)):
            if ((matches[i][0] >= median_x - self.img1.shape[1] / height_coefficient * mask_correction)
                    and (matches[i][0] <= median_x + self.img1.shape[1] / height_coefficient * mask_correction)):
                if ((matches[i][1] >= median_y - self.img1.shape[1] / height_coefficient * mask_correction)
                        and (matches[i][1] <= median_y + self.img1.shape[1] / height_coefficient * mask_correction)):
                    correct_matches.append(matches[i])
                    correct_matches_index.append(i)

        return correct_matches, correct_matches_index

    # Вычисление матрицы преобразования координат
    def transformation_matrix(self, main_matches, matches_2):
        # Массивы с точками соответствия
        pts1 = np.float32([m for m in matches_2]).reshape(-1, 1, 2)
        pts2 = np.float32([m for m in main_matches]).reshape(-1, 1, 2)
        H, mask = cv.findHomography(pts1, pts2, cv.RANSAC)
        return H

    # Определение положения на опорном кадре с помощью матрицы преобразования
    def true_center(self, img, main_matches, matches):

        crop_center = np.array([[img.shape[1] / 2, img.shape[0] / 2]], dtype='float32').reshape(-1, 1, 2)
        H = self.transformation_matrix(main_matches, matches)
        try:
            find_center = cv.perspectiveTransform(crop_center, H)
            true_center = []
            true_center.append(int(find_center[0][0][0]))
            true_center.append(int(find_center[0][0][1]))

            # Отсеивание выбросов
            return true_center

        except cv2.error:
            print("Ошибка матрицы гомографии.\n")
            return None

    def comparator(self):
        kp2, des2 = self.method.get_kp_and_des(self.gray)

        if kp2 == None or len(kp2) > 3:
            if kp2 == None:
                self.kp1, kp2, good_matches = self.method.find_and_get_matches(img1=self.img1, img2=self.gray)
            else:
                _, _, good_matches = self.method.find_and_get_matches(des1=self.des1, des2=des2)

            #print(f"1: {len(self.kp1)}, 2: {len(kp2)}, Общих: {None if good_matches == None else len(good_matches)}")
            if good_matches != None:
                main_matches = self.find_area(good_matches, self.kp1)
                #matches_index = [i for i in range(len(main_matches))]
                main_matches, matches_index = self.pixel_mask(main_matches)
                matches_2 = self.location_images_2(good_matches, kp2, matches_index)
                #main_matches, matches_2 = self.deleting_identical_points(main_matches, matches_2)

                if len(main_matches) > 3:
                    self.center = self.true_center(self.gray, main_matches, matches_2)
                    if self.center:
                        main_img_with_points = cv.circle(cv.imread("main_with_points_new.jpg"), 
                                                self.center, radius=10, color=(0, 0, 0), 
                                                thickness=20)
                        cv.imwrite("main_with_points_new.jpg", main_img_with_points)
                        return self.my_map.calculate(self.center)
        #cv.imshow("gray", self.gray)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return 0
