#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import cv2
import pytesseract
import numpy as np
from tqdm import tqdm

class TessRecogn:
    def __init__(self, improved_recognition, data_list_gorizontal, data_list_vertical):
        if isinstance(improved_recognition, str):
            self.improved_recognition = improved_recognition
        if isinstance(data_list_gorizontal, list):
            self.data_list_gorizontal = data_list_gorizontal
        if isinstance(data_list_vertical, list):
            self.data_list_vertical = data_list_vertical

    def return_abs_path_and_save_data_list(self, gorizontal=False, vertical=False):
        # функция сохранения распознанных данных из паспорта
        # т.к. tesseract'у нужны изображения
        list = []
        counter_image = 0
        folder = f"{os.path.abspath('./cache/picts_from_tess')}"
        if not os.path.exists(folder):
            os.makedirs(folder)
        if gorizontal == True:
            for data_gorizontal in self.data_list_gorizontal:
                name_pict = f"{folder}/{counter_image}_gorizontal.png"
                cv2.imwrite(name_pict, data_gorizontal)
                counter_image += 1
                list.append(name_pict)
        if vertical == True:
            for data_vertical in self.data_list_vertical:
                name_pict = f"{folder}/{counter_image}_vertical.png"
                cv2.imwrite(name_pict, data_vertical)
                counter_image += 1
                list.append(name_pict)
        return list

    def apply_threshold(self, erosion, argument):
        switcher = {
                1: cv2.threshold(cv2.medianBlur(erosion, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                2: cv2.threshold(cv2.medianBlur(erosion, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                3: cv2.threshold(cv2.medianBlur(erosion, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                4: cv2.threshold(cv2.medianBlur(erosion, 7), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                5: cv2.threshold(cv2.medianBlur(erosion, 9), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                6: cv2.threshold(cv2.GaussianBlur(erosion, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                7: cv2.threshold(cv2.GaussianBlur(erosion, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                8: cv2.threshold(cv2.GaussianBlur(erosion, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                9: cv2.threshold(cv2.GaussianBlur(erosion, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                10: cv2.threshold(cv2.GaussianBlur(erosion, (9, 9), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                11: cv2.threshold(cv2.GaussianBlur(erosion, (11, 11), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                12: cv2.threshold(cv2.GaussianBlur(erosion, (13, 13), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                13: cv2.threshold(cv2.GaussianBlur(erosion, (15, 15), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                14: cv2.threshold(cv2.GaussianBlur(erosion, (17, 17), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                15: cv2.threshold(cv2.GaussianBlur(erosion, (19, 19), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                16: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 1), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                17: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                18: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 5), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                19: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 7), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                20: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 9), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                21: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 11), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                22: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 13), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                23: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 15), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                24: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 17), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                25: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 19), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                26: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 21), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                27: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 23), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                28: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 25), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                29: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 27), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                30: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 29), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                31: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 31), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                32: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (1, 1), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                33: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (3, 3), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                34: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                35: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (7, 7), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                36: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (9, 9), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                37: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (11, 11), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                38: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (13, 13), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                39: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (15, 15), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                40: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (17, 17), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                41: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (19, 19), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                42: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (21, 21), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                43: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (23, 23), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                44: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (25, 25), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                45: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (27, 27), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                46: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (29, 29), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                47: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (31, 31), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                48: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (33, 33), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                49: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (35, 35), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                50: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (37, 37), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                51: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (39, 39), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                52: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (41, 41), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                53: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (43, 43), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                54: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (45, 45), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                55: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (47, 47), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                56: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (49, 49), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                57: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (51, 51), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                58: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (53, 53), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                59: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (55, 55), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                60: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (57, 57), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                61: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (59, 59), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                62: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (61, 61), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                63: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (63, 63), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        }
        return switcher.get(argument)

    def image_preparation(self, *args, gorizontal=False, vertical=False):
        # подготовка изображения перед распознованием tesseract
        image = args[0]
        kernel = np.ones((1, 1), np.uint8)
        image_read = cv2.imread(image)
        if gorizontal == True:
            resize = cv2.resize(image_read, None, fx=7, fy=7.5, interpolation=cv2.INTER_CUBIC)
        if vertical == True:
            resize = cv2.resize(image_read, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
        dilation = cv2.dilate(gray, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        if self.improved_recognition == "on":
            list_thresh = []
            for num in range(1, 64):
                thresh = self.apply_threshold(erosion, num)
                list_thresh.append(thresh)
            return list_thresh
        if self.improved_recognition == "off":
            if gorizontal == True:
                thresh = self.apply_threshold(erosion, 2)
            if vertical == True:
                thresh = self.apply_threshold(erosion, 1)
            return thresh

    def recognition_data(self, *args, gorizontal=False, vertical=False):
        # распознование tesseract
        list_data_recogn = []

        if self.improved_recognition == "on":
            list_thresh = args[0]
            for image_thresh in list_thresh:
                text = pytesseract.image_to_string(image_thresh, lang = "rus")
                if gorizontal == True:
                    if len(text) > 4:
                        text = re.sub("[^А-Я 0-9.]", "", text)
                        list_data_recogn.append(text)
                if vertical == True:
                    if len(text) >= 9 or len(text) <= 11:
                        text = re.sub("[^0-9]", "", text)
                        list_data_recogn.append(text)

        if self.improved_recognition == "off":
            image_thresh = args[0]
            text = pytesseract.image_to_string(image_thresh, lang = "rus")
            if gorizontal == True:
                if len(text) > 4:
                    text = re.sub("[^А-Я 0-9.]", "", text)
                    list_data_recogn.append(text)
            if vertical == True:
                if len(text) >= 9 or len(text) <= 11:
                    text = re.sub("[^0-9]", "", text)
                    list_data_recogn.append(text)
        return list_data_recogn


    def main(self):
        list_all_data_gorizontal = []
        list_all_data_vertical = []

        list_aps_path_gorizontal_data = self.return_abs_path_and_save_data_list(gorizontal=True)
        list_aps_path_vertical_data = self.return_abs_path_and_save_data_list(vertical=True)

        print("\n***Распознование остальных данных:***\n")
        for image_gorizontal in tqdm(list_aps_path_gorizontal_data):
            thresh_image_gorizontal = self.image_preparation(image_gorizontal, gorizontal=True)
            list_text_gorizontal = self.recognition_data(thresh_image_gorizontal, gorizontal=True)
            if len(list_text_gorizontal) >= 1:
                list_all_data_gorizontal.append(list_text_gorizontal)

        print("\n***Распознование серии и номера паспорта:***\n")
        for image_vertical in tqdm(list_aps_path_vertical_data):
            thresh_image_vertical = self.image_preparation(image_vertical, vertical=True)
            list_text_vertical = self.recognition_data(thresh_image_vertical, vertical=True)
            if len(list_text_vertical) >= 1:
                list_all_data_vertical.append(list_text_vertical)

        return list_all_data_gorizontal, list_all_data_vertical
