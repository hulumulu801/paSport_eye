#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import cv2
import glob
import logging
import pytesseract
import numpy as np
from natsort import natsorted
from collections import OrderedDict
from progress.bar import IncrementalBar

def text_recognition(improved_recognition):
    def apply_threshold(erosion, argument):
        try:
            # функция "поиска порога изображения"
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
        except Exception as e:
            logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
            logging.exception("WARNING")
    try:
        # функция распознования текста tesseract'ом
        list_text_recognition_vertical = []
        list_text_recognition_gorizontal = []
        folder_name = "./scale_image/other"
        path = os.path.abspath(folder_name)

        if str(improved_recognition) == "ON":
            for f in natsorted(glob.glob(os.path.join(path, "*_vertical.jpg"))):
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.imread(f)
                resize = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC) # fx = 4, fy = 4
                gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                dilation = cv2.dilate(gray, kernel, iterations = 1)
                erosion = cv2.erode(dilation, kernel, iterations = 1)
                len_text = 1
                bar = IncrementalBar("РАСПОЗНАВАНИЕ СЕРИИ И НОМЕРА ПАСПОРТА, ПОДОЖДИТЕ...", max = 63)
                while True:
                    if len_text <= 63:
                        bar.next()
                        image = apply_threshold(erosion, len_text)
                        text = pytesseract.image_to_string(image, lang = "rus")
                        text = re.sub("[^0-9]", "", text)
                        if len(text) <= 9 or len(text) >= 11:
                            len_text += 1
                        else:
                            list_text_recognition_vertical.append(text)
                            len_text += 1
                    else:
                        break
            list_text_recognition_vertical = list(set(list_text_recognition_vertical))

            for f in natsorted(glob.glob(os.path.join(path, "*_gorizontal.jpg"))):
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.imread(f)
                resize = cv2.resize(image, None, fx = 5, fy = 5, interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                dilation = cv2.dilate(gray, kernel, iterations = 1)
                erosion = cv2.erode(dilation, kernel, iterations = 1)
                len_text = 1
                bar = IncrementalBar("РАСПОЗНАВАНИЕ ДАННЫХ, ПОДОЖДИТЕ...", max = 63)
                while True:
                    if len_text <= 63:
                        bar.next()
                        image = apply_threshold(erosion, len_text)
                        text = pytesseract.image_to_string(image, lang = "rus")
                        # numbers_text = re.sub("[^0-9.]", "", text)
                        text = re.sub("[^А-Я ]", "", text)
                        if len(text) < 4:
                            len_text += 1
                        else:
                            list_text_recognition_gorizontal.append(text)
                            len_text += 1
                    else:
                        break
            list_text_recognition_gorizontal = list(OrderedDict.fromkeys(list_text_recognition_gorizontal))

        elif str(improved_recognition) == "OFF":
            for f in natsorted(glob.glob(os.path.join(path, "*_vertical.jpg"))):
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.imread(f)
                resize = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                dilation = cv2.dilate(gray, kernel, iterations = 1)
                erosion = cv2.erode(dilation, kernel, iterations = 1)
                image = apply_threshold(erosion, 1)
                text = pytesseract.image_to_string(image, lang = "rus")
                text = re.sub("[^0-9. ]", "", text)
                list_text_recognition_vertical.append(text)

            for f in natsorted(glob.glob(os.path.join(path, "*_gorizontal.jpg"))):
                kernel = np.ones((1, 1), np.uint8)
                image = cv2.imread(f)
                resize = cv2.resize(image, None, fx = 7, fy = 7.5, interpolation = cv2.INTER_CUBIC)
                gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
                dilation = cv2.dilate(gray, kernel, iterations = 1)
                erosion = cv2.erode(dilation, kernel, iterations = 1)
                image = apply_threshold(erosion, 2)
                text = pytesseract.image_to_string(image, lang = "rus")
                text = re.sub("[^а-яА-Я 0-9.]", "", text)
                list_text_recognition_gorizontal.append(text)

        else:
            print("ТОЛЬКО: ON или OFF")

        return list_text_recognition_vertical, list_text_recognition_gorizontal
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")
