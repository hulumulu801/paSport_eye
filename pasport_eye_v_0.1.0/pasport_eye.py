#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import csv
import cv2
import sys
import dlib
import glob
import pickle
import base64
import shutil
import imutils
import logging
import argparse
import pytesseract
import numpy as np
from PIL import Image
from skimage import io
from uuid import uuid4
from imutils import paths
from natsort import natsorted
from scipy.spatial import distance
from skimage.draw import polygon_perimeter

def get_name_file_and_folder_name(image_path):
# функция возврата имени файла и имени папки
    try:
        name_base_file = str(image_path).split("/")[-1]
        name_base_folder = str(image_path).split("/")[0]
        return name_base_file, name_base_folder
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def get_scale_image(image_path):
    def get_full_name_scale_image(image_path):
        try:
            # функция сохранения масштабированного изображения
            folder = "./scale_image"
            name_file = str(image_path).split("/")[-1]
            if not os.path.exists(folder):
                os.makedirs(folder)
            path = os.path.abspath(folder)
            abs_path = path + "/" + name_file
            return abs_path
        except Exception as e:
            logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
            logging.exception("WARNING")
    try:
        # функция масштабирования изображения
        full_name_scale_image = get_full_name_scale_image(image_path)
        image = Image.open(image_path)
        size = (650, 900)
        out_image = image.resize(size)
        out_image.save(full_name_scale_image)
        return full_name_scale_image
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def get_find_a_face_in_the_image(scale_image):
# функция обрезки лица из паспорта
    try:
        path = "./"
        abs_path_casc = os.path.abspath(path)
        casc_path = abs_path_casc + "/haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(casc_path)
        image = cv2.imread(scale_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
                                                gray,
                                                scaleFactor = 1.4,
                                                minNeighbors = 6,
                                                flags = cv2.CASCADE_SCALE_IMAGE
        )
        if str(type(faces)) == "<class 'numpy.ndarray'>":
            for (x, y, w, h) in faces:
                new_x = x - 20
                new_y = y - 70
                new_w = w + 30
                new_h = h + 50
                cropped_face = image[
                                new_y : y + new_h,
                                new_x : x + new_w
                                ].copy()
                cv2.rectangle(image, (new_x, new_y), (x + new_w, y + new_h), (255, 255, 255), - 1).copy()
                break
            return image, cropped_face
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def get_select_all_data_gorizont(cropp_pict_face):
    def get_contour_precedence(contour, cols):
        try:
            # функция приоритета контура
            tolerance_factor = 10
            origin = cv2.boundingRect(contour)
            return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
        except Exception as e:
            logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
            logging.exception("WARNING")
    try:
        # функция выделения данных(горизонтальных) из паспорта
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (47, 47))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        image = cropp_pict_face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        gradX = cv2.Sobel(blackhat, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = - 1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        p = int(image.shape[1] * 0.05)
        thresh[:, 0 : p] = 0
        thresh[:, image.shape[1] - p:] = 0
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts.sort(key = lambda x : get_contour_precedence(x, image.shape[1]))
        data_lists = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            contour_area = float(h) * float(w)
            if not contour_area > 10000 and contour_area > 800 and not h > 22:
                pX = int((x + w) * 0.035)
                pY = int((y + h) * 0.015)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
                cropped = image[
                                y : y + h,
                                x : x + w
                                ].copy()
                data_lists.append(cropped)

                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # for i in range(len(cnts)):
            #     cv2.putText(image, str(i), cv2.boundingRect(cnts[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])
                # try:
                #     cv2.imshow("cropped", cropped)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                # except Exception as e:
                #     pass
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return data_lists
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def get_select_all_data_vertical(cropp_pict_face):
    def get_contour_precedence(contour, cols):
        try:
            # функция приоритета контура
            tolerance_factor = 1000
            origin = cv2.boundingRect(contour)
            return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
        except Exception as e:
            logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
            logging.exception("WARNING")
    try:
        # функция выделения данных(вертикальных(серия и номер)) из паспорта
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (47, 47))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        image = cropp_pict_face
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
        gradX = cv2.Sobel(blackhat, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = - 1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqKernel)
        thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        p = int(image.shape[1] * 0.05)
        thresh[:, 0 : p] = 0
        thresh[:, image.shape[1] - p:] = 0
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts.sort(key = lambda x : get_contour_precedence(x, image.shape[1]))
        data_lists = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            contour_area = float(h) * float(w)
            if not contour_area >= 2000 and contour_area > 1200 and w > 50:
                pX = int((x + w) * 0.375)
                pY = int((y + h) * 0.15)
                (x, y) = (x - pX, y - pY)
                (w, h) = (w + (pX * 2), h + (pY * 2))
                cropped = image[
                                y : y + h,
                                x : x + w
                                ].copy()
                data_lists.append(cropped)

                # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # for i in range(len(cnts)):
            #     cv2.putText(image, str(i), cv2.boundingRect(cnts[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])
                # try:
                #     cv2.imshow("cropped", cropped)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()
                # except Exception as e:
                #     pass
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return data_lists
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def save_data(data_passp_gorizontal, data_passp_vertical):
# функция сохранения распознанных данных из паспорта
# т.к. tesseract'у нужны картинки
    try:
        name_folder = "./scale_image/other"
        path = os.path.abspath(name_folder)
        if not os.path.exists(path):
            os.makedirs(path)
        counter_image = 0
        for passp_vertical in data_passp_vertical:
            name_pict = path + "/" + str(counter_image) + "_vertical.jpg"
            try:
                cv2.imwrite(name_pict, passp_vertical)
            except Exception as e:
                logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
                logging.exception("WARNING")
            counter_image += 1
        for passp_gorizontal in data_passp_gorizontal:
            name_pict = path + "/" + str(counter_image) + "_gorizontal.jpg"
            cv2.imwrite(name_pict, passp_gorizontal)
            counter_image += 1
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def text_recognition():
    def apply_threshold(erosion, argument):
        try:
            # функция "поиска порога изображения"
            switcher = {
                1: cv2.threshold(cv2.GaussianBlur(erosion, (11, 11), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                2: cv2.threshold(cv2.GaussianBlur(erosion, (3, 3), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                3: cv2.threshold(cv2.GaussianBlur(erosion, (1, 1), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                4: cv2.threshold(cv2.medianBlur(erosion, 7), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                5: cv2.threshold(cv2.medianBlur(erosion, 5), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                6: cv2.threshold(cv2.medianBlur(erosion, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                7: cv2.threshold(cv2.medianBlur(erosion, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                8: cv2.adaptiveThreshold(cv2.GaussianBlur(erosion, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
                9: cv2.adaptiveThreshold(cv2.medianBlur(erosion, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2),
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
        for f in natsorted(glob.glob(os.path.join(path, "*_vertical.jpg"))):
            kernel = np.ones((1, 1), np.uint8)
            image = cv2.imread(f)
            resize = cv2.resize(image, None, fx = 4, fy = 4, interpolation = cv2.INTER_CUBIC)
            gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            dilation = cv2.dilate(gray, kernel, iterations = 1)
            erosion = cv2.erode(dilation, kernel, iterations = 1)
            image = apply_threshold(erosion, 7)
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
            image = apply_threshold(erosion, 6)
            text = pytesseract.image_to_string(image, lang = "rus")
            text = re.sub("[^а-яА-Я 0-9.]", "", text)
            list_text_recognition_gorizontal.append(text)
        return list_text_recognition_vertical, list_text_recognition_gorizontal
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def main_2(image_path, detector, sp, facerec):
# функция поиска лица с помощью dlib,
# переворачивания изображения, если оно в наклоне,
# извлечение данных и распознование их,
# извлечения дескрипторов
    try:
        coup_counter = True
        coup_counter_1 = 0
        while coup_counter:
            img = dlib.load_rgb_image(image_path)
            resize = cv2.resize(img, (450, 800))
            gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 2)
            if len(dets) == 1:
                coup_counter = False

                scale_image = get_scale_image(image_path) # функция масштабирования изображения
                cropp_pict_face, cropped_face = get_find_a_face_in_the_image(scale_image) # функция обрезки лица из паспорта
                data_passp_gorizontal = get_select_all_data_gorizont(cropp_pict_face) # функция выделения данных(горизонтальных) из паспорта
                data_passp_vertical = get_select_all_data_vertical(cropp_pict_face) # функция выделения данных(вертикальных(серия и номер)) из паспорта
                save_data(data_passp_gorizontal, data_passp_vertical) # функция сохранения распознанных данных из паспорта
                list_text_recognition_vertical, list_text_recognition_gorizontal = text_recognition() # функция распознования текста tesseract'ом

                list_face_descriptor = []
                for k, d in enumerate(dets):
                    polygon_perimeter([d.top(), d.top(), d.bottom(), d.bottom()],
                                        [d.right(), d.left(), d.left(), d.right()])
                    shape = sp(img, d)
                    face_descriptor = facerec.compute_face_descriptor(img, shape) # дескрипторы лица
                    list_face_descriptor.append(face_descriptor)
                return list_text_recognition_vertical, list_text_recognition_gorizontal, list_face_descriptor, cropped_face
            else:
                if coup_counter_1 <= 4:
                    coup_counter_1 += 1
                    img_1 = Image.open(image_path)
                    img_1.transpose(Image.ROTATE_90).save(str(image_path))
                else:
                    break
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def save_the_cut_face(cropped, name_base_folder, uuid):
# функция сохранения обрезанного лица
    try:
        name_folder = "./" + str(name_base_folder) + "_recogn/pict"
        name_pict = uuid + ".jpg"
        path = os.path.abspath(name_folder)
        if not os.path.exists(path):
            os.makedirs(path)
        abs_path = path + "/" + name_pict
        try:
            cv2.imwrite(abs_path, cropped)
        except Exception as e:
            logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
            logging.exception("WARNING")
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def save_descriptors(list_face_descriptor, name_base_folder, uuid):
# функция сохранения дескрипторов в .pickle
    try:
        name_folder = "./" + str(name_base_folder) + "_recogn/descript"
        name_file = uuid + ".pickle"
        path = os.path.abspath(name_folder)
        if not os.path.exists(path):
            os.makedirs(path)
        abs_path = path + "/" + name_file
        for face_descriptor in list_face_descriptor:
            with open(abs_path, 'wb') as f:
                pickle.dump(face_descriptor, f)
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def get_path_csv_file(name_base_folder):
# полный путь фала .csv
    try:
        path = "./"
        name_file = name_base_folder + ".csv"
        abs_path = os.path.abspath(name_file)
        return abs_path
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def converting_cropped_face_in_base64(cropped_face, uuid):
    def save_crop_pict_from_base64(cropped_face, uuid):
        try:
            # Функция сохранения кропнутого лица из паспорта для base64
            path_scale_image = os.path.abspath("./scale_image")
            folder = "base_64"
            path_folder = path_scale_image + "/" + folder
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)
            abs_path = path_folder + "/" + str(uuid) + ".jpeg"
            try:
                resize = cv2.resize(cropped_face, (100, 100))
                cv2.imwrite(abs_path, resize)
            except Exception as e:
                logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
                logging.exception("WARNING")
            return abs_path
        except Exception as e:
            logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
            logging.exception("WARNING")
    try:
        # .jpg ==> base64
        path_to_image = save_crop_pict_from_base64(cropped_face, uuid) # Функция сохранения кропнутого лица из паспорта для base64
        with open(path_to_image, "rb") as f:
            base_64 = base64.b64encode(f.read())
            base64_message = base_64.decode("utf-8")
        return base64_message
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def get_data_pasport(list_text_recognition_vertical, list_text_recognition_gorizontal):
    try:
        new_list_text_recognition_vertical = []
        counter = 0
        for text_recognition_vertical in list_text_recognition_vertical:
            if counter <= 0:
                recognition_vertical = str(text_recognition_vertical) + "_or_"
                new_list_text_recognition_vertical.append(recognition_vertical)
                counter += 1
            elif counter >= 1:
                new_list_text_recognition_vertical.append(text_recognition_vertical)
                break
        series_and_number = "".join(new_list_text_recognition_vertical)
        series_and_number = re.sub("[^0-9_or]", "", str(series_and_number))
        series_and_number = "Серия_и_номер_паспорта:" + str(series_and_number)

        new_list_text_recognition_gorizontal = []
        for text_recognition_gorizontal in list_text_recognition_gorizontal:
            text_recognition_gorizontal = re.sub("\s", "_", str(text_recognition_gorizontal))
            if len(text_recognition_gorizontal) >= 5:
                text_recognition_gorizontal = str(text_recognition_gorizontal) + "/"
                new_list_text_recognition_gorizontal.append(text_recognition_gorizontal)
        all_data = "".join(new_list_text_recognition_gorizontal)
        all_data = "Остальные_данные:" + str(all_data)

        return series_and_number, all_data
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def write_csv(path_csv_file, list_face_descriptor, base_64, series_and_number_pasport, all_data_pasport):
# запись данных в csv
    try:
        for face_descriptor in list_face_descriptor:
            descriptor = re.sub("[^0-9-,.\n]", "", str(face_descriptor))
            descriptor = re.sub("[\n]", "|", str(descriptor))

            data = {
                    "descriptor": descriptor,
                    "base_64": base_64,
                    "series_and_number": series_and_number_pasport,
                    "all_data": all_data_pasport
            }
            with open(path_csv_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow((
                                    data["descriptor"],
                                    data["base_64"],
                                    data["series_and_number"],
                                    data["all_data"]
                ))
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def all_del_folder():
# функция удаления всех созданных папок
    try:
        path = "./"
        folder_name_scale_image = path + "scale_image"
        abs_path_folder_name_scale_image = os.path.abspath(folder_name_scale_image)
        if os.path.exists(abs_path_folder_name_scale_image):
            shutil.rmtree(abs_path_folder_name_scale_image)
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def main():
    try:
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        ap = argparse.ArgumentParser(
                                    prog = "python3 pasport_eye.py",
                                    usage = "%(prog)s -i /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ -m 1 или 2",
                                    description = '''
                                    Скрипт для распознавания паспортов РФ.
                                    Работает только с первой страницей паспорта.
                                    В дальнейшем будущем будет коллаборация с нейронными сетями для лучшего
                                    распознавания!
                                    ВЕРСИЯ: 0.1.0
                                    '''
        )
        ap.add_argument(
                        "-i", "--images", required = True,
                        help = "Путь к каталогу изображений!"
        )
        ap.add_argument(
                        "-m", "--method", required = True,
                        help = '''
                        Метод выполнения скрипта: 1 или 2.
                        №1 - создается папка с названием базы + _recogn,
                        в ней подпапка pict - лица из паспортов
                        и подпапка descript - дескрипторы в двоичном формате,
                        выхлоп распознования - в терминал.
                        №2 - дескрипторы(в формате: float), изображение лиц(в формате: base64),
                        а так же данные паспортов сохраняются в НАЗВАНИЕ_БАЗЫ_С_ИЗОБРАЖЕНИЯМИ.csv
                        '''
        )
        args = vars(ap.parse_args())
        for image_path in paths.list_images(args["images"]):
            uuid = str(uuid4())
            name_base_file, name_base_folder = get_name_file_and_folder_name(image_path) # функция возврата имени файла и имени папки
            list_text_recognition_vertical, list_text_recognition_gorizontal, list_face_descriptor, cropped_face = main_2(image_path, detector, sp, facerec) # функция поиска лица с помощью dlib, переворачивания изображения, если оно в наклоне, извлечение данных и распознование их, извлечения дескрипторов
            if int(args["method"]) == 1:
                save_the_cut_face(cropped_face, name_base_folder, uuid) # функция сохранения обрезанного лица
                save_descriptors(list_face_descriptor, name_base_folder, uuid) # функция сохранения дескрипторов в .pickle
                print("Имя файла: {}".format(str(name_base_file)))
                print("\n***Серия и номер паспорта:***\n")
                for recognition_vertical in list_text_recognition_vertical:
                    print(recognition_vertical)
                print("\n***Остальные данные:***\n")
                for recognition_gorizontal in list_text_recognition_gorizontal:
                    if len(recognition_gorizontal) >= 5:
                        print(recognition_gorizontal)
                print("#" * 70)
            elif int(args["method"]) == 2:
                path_csv_file = get_path_csv_file(name_base_folder) # полный путь фала .csv
                base_64 = converting_cropped_face_in_base64(cropped_face, uuid) # .jpg ==> base64
                series_and_number_pasport, all_data_pasport = get_data_pasport(list_text_recognition_vertical, list_text_recognition_gorizontal)
                write_csv(path_csv_file, list_face_descriptor, base_64, series_and_number_pasport, all_data_pasport) # запись данных в csv
            all_del_folder() # функция удаления всех созданных папок
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

if __name__ == "__main__":
    main()
