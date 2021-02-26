#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import cv2
import csv
import dlib
import shutil
import pickle
import base64
import logging
import argparse
from PIL import Image
from uuid import uuid4
from imutils import paths
from skimage.draw import polygon_perimeter
from scripts_pasport_eye.text_recognition import text_recognition
from scripts_pasport_eye.find_a_face_in_the_image import get_find_a_face_in_the_image
from scripts_pasport_eye.data_passp import get_select_all_data_gorizont, get_select_all_data_vertical

def get_name_file_and_folder_name(image_path):
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
        full_name_scale_image = get_full_name_scale_image(image_path)
        image = Image.open(image_path)
        size = (650, 900)
        out_image = image.resize(size)
        out_image.save(full_name_scale_image)
        return full_name_scale_image
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def save_data(data_passp_gorizontal, data_passp_vertical):
    try:
        # функция сохранения распознанных данных из паспорта
        # т.к. tesseract'у нужны картинки
        name_folder = "./scale_image/other"
        path = os.path.abspath(name_folder)
        if not os.path.exists(path):
            os.makedirs(path)
        counter_image = 0
        for passp_vertical in data_passp_vertical:
            name_pict = path + "/" + str(counter_image) + "_vertical.jpg"
            cv2.imwrite(name_pict, passp_vertical)
            counter_image += 1
        for passp_gorizontal in data_passp_gorizontal:
            name_pict = path + "/" + str(counter_image) + "_gorizontal.jpg"
            cv2.imwrite(name_pict, passp_gorizontal)
            counter_image += 1
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def main_2(image_path, detector, sp, facerec, improved_recognition):
    try:
        # функция поиска лица с помощью dlib,
        # переворачивания изображения, если оно в наклоне,
        # извлечение данных и распознование их,
        # извлечения дескрипторов
        coup_counter = True
        coup_counter_1 = 0
        while coup_counter:
            img = dlib.load_rgb_image(image_path)
            resize = cv2.resize(img, (600, 800))
            gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 2)
            if len(dets) == 1:
                coup_counter = False
                scale_image = get_scale_image(image_path) # функция масштабирования изображения
                dict_image_and_cropped_face = get_find_a_face_in_the_image(scale_image) # функция "обрезки" лица из паспорта
                if str(type(dict_image_and_cropped_face)) == "<class 'dict'>":
                    cropp_pict_face = dict_image_and_cropped_face["image"] # изображение паспорта без снимка лица
                    cropped_face = dict_image_and_cropped_face["cropped_face"] # снимок лица из паспорта
                    data_passp_gorizontal = get_select_all_data_gorizont(cropp_pict_face) # функция выделения данных(горизонтальных) из паспорта
                    data_passp_vertical = get_select_all_data_vertical(cropp_pict_face) # функция выделения данных(вертикальных(серия и номер)) из паспорта
                    save_data(data_passp_gorizontal, data_passp_vertical) # функция сохранения распознанных данных из паспорта
                    list_text_recognition_vertical, list_text_recognition_gorizontal = text_recognition(improved_recognition) # функция распознования текста tesseract'ом

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
    try:
        # функция сохранения обрезанного лица
        name_folder = "./" + str(name_base_folder) + "_recogn/pict"
        name_pict = uuid + ".jpg"
        path = os.path.abspath(name_folder)
        if not os.path.exists(path):
            os.makedirs(path)
        abs_path = path + "/" + name_pict
        cv2.imwrite(abs_path, cropped)
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

def save_descriptors(list_face_descriptor, name_base_folder, uuid):
    try:
        # функция сохранения дескрипторов в .pickle
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
    try:
        # полный путь фала .csv
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
            resize = cv2.resize(cropped_face, (100, 100))
            cv2.imwrite(abs_path, resize)
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
            if counter < len(list_text_recognition_vertical) - 1:
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
    try:
        # запись данных в csv
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
    try:
        # функция удаления всех созданных папок
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
        ap = argparse.ArgumentParser(
                                        prog = "python3 pasport_eye.py",
                                        usage = "%(prog)s -i /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ -m 1 или 2 -i_r ON или OFF",
                                        description = '''
                                        Скрипт для распознавания паспортов РФ.
                                        Работает только с первой страницей паспорта.
                                        В дальнейшем будущем будет коллаборация с нейронными сетями для лучшего
                                        распознавания!
                                        ВЕРСИЯ: 0.1.1
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
        ap.add_argument(
                        "-i_r", "--improved_recognition", required = True,
                        help = '''
                        Метод выполнения скрипта: ON или OFF.
                        ON - более точное распознавание, но и "выхлоп" от
                        скрипта более "жирный", так же, медленная работа скрипта.
                        OFF - все как обычно, но возможно, не точное распознавание.
                        '''
        )
        args = vars(ap.parse_args())
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

        improved_recognition = str(args["improved_recognition"])

        for image_path in paths.list_images(args["images"]):
            print("\n\n*** Идет распознавание изображения: {} ***\n".format(str(image_path)))
            uuid = str(uuid4())
            name_base_file, name_base_folder = get_name_file_and_folder_name(image_path) # функция возврата имени файла и имени папки
            try:
                list_text_recognition_vertical, list_text_recognition_gorizontal, list_face_descriptor, cropped_face = main_2(image_path, detector, sp, facerec, improved_recognition) # функция поиска лица с помощью dlib, переворачивания изображения, если оно в наклоне, извлечение данных и распознование их, извлечения дескрипторов
                if int(args["method"]) == 1:
                    save_the_cut_face(cropped_face, name_base_folder, uuid) # функция сохранения обрезанного лица
                    save_descriptors(list_face_descriptor, name_base_folder, uuid) # функция сохранения дескрипторов в .pickle
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
                pass
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")

if __name__ == "__main__":
    main()
