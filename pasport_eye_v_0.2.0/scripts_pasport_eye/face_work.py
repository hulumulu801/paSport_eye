#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import sys
import dlib
import shutil
from PIL import Image
from time import sleep
from skimage.draw import polygon_perimeter

class FaceWork:
    def __init__(self, path_pict_no_bg: str, visualization=False):
        self.path_pict_no_bg = path_pict_no_bg
        self.visualization = visualization # визуализация изображения

    def deleted_folder_cache(self):
        # функция удаления папки: ./cache
        abs_path_folder_cache = f"{os.path.abspath('./cache')}"
        if os.path.exists(abs_path_folder_cache):
            shutil.rmtree(abs_path_folder_cache, ignore_errors=True)

    def check_and_return_full_path_of_files(self):
        # проверяем на наличие файлы:
        # - shape_predictor_68_face_landmarks.dat
        # - dlib_face_recognition_resnet_model_v1.dat
        # - haarcascade_frontalface_default.xml
        # если они есть: возвращаем полный путь
        # если их нет: удалеем папку: ./cache и выходим!
        abs_path_folder = f"{os.path.abspath('./')}"

        shape_predictor_68_face_landmarks = f"{abs_path_folder}/shape_predictor_68_face_landmarks.dat"
        dlib_face_recognition_resnet_model_v1 = f"{abs_path_folder}/dlib_face_recognition_resnet_model_v1.dat"
        haarcascade_frontalface_default = f"{abs_path_folder}/haarcascade_frontalface_default.xml"

        if os.path.isfile(shape_predictor_68_face_landmarks) == True and os.path.isfile(dlib_face_recognition_resnet_model_v1) == True and os.path.isfile(haarcascade_frontalface_default) == True:
            sp = dlib.shape_predictor(shape_predictor_68_face_landmarks)
            facerec = dlib.face_recognition_model_v1(dlib_face_recognition_resnet_model_v1)
            face_cascade = cv2.CascadeClassifier(haarcascade_frontalface_default)
            return sp, facerec, face_cascade
        else:
            self.deleted_folder_cache()
            print(
                    '''
                    Файлы:
                    -shape_predictor_68_face_landmarks.dat
                    -dlib_face_recognition_resnet_model_v1.dat
                    -haarcascade_frontalface_default.xml
                    не обнаружены!
                    Выход!
                    '''
            )
            sys.exit()

    def return_abs_path_resize_pict(self, path_pict_no_bg):
        name_file = path_pict_no_bg.split("/")[-1]
        abs_path_folder = f"{os.path.abspath('./cache/picts_resize')}"
        if not os.path.exists(abs_path_folder):
            os.makedirs(abs_path_folder)
        full_name = f"{abs_path_folder}/{name_file}"
        return full_name

    def scale_image(self, path_pict_no_bg, abs_path_resize_pict):
        # изменяем размер изображения до 650x900
        # сохраняем
        image = Image.open(path_pict_no_bg)
        size = (650, 900)
        out_image = image.resize(size)
        out_image.save(abs_path_resize_pict)

    def find_face_and_resize_pict(self, path_pict_no_bg, sp, facerec):
        # функция поиска лица на изображении
        # если изображение перевернутое:
        # - переворачиваем изображение на 90 градусов и продолжаем поиск лица;
        # если лицо найдено:
        # - останавливаем переворачивание,
        # - изменяем размер изображения до 650x900
        # - извлекаем дескрипторы
        # - сохраняем
        detector = dlib.get_frontal_face_detector()
        list_face_descriptor = []
        coup_counter = True
        coup_counter_1 = 0

        while coup_counter:
            img = dlib.load_rgb_image(path_pict_no_bg)
            resize = cv2.resize(img, (600, 800))
            gray = cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
            dets = detector(gray, 2)
            if len(dets) == 1:
                coup_counter = False
                abs_path_resize_pict = self.return_abs_path_resize_pict(path_pict_no_bg)
                self.scale_image(path_pict_no_bg, abs_path_resize_pict)
                if self.visualization == True:
                    win = dlib.image_window()
                    win.set_image(resize)
                for k, d in enumerate(dets):
                    polygon_perimeter(
                                        [d.top(), d.top(), d.bottom(), d.bottom()],
                                        [d.right(), d.left(), d.left(), d.right()]
                    )
                    shape = sp(resize, d)
                    if self.visualization == True:
                        win.clear_overlay()
                        win.add_overlay(d)
                        win.add_overlay(shape)
                        sleep(4)
                    face_descriptor = facerec.compute_face_descriptor(resize, shape)
                    list_face_descriptor.append(face_descriptor)
                return abs_path_resize_pict, list_face_descriptor
            else:
                if coup_counter_1 <= 4:
                    coup_counter_1 += 1
                    img_1 = Image.open(path_pict_no_bg)
                    img_1.transpose(Image.ROTATE_90).save(str(path_pict_no_bg))
                else:
                    break

    def cropping_face_from_passport(self, face_cascade, abs_path_resize_pict):
        # обрезка лица из паспорта
        # возврат из функции словаря с данными:
        # - изображение без фото лица
        # - само фото лица
        image = cv2.imread(abs_path_resize_pict)
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
                if self.visualization == True:
                    cv2.imshow("image_no_face", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                break
            dict_image_no_face_and_cropped_face = {
                                                    "image_no_face": image,
                                                    "cropped_face": cropped_face
            }
            return dict_image_no_face_and_cropped_face

    def main(self):
        path_pict_no_bg = self.path_pict_no_bg # путь до изображения(без заднего плана)
        sp, facerec, face_cascade = self.check_and_return_full_path_of_files()
        abs_path_resize_pict, list_face_descriptor = self.find_face_and_resize_pict(path_pict_no_bg, sp, facerec)
        dict_image_no_face_and_cropped_face = self.cropping_face_from_passport(face_cascade, abs_path_resize_pict)
        return dict_image_no_face_and_cropped_face, list_face_descriptor
