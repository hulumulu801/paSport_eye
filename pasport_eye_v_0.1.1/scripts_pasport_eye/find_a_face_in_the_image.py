#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import logging
#################################################################################################################
def get_find_a_face_in_the_image(scale_image):
    try:
        # функция "обрезки" лица из паспорта
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

        if str(type(faces)) == "<class 'tuple'>":
            pass
        elif str(type(faces)) == "<class 'numpy.ndarray'>":
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

            dict_image_and_cropped_face = {
                                            "image": image,
                                            "cropped_face": cropped_face
            }
            return dict_image_and_cropped_face
    except Exception as e:
        logging.basicConfig(filename = "LOG.log", filemode = "w", format = "%(name)s - %(levelname)s - %(message)s")
        logging.exception("WARNING")
