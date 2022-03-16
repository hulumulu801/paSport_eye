#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import imutils
import numpy as np

class SDWCV2:
    def __init__(self, dict_image_no_face_and_cropped_face: dict, visualization=False):
        self.img = dict_image_no_face_and_cropped_face["image_no_face"]
        self.visualization = visualization

    def return_image_preparation(self, image, gorizontal=False, vertical=False):
        # функция "подготовки" изображения
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (47, 47))
        if gorizontal == True:
            sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        if vertical == True:
            sq_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rect_kernel)
        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        (min_val, max_val) = (np.min(grad_x), np.max(grad_x))
        grad_x = (255*((grad_x-min_val)/(max_val-min_val))).astype("uint8")
        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, sq_kernel)
        thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        p = int(image.shape[1]*0.05)
        thresh[:, 0:p] = 0
        thresh[:, image.shape[1]-p:] = 0
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts, image

    def get_contour_precedence(self, contour, cols, gorizontal=False, vertical=False):
        # функция приоритета контура
        if gorizontal == True:
            tolerance_factor = 10
            origin = cv2.boundingRect(contour)
            return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
        if vertical == True:
            tolerance_factor = 1000
            origin = cv2.boundingRect(contour)
            return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

    def get_select_all_data(self, *args, gorizontal=False, vertical=False):
        # функция выделения данных(горизонтальных, вертикальных(серия и номер)) из паспорта
        cnts = list(args[0])
        image = args[1]
        data_lists = []
        if gorizontal == True:
            cnts.sort(key = lambda x : self.get_contour_precedence(x, image.shape[1], gorizontal=True))
        if vertical == True:
            cnts.sort(key = lambda x : self.get_contour_precedence(x, image.shape[1], vertical=True))
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            contour_area = float(h) * float(w)

            if gorizontal == True:
                if not contour_area > 10000 and contour_area > 500 and not h > 22:
                    pX = int((x + w) * 0.035)
                    pY = int((y + h) * 0.015)
                    (x, y) = (x - pX, y - pY)
                    (w, h) = (w + (pX * 2), h + (pY * 2))
                    cropped = image[
                                    y : y + h,
                                    x : x + w
                    ].copy()
                    data_lists.append(cropped)

                    if self.visualization == True:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if vertical == True:
                if not contour_area >= 2000 and contour_area > 700 and w > 50:
                    pX = int((x + w) * 0.375)
                    pY = int((y + h) * 0.15)
                    (x, y) = (x - pX, y - pY)
                    (w, h) = (w + (pX * 2), h + (pY * 2))
                    cropped = image[
                                    y : y + h,
                                    x : x + w
                    ].copy()
                    data_lists.append(cropped)

                    if self.visualization == True:
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if self.visualization == True:
            cv2.imshow("highlight_data", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return data_lists

    def main(self):
        image = self.img

        cnts_image_gorizontal, image_goriz = self.return_image_preparation(image, gorizontal=True)
        cnts_image_vertical, image_vert = self.return_image_preparation(image, vertical=True)

        data_list_gorizontal = self.get_select_all_data(cnts_image_gorizontal, image_goriz, gorizontal=True)
        data_list_vertical =  self.get_select_all_data(cnts_image_vertical, image_vert, vertical=True)
        return data_list_gorizontal, data_list_vertical
