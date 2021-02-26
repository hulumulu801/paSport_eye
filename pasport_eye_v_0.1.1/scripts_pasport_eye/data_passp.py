#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import imutils
import logging
import numpy as np
#########################################################################################################
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
#########################################################################################################
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
