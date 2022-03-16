#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
from scripts_pasport_eye.output import OUTput
from scripts_pasport_eye.remove_bg import RemoveBG
from scripts_pasport_eye.face_work import FaceWork
from scripts_pasport_eye.text_recognition import TessRecogn
from scripts_pasport_eye.search_data_with_cv2 import SDWCV2
from scripts_pasport_eye.script_return_abs_path import ReturnAbsPathPict
from scripts_pasport_eye.script_convert_image import ReturnPathConvertImage

def main():
    ap = ap = argparse.ArgumentParser(
                                        prog="python3 pasport_eye.py",
                                        usage="%(prog)s -p /ПУТЬ/К/ПАПКЕ/С/ИЗОБРАЖЕНИЯМИ --improved_recognition ON или OFF -o terminal или scv",
                                        description='''
                                                        Скрипт для распознавания паспортов РФ.
                                                        Работает только с первой страницей паспорта.
                                                        В дальнейшем будущем будет коллаборация с нейронными сетями
                                                        для лучшего распознавания!
                                                        ВЕРСИЯ: 0.1.2
                                                        '''
    )
    ap.add_argument(
                    "-p", "--path_pict", required=True,
                    help="Путь к каталогу изображений паспортов!"
    )
    ap.add_argument(
                    "--improved_recognition", required=True,
                    help='''
                            Метод улучшенного распознования: ON или OFF.
                            ON - более точное распознавание, но и "выхлоп" от
                            скрипта более "жирный", так же, медленная работа скрипта.
                            OFF - все как обычно, но возможно, не точное распознавание.
                    '''
    )
    ap.add_argument(
                    "-o", "--output", required = True,
                    help = '''

                    Метод выполнения скрипта: trminal или csv.
                    terminal - создается папка: название_базы + _recogn,
                    в ней подпапка pict - лица из паспортов
                    и подпапка descript - дескрипторы в двоичном формате,
                    выхлоп распознования - в терминал.

                    csv - дескрипторы(в формате: float), изображение лиц(в формате: base64),
                    а так же данные паспортов сохраняются в НАЗВАНИЕ_БАЗЫ_С_ИЗОБРАЖЕНИЯМИ.csv
                    '''
    )
    args = vars(ap.parse_args())

    if str(args["improved_recognition"]).lower() == "on" or str(args["improved_recognition"]).lower() == "off":
        improved_recognition = str(args["improved_recognition"]).lower()
    else:
        print("improved_recognition только ON или OFF\nВыход!")
        sys.exit()

    if str(args["output"]).lower() == "terminal" or str(args["output"]).lower() == "csv":
        output = str(args["output"]).lower()
    else:
        print("output только terminal или csv\nВыход!")
        sys.exit()


    # список изображений(абсолютный путь)
    list_abs_path_pict = ReturnAbsPathPict(args).main()
    for abs_path_pict in list_abs_path_pict:
        print(f"Идет распознование изображения: {abs_path_pict}\n")
        # возврат полного пути + конвертация изображения в .png
        path_cache_pict = ReturnPathConvertImage(abs_path_pict).main()
        # возврат полного пути + удаление заднего плана
        path_pict_no_bg = RemoveBG(path_cache_pict, visualization=False).main()

        # возврат словаря с данными(изображение без фото лица, само фото лица) + дескрипторы лица
        dict_image_no_face_and_cropped_face, list_face_descriptor = FaceWork(path_pict_no_bg, visualization=False).main()
        # возврат списка выделенных данных с помощью cv2
        data_list_gorizontal, data_list_vertical = SDWCV2(dict_image_no_face_and_cropped_face, visualization=False).main()
        # возврат списка распознанных с помощью tesseract данных
        list_all_data_gorizontal, list_all_data_vertical = TessRecogn(
                                                                        improved_recognition,
                                                                        data_list_gorizontal,
                                                                        data_list_vertical
        ).main()

        OUTput(
                args,
                output,
                abs_path_pict,
                dict_image_no_face_and_cropped_face,
                list_face_descriptor,
                list_all_data_gorizontal,
                list_all_data_vertical
        ).main()

if __name__ == "__main__":
    main()
