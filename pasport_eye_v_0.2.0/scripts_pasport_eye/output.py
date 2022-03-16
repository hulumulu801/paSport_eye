#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import os
import cv2
import csv
import base64
import pickle
import shutil

class OUTput:
    def __init__(
                self,
                args: dict,
                output: str,
                abs_path_pict: str,
                dict_image_no_face_and_cropped_face: dict,
                list_face_descriptor: list,
                list_all_data_gorizontal: list,
                list_all_data_vertical: list
    ):
        if isinstance(args, dict):
            self.folder = args["path_pict"]
        if isinstance(output, str):
            self.output = output
        if isinstance(abs_path_pict, str):
            self.abs_path_pict = abs_path_pict
        if isinstance(dict_image_no_face_and_cropped_face, dict):
            self.image_cropped_fase = dict_image_no_face_and_cropped_face["cropped_face"]
        if isinstance(list_face_descriptor, list):
            self.list_face_descriptor = list_face_descriptor
        if isinstance(list_all_data_gorizontal, list):
            self.list_all_data_gorizontal = list_all_data_gorizontal
        if isinstance(list_all_data_vertical, list):
            self.list_all_data_vertical = list_all_data_vertical

    def create_and_return_path_and_scv_file(self, output=False):
        name_pict = self.abs_path_pict.split("/")[-1]

        abs_path_folder = f"{os.path.abspath(self.folder)}_recogn"

        abs_path_pict = f"{abs_path_folder}/pict"
        if not os.path.exists(abs_path_pict):
            os.makedirs(abs_path_pict)

        if output == "terminal":
            if not os.path.exists(abs_path_folder):
                os.makedirs(abs_path_folder)

            abs_path_pict_descript = f"{abs_path_folder}/descript"
            if not os.path.exists(abs_path_pict_descript):
                os.makedirs(abs_path_pict_descript)

            return name_pict, abs_path_pict, abs_path_pict_descript

        if output == "csv":
            abs_path_csv = f"{abs_path_folder}.csv"
            return abs_path_csv, name_pict, abs_path_pict

    def save_data(self, name_pict, abs_path_pict, abs_path_pict_descript=False, output=False):
        if output == "terminal":
            cv2.imwrite(f"{abs_path_pict}/{name_pict}", self.image_cropped_fase)
            for face_descriptor in self.list_face_descriptor:
                with open(f"{abs_path_pict_descript}/{name_pict.split('.')[0]}.pickle", "wb") as f:
                    pickle.dump(face_descriptor, f)
        if output == "csv":
            cv2.imwrite(f"{abs_path_pict}/{name_pict}", self.image_cropped_fase)

    def output_text(self, output=False):
        if output == "terminal":
            print("\nСерия и номер паспорта:\n")
        if output == "csv":
            new_list_text_recognition_vertical = []
        for list_data_vertical in self.list_all_data_vertical:
            for data_vertical in list_data_vertical:
                if len(data_vertical) == 0:
                    if output == "terminal":
                        print("Данные не распознаны!")
                    pass
                else:
                    if output == "terminal":
                        print(data_vertical)
                    if output == "csv":
                        text_recognition_vertical = f"{str(data_vertical)}/"
                        new_list_text_recognition_vertical.append(text_recognition_vertical)

        if output == "terminal":
            print("\nОстальные данные:\n")
        if output == "csv":
            new_list_text_recognition_gorizontal = []
        for list_data_gorizontal in self.list_all_data_gorizontal:
            for data_gorizontal in list_data_gorizontal:
                if len(data_gorizontal) == 0:
                    pass
                else:
                    if output == "terminal":
                        print(data_gorizontal)
                    if output == "csv":
                        text_recognition_gorizontal = f"{str(data_gorizontal)}/"
                        new_list_text_recognition_gorizontal.append(text_recognition_gorizontal)
        if output == "csv":
            text_recognition_gorizontal = "".join(new_list_text_recognition_gorizontal)
            text_recognition_gorizontal = f"Остальные_данные:{text_recognition_gorizontal}"
            text_recognition_vertical = "".join(new_list_text_recognition_vertical)
            text_recognition_vertical = f"Серия_и_номер_паспорта:{text_recognition_vertical}"
            return text_recognition_vertical, text_recognition_gorizontal

    def converting_cropped_face_in_base64(self, name_pict, abs_path_pict):
        abs_path_pict = f"{abs_path_pict}/{name_pict}"
        with open(abs_path_pict, "rb") as f:
            base_64 = base64.b64encode(f.read())
            base_64 = base_64.decode("utf-8")
        return base_64

    def write_csv(self, abs_path_csv, base_64, text_recognition_vertical, text_recognition_gorizontal):
        for face_descriptor in self.list_face_descriptor:
            descriptor = re.sub("[^0-9-,.\n]", "", str(face_descriptor))
            descriptor = re.sub("[\n]", "|", str(descriptor))

            data = {
                    "descriptor": descriptor,
                    "base_64": base_64,
                    "series_and_number": text_recognition_vertical,
                    "all_data": text_recognition_gorizontal
            }
            with open(abs_path_csv, "a") as f:
                writer = csv.writer(f)
                writer.writerow((
                                    data["descriptor"],
                                    data["base_64"],
                                    data["series_and_number"],
                                    data["all_data"]
                ))

    def del_folder(self, output=False):
        folder_cache = f"{os.path.abspath('./')}/cache"

        if output == "terminal" or output == "csv":
            if os.path.exists(folder_cache):
                shutil.rmtree(folder_cache)

        if output == "csv":
            abs_path_folder = f"{os.path.abspath(self.folder)}_recogn"
            if os.path.exists(abs_path_folder):
                shutil.rmtree(abs_path_folder)


    def main(self):
        if self.output == "terminal":
            name_pict, abs_path_pict, abs_path_pict_descript = self.create_and_return_path_and_scv_file(output="terminal")
            self.save_data(name_pict, abs_path_pict, abs_path_pict_descript=abs_path_pict_descript, output="terminal")
            self.output_text(output="terminal")
            self.del_folder(output="terminal")

        if self.output == "csv":
            abs_path_csv, name_pict, abs_path_pict = self.create_and_return_path_and_scv_file(output="csv")
            text_recognition_vertical, text_recognition_gorizontal = self.output_text(output="csv")
            self.save_data(name_pict, abs_path_pict, output="csv")
            base_64 = self.converting_cropped_face_in_base64(name_pict, abs_path_pict)
            self.write_csv(abs_path_csv, base_64, text_recognition_vertical, text_recognition_gorizontal)
            self.del_folder(output="csv")
