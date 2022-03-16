#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import cv2
import numpy as np
from PIL import Image
from uuid import uuid4
from rembg.bg import remove

class RemoveBG:
    def __init__(self, *args: str, visualization=False):
        self.abs_path_pict = args[0]
        self.visualization = visualization

    def return_output_path_pict(self, input_path_pict):
        name_file = f"{input_path_pict.split('/')[-1].split('.')[0]}_{uuid4()}"
        format_file = input_path_pict.split("/")[-1].split(".")[-1]
        file = f"{name_file}.{format_file}"

        folder = "./cache/picts_rem_bg"
        abs_path_folder = os.path.abspath(folder)
        if not os.path.exists(abs_path_folder):
            os.makedirs(abs_path_folder)
        abs_path_file = f"{abs_path_folder}/{file}"
        return abs_path_file

    def rem_bg(self, input_path_pict, output_path_pict):
        input = Image.open(input_path_pict)
        output = remove(input)
        output.save(output_path_pict)

    def del_picts(self, input_path_pict):
        os.remove(input_path_pict)

    def main(self):
        input_path_pict = self.abs_path_pict
        if self.visualization == True:
            image = cv2.imread(input_path_pict)
            cv2.imshow("orig_image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        output_path_pict = self.return_output_path_pict(input_path_pict)
        self.rem_bg(input_path_pict, output_path_pict)
        self.del_picts(input_path_pict)
        return output_path_pict
