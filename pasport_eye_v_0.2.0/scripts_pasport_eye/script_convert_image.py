#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
from PIL import Image

class ReturnPathConvertImage:
    def __init__(self, *args: str):
        self.abs_path_pict = args[0]

    def return_path_cache_pict_and_make_dir_cache(self, path_original_pict):
        full_name_file = path_original_pict.split("/")[-1]
        format_name = full_name_file.split(".")[-1]
        name_file_png = f"{full_name_file.split('.')[0]}.png"

        folder_cache = "./cache/picts"
        abs_path_folder_cache = os.path.abspath(folder_cache)

        if not os.path.exists(abs_path_folder_cache):
            os.makedirs(abs_path_folder_cache)

        if not format_name == "png":
            abs_path_name_file_cache = f"{abs_path_folder_cache}/{name_file_png}"
            return abs_path_name_file_cache
        else:
            path_copy_file = f"{abs_path_folder_cache}/{full_name_file}"
            shutil.copyfile(path_original_pict, path_copy_file)
            return path_copy_file

    def convert_image(self, path_original_pict, path_cache_pict):
        img = Image.open(path_original_pict)
        img.save(path_cache_pict)

    def main(self):
        path_original_pict = self.abs_path_pict
        path_cache_pict = self.return_path_cache_pict_and_make_dir_cache(path_original_pict)
        if path_cache_pict:
            self.convert_image(path_original_pict, path_cache_pict)
            return path_cache_pict
