#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

class ReturnAbsPathPict:
    def __init__(self, *args: str):
        self.args = args[0]

    def return_list_abs_path_pict(self, path_folder):
        list = []

        folder = f"./{path_folder}"
        abs_path_folder = os.path.abspath(folder)

        onlyfiles = [f for f in os.listdir(abs_path_folder) if os.path.isfile(os.path.join(abs_path_folder, f))]

        for file in onlyfiles:
            abs_file = f"{abs_path_folder}/{file}"
            list.append(abs_file)
        return list

    def main(self):
        path_folder = self.args["path_pict"]
        list_abs_path_pict = self.return_list_abs_path_pict(path_folder)
        return list_abs_path_pict
