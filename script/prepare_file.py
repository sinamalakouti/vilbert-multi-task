import os
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np


class Prepare_files():

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.prepare()


    def get_parser(self):
         parser = argparse.ArgumentParser()
         parser.add_argument(
                 "--dir_path", default=None, type=str, help="path of npy files directory"
            )

         return parser

    def prepare(self):
        dir_path = self.args.dir_path
        all_data = []
        for f in listdir(dir_path):
            file_name = f
            file_path = os.path.abspath(f)
            data = np.load(file_path, allow_pickle=True)
            data = data.ravel()
            temp = data[0]
            temp['file_name'] = file_name
            temp['file_path'] = file_path
            all_data.append(temp)
            np.save(file_path, data)
        np.save(dir_path +"/all_data.npym", all_data)


if __name__ == '__main__':
    obj = Prepare_files()
