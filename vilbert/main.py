
import argparse
import json
import logging
import os
import random
from io import open
import math
import sys
import pandas as pd
import requests

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

import utils as utils
from datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
import torch.distributed as dist


def create_dataset_files(dataset_path):
    annotations = []
    captions =  []
    images =  []
    labels = ['Visible','Subjective','Action','Story','Meta','Irrelevant','Other','When','Where','How','Comment']
    data = pd.read_csv(dataset_path, sep='\t')
    for index, row in data.iterrows():
        if not image_found(index,   row['url']):
            continue
        captions.append(row['caption'])
        images.append(row['url'])
        instance_annotations = []
        for l in labels:
            instance_annotations.append(row[l])
        annotations.append(instance_annotations)

    annotations = np.array(annotations)
    df_annotations = pd.DataFrame(annotations, columns= list(labels))

    captions = np.array(captions)
    images = np.array(images)
    # np.save('all_annotations.npy', annotations)
    df_annotations.to_csv('all_annotations.csv')
    np.save('all_captions.npy', captions)
    np.save('all_images.npy', images)


    return annotations, captions, images
#

def image_found(ind, pic_url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    try:
        response = requests.get(pic_url, stream=True, timeout=60)
        if response.headers["content-type"] in image_formats:
            return True
        else:
            print("image with index: {} is not found \n".format(ind))
            return False
    except:
        print("image with index: {} is not found \n".format(ind))
        return False


def download_images(images_url, output_path):

    for ind, pic_url in enumerate(images_url):
        # pic_url = images_url[10]
        # print(pic_url)
        image_formats = ("image/png", "image/jpeg", "image/jpg")
        with open(output_path+'/pic{}.jpg'.format(ind), 'wb') as handle:
            try:
                response = requests.get(pic_url, stream=True, timeout=60)
                if response.headers["content-type"] in image_formats:
                    None
                else:
                    print("image with index: {} is not found \n".format(ind))
                    print(False)

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)
            except:
                print("[INFO] error downloading {}...skipping".format(ind))


if __name__ == '__main__':
    print("here")
    # an , cap , images = create_dataset_files('/Users/sinamalakouti/PycharmProjects/DiscourseRelationProject/data/dataset123/data-both-04-08-cleaned.tsv')
    images = np.load("all_images.npy")
    # print(an.shape)
    # print(cap.shape)
    print(images.shape)
    # print(images)
    
    download_images(images,"../data/images")