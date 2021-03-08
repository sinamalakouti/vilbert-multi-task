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
import torch.distributed as dist


def create_dataset_files(name, dataset_path):
    annotations = []
    captions = []
    images = []
    labels = ['Visible', 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant', 'Other', 'When', 'Where', 'How',
              'Comment']
    data = pd.read_csv(dataset_path + 'train.csv', sep=',')
    counter = 0
    for index, row in data.iterrows():

        if not image_found(index, row['url']):
            continue
        captions.append(row['caption'])
        images.append(row['url'])
        instance_annotations = []
        for l in labels:
            instance_annotations.append(row[l])
        annotations.append(instance_annotations)

    annotations = np.array(annotations)
    df_annotations = pd.DataFrame(annotations, columns=list(labels))

    captions = np.array(captions)
    images = np.array(images)
    # np.save('all_annotations.npy', annotations)
    df_annotations.to_csv(dataset_path + 'all_annotations_{}.csv'.format(name))
    np.save(dataset_path + 'all_captions_{}.npy'.format(name), captions)
    np.save(dataset_path + 'all_images_{}.npy'.format(name), images)

    return annotations, captions, images


#

def image_found(ind, pic_url):
    image_formats = ("image/png", "image/jpeg", "image/jpg")
    try:
        response = requests.get(pic_url, stream=True, timeout=60)
        if response.headers["content-type"] in image_formats:
            return True
        else:
            print(pic_url)
            print("image with index: {} is not found \n".format(ind))
            return False
    except:
        print(pic_url)
        print("image with index: {} is not found \n".format(ind))
        return False


def download_images(images_url, dir_url, output_path, train_test):

    labels_headers = ["Visible", 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant']

    all_captions = np.load(dir_url + "all_captions_{}.npy".format(train_test), allow_pickle=True)
    all_captions = all_captions.ravel()
    all_targets = pd.read_csv( dir_url + "all_annotations_{}.csv".formats(train_test), index_col= 0)
    print(len(all_captions))
    print(len(images_url))
    captions = {}
    targets = {}
    for ind, pic_url in enumerate(images_url):
        # pic_url = images_url[10]
        # print(pic_url)
        image_formats = ("image/png", "image/jpeg", "image/jpg")

        with open(output_path + '/pic{}.jpg'.format(ind), 'wb') as handle:
            try:
                response = requests.get(pic_url, stream=True, timeout=60)
                if response.headers["content-type"] in image_formats:
                    None
                else:
                    print("image with index: {} is not found \n".format(ind))
                    print(False)
                captions['pic{}'.format(ind)] = all_captions[ind]
                targets['pic{}'.format(ind)] = {}
                for l in labels_headers:
                    targets['pic{}'.format(ind)][l] = str(all_targets.at[ind,l])

                for block in response.iter_content(1024):
                    if not block:
                        break

                    handle.write(block)
            except:
                print(images_url[ind])
                print(response)
                print(all_captions[ind])
                print("[INFO] error downloading {}...skipping".format(ind))
        if ind == 10:
            break
    with open( dir_url + 'captions_all_json.json', 'w') as outfile:
        json.dump(captions, outfile)

    with open( dir_url +  'all_targets_json.json', 'w') as outfile:
        json.dump(targets, outfile)


def LoadDatasets(args, task_cfg, ids, split="trainval"):
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case=True
    )

    task_feature_reader1 = {}
    task_feature_reader2 = {}
    for i, task_id in enumerate(ids):
        task = "TASK" + task_id + "1"
        if task_cfg[task]["features_h5path1"] not in task_feature_reader1:
            task_feature_reader1[task_cfg[task]["features_h5path1"]] = None
        if task_cfg[task]["features_h5path2"] not in task_feature_reader2:
            task_feature_reader2[task_cfg[task]["features_h5path2"]] = None

    # initilzie the feature reader
    # for features_h5path in task_feature_reader1.keys():
    #     if features_h5path != "":
    #         task_feature_reader1[features_h5path] = ImageFeaturesH5Reader(
    #             features_h5path, args.in_memory
    #         )
    # for features_h5path in task_feature_reader2.keys():
    #     if features_h5path != "":
    #         task_feature_reader2[features_h5path] = ImageFeaturesH5Reader(
    #             features_h5path, args.in_memory
    #         )
    #
    # task_datasets_train = {}
    # task_datasets_val = {}
    # task_dataloader_train = {}
    # task_dataloader_val = {}
    # task_ids = []
    # task_batch_size = {}
    # task_num_iters = {}
    #
    # for i, task_id in enumerate(ids):
    #     task = "TASK" + task_id
    #     task_name = task_cfg[task]["name"]
    #     task_ids.append(task)
    #     batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
    #     num_workers = args.num_workers
    #     if args.local_rank != -1:
    #         batch_size = int(batch_size / dist.get_world_size())
    #         num_workers = int(num_workers / dist.get_world_size())
    #
    #     # num_workers = int(num_workers / len(ids))
    #     logger.info(
    #         "Loading %s Dataset with batch size %d"
    #         % (task_cfg[task]["name"], batch_size)
    #     )
    #
    #     task_datasets_train[task] = None
    #     if "train" in split:
    #         task_datasets_train[task] = DatasetMapTrain[task_name](
    #             task=task_cfg[task]["name"],
    #             dataroot=task_cfg[task]["dataroot"],
    #             annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
    #             split=task_cfg[task]["train_split"],
    #             image_features_reader=task_feature_reader1[
    #                 task_cfg[task]["features_h5path1"]
    #             ],
    #             gt_image_features_reader=task_feature_reader2[
    #                 task_cfg[task]["features_h5path2"]
    #             ],
    #             tokenizer=tokenizer,
    #             bert_model=args.bert_model,
    #             clean_datasets=args.clean_train_sets,
    #             padding_index=0,
    #             max_seq_length=task_cfg[task]["max_seq_length"],
    #             max_region_num=task_cfg[task]["max_region_num"],
    #         )
    #
    #     task_datasets_val[task] = None
    #     if "val" in split:
    #         task_datasets_val[task] = DatasetMapTrain[task_name](
    #             task=task_cfg[task]["name"],
    #             dataroot=task_cfg[task]["dataroot"],
    #             annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
    #             split=task_cfg[task]["val_split"],
    #             image_features_reader=task_feature_reader1[
    #                 task_cfg[task]["features_h5path1"]
    #             ],
    #             gt_image_features_reader=task_feature_reader2[
    #                 task_cfg[task]["features_h5path2"]
    #             ],
    #             tokenizer=tokenizer,
    #             bert_model=args.bert_model,
    #             clean_datasets=args.clean_train_sets,
    #             padding_index=0,
    #             max_seq_length=task_cfg[task]["max_seq_length"],
    #             max_region_num=task_cfg[task]["max_region_num"],
    #         )
    #
    #     task_num_iters[task] = 0
    #     task_batch_size[task] = 0
    #     if "train" in split:
    #         if args.local_rank == -1:
    #             train_sampler = RandomSampler(task_datasets_train[task])
    #         else:
    #             # TODO: check if this works with current data generator from disk that relies on next(file)
    #             # (it doesn't return item back by index)
    #             train_sampler = DistributedSampler(task_datasets_train[task])
    #
    #         task_dataloader_train[task] = DataLoader(
    #             task_datasets_train[task],
    #             sampler=train_sampler,
    #             batch_size=batch_size,
    #             num_workers=num_workers,
    #             pin_memory=True,
    #         )
    #
    #         task_num_iters[task] = len(task_dataloader_train[task])
    #         task_batch_size[task] = batch_size
    #
    #     if "val" in split:
    #         task_dataloader_val[task] = DataLoader(
    #             task_datasets_val[task],
    #             shuffle=False,
    #             batch_size=batch_size,
    #             num_workers=2,
    #             pin_memory=True,
    #         )
    #
    # return (
    #     task_batch_size,
    #     task_num_iters,
    #     task_ids,
    #     task_datasets_train,
    #     task_datasets_val,
    #     task_dataloader_train,
    #     task_dataloader_val,
    # )




def create_train_test(datapath):
    data = pd.read_csv(datapath, sep='\t')
    labels_headers = ["Visible", 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant']

    X = [i for i in range(data.shape[0])]
    X = np.array(X)
    y = data[labels_headers].values

    (X_train, y_train), (X_test, y_test) = train_test_split(X,y)
    train = data.iloc[X_train]
    test = data.iloc[X_test]
    train.to_csv('train.csv')
    test.to_csv('test.csv')






def train_test_split(X, y, test_size =0.33):


    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.33, random_state=0)

    for train_index, test_index in msss.split(X, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
    print("here")
    # create_dataset_files('train', '/Users/sinamalakouti/PycharmProjects/vilbert-multi-task/data/discoursedata/train/')
    # create_train_test('/Users/sinamalakouti/PycharmProjects/DiscourseRelationProject/data/dataset123/data-both-04-08-cleaned.tsv')
    # images = np.load("../data/discoursedata/test/all_images_test.npy").ravel()
    # print(images)
    # bert_model = "bert-base-uncased"
    # bert_weight_name = json.load(
    #     open("./../config/" + bert_model + "_weight_name.json", "r")
    # )
    # task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = LoadDatasets(
    #     '', "task_cfg", ""
    # )

    # print(an.shape)
    # print(cap.shape)
    # print(images.shape)
    # print(images)
    images = np.load("../data/discoursedata/trian/all_images.npy").ravel()
    download_images(images,"../data/discoursedata/train/", "../data/discoursedata/train/images",'test')
