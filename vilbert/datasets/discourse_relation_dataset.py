# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import math
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td
from vilbert.datasets._image_features_reader import ImageFeaturesH5Reader
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# def iou(anchors, gt_boxes):
#     """
#     anchors: (N, 4) ndarray of float
#     gt_boxes: (K, 4) ndarray of float
#     overlaps: (N, K) ndarray of overlap between boxes and query_boxes
#     """
#     N = anchors.shape[0]
#     K = gt_boxes.shape[0]
#
#     gt_boxes_area = (
#         (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
#     ).reshape(1, K)
#
#     anchors_area = (
#         (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
#     ).reshape(N, 1)
#
#     boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
#     query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)
#
#     iw = (
#         np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
#         - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
#         + 1
#     )
#     iw[iw < 0] = 0
#
#     ih = (
#         np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
#         - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
#         + 1
#     )
#     ih[ih < 0] = 0
#
#     ua = anchors_area + gt_boxes_area - (iw * ih)
#     overlaps = iw * ih / ua
#
#     return overlaps


def deserialize_lmdb(ds):
    return msgpack.loads(
        ds[1],
        raw=False,
        max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN,
    )


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
        self,
        image_feat=None,
        image_target=None,
        caption=None,
        lm_labels=None,
        image_loc=None,
        num_boxes=None,
        overlaps=None,
    ):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            tokens_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            tokens_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.image_feat = image_feat
        self.caption = caption
        # self.lm_labels = lm_labels  # masked words for language model
        self.image_loc = image_loc
        # self.image_target = image_target
        self.num_boxes = num_boxes
        # self.overlaps = overlaps


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
        self,
        input_ids=None,
        input_mask=None,
        segment_ids=None,
        lm_label_ids=None,
        image_feat=None,
        # image_target=None,
        image_loc=None,
        # image_label=None,
        image_mask=None,
        # masked_label=None,
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.lm_label_ids = lm_label_ids
        self.image_feat = image_feat
        self.image_loc = image_loc
        # self.image_label = image_label
        # self.image_target = image_target
        self.image_mask = image_mask
        # self.masked_label = masked_label


# class ConceptCapLoaderTrain(object):
#     """
#     Data loader. Combines a dataset and a sampler, and provides
#     single- or multi-process iterators over the dataset.
#     Arguments:
#         mode (str, required): mode of dataset to operate in, one of ['train', 'val']
#         batch_size (int, optional): how many samples per batch to load
#             (default: 1).
#         shuffle (bool, optional): set to ``True`` to have the data reshuffled
#             at every epoch (default: False).
#         num_workers (int, optional): how many subprocesses to use for data
#             loading. 0 means that the data will be loaded in the main process
#             (default: 0)
#         cache (int, optional): cache size to use when loading data,
#         drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
#             if the dataset size is not divisible by the batch size. If ``False`` and
#             the size of dataset is not divisible by the batch size, then the last batch
#             will be smaller. (default: False)
#         cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
#             to the GPU for you (necessary because this lets us to uint8 conversion on the
#             GPU, which is faster).
#     """
#
#     def __init__(
#         self,
#         corpus_path,
#         tokenizer,
#         bert_model,
#         seq_len,
#         encoding="utf-8",
#         visual_target=0,
#         hard_negative=False,
#         batch_size=512,
#         shuffle=False,
#         num_workers=25,
#         cache=10000,
#         drop_last=False,
#         cuda=False,
#         local_rank=-1,
#         objective=0,
#         visualization=False,
#     ):
#         TRAIN_DATASET_SIZE = 3119449
#
#         if dist.is_available() and local_rank != -1:
#
#             num_replicas = dist.get_world_size()
#             rank = dist.get_rank()
#
#             lmdb_file = os.path.join(
#                 corpus_path, "training_feat_part_" + str(rank) + ".lmdb"
#             )
#         else:
#             lmdb_file = os.path.join(corpus_path, "training_feat_all.lmdb")
#             # lmdb_file = os.path.join(corpus_path, "validation_feat_all.lmdb")
#
#             print("Loading from %s" % lmdb_file)
#
#         ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
#         self.num_dataset = len(ds)
#         ds = td.LocallyShuffleData(ds, cache)
#         caption_path = os.path.join(corpus_path, "caption_train.json")
#         # caption_path = os.path.join(corpus_path, "caption_val.json")
#
#         preprocess_function = BertPreprocessBatch(
#             caption_path,
#             tokenizer,
#             bert_model,
#             seq_len,
#             36,
#             self.num_dataset,
#             encoding="utf-8",
#             visual_target=visual_target,
#             objective=objective,
#         )
#
#         ds = td.PrefetchData(ds, 5000, 1)
#         ds = td.MapData(ds, preprocess_function)
#         # self.ds = td.PrefetchData(ds, 1)
#         ds = td.PrefetchDataZMQ(ds, num_workers)
#         self.ds = td.BatchData(ds, batch_size)
#         # self.ds = ds
#         self.ds.reset_state()
#
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#
#     def __iter__(self):
#
#         for batch in self.ds.get_data():
#             input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, masked_label, image_id = (
#                 batch
#             )
#
#             batch_size = input_ids.shape[0]
#
#             sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
#             sum_count[sum_count == 0] = 1
#             g_image_feat = np.sum(image_feat, axis=1) / sum_count
#             image_feat = np.concatenate(
#                 [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
#             )
#             image_feat = np.array(image_feat, dtype=np.float32)
#
#             g_image_loc = np.repeat(
#                 np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
#             )
#             image_loc = np.concatenate(
#                 [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
#             )
#
#             image_loc = np.array(image_loc, dtype=np.float32)
#             g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
#             image_mask = np.concatenate([g_image_mask, image_mask], axis=1)
#
#             batch = (
#                 input_ids,
#                 input_mask,
#                 segment_ids,
#                 lm_label_ids,
#                 is_next,
#                 image_feat,
#                 image_loc,
#                 image_target,
#                 image_label,
#                 image_mask,
#             )
#
#             yield tuple([torch.tensor(data) for data in batch] + [image_id])
#
#     def __len__(self):
#         return self.ds.size()
#

class DiscourseRelationDataset(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
        self,
        labels_header,
        dataroot,
        tokenizer,
        bert_model,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        batch_size=512,
        shuffle=False,
        num_workers=25,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    ):

        self.seq_len = seq_len
        self.region_len = 101
        # self.labels_header = labels_header
        lmdb_file = os.path.join(dataroot)
        caption_path = os.path.join(dataroot, "captions_all_json.json")
        print("Loading from %s" % lmdb_file)
        ds = ImageFeaturesH5Reader(
            lmdb_file, True
        )

        self.image_reader = ImageFeaturesH5Reader(
            lmdb_file, True
        )
        self.image_name = self.image_reader.keys()
        # ds = td.LMDBSerializer.load(lmdb_file, shuffle=False)
        self.num_dataset = len(ds)
        # preprocess_function = BertPreprocessBatch(
        #     caption_path,
        #     tokenizer,
        #     bert_model,
        #     seq_len,
        #     101,
        #     self.num_dataset,
        #     encoding="utf-8",
        #     visual_target=visual_target,
        #     visualization=visualization,
        #     objective=objective,
        # )
        # self.ds = td.MapData(ds, preprocess_function)
        self.tokenizer = tokenizer
        # self.ds = td.BatchData(ds, batch_size, remainder=True)
        print("man ye kharama ")
        # self.ds.reset_state()
        self.captions = json.load(open(caption_path, "r"))

        self.batch_size = batch_size
        self.num_workers = num_workers
    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()


    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_region_length
    ):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        # image_target = example.image_target
        num_boxes = int(example.num_boxes)
        # overlaps = example.overlaps

        self._truncate_seq_pair(tokens, max_seq_length - 2)

        # tokens, tokens_label = self.random_word(tokens, tokenizer, is_next)
        # image_feat, image_loc, image_label, masked_label = self.random_region(
        #     image_feat, image_loc, num_f, is_next, overlaps
        # )

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = [-1] + tokens_label + [-1]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        segment_ids = [0] * len(tokens)

        input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            # image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        # assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            # lm_label_ids=np.array(lm_label_ids),
            # is_next=np.array(example.is_next),
            image_feat=image_feat,
            # image_target=image_target,
            image_loc=image_loc,
            # image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            # masked_label=masked_label,
        )
        return features
    #todo complete this
    def __getitem__(self, index):

        image_id = self.image_name[index]
        image_feature, num_boxes, image_location, _, _ = self.image_reader[image_id]
        caption = self.captions[image_id.decode()]
        tokens_caption = self.tokenizer.encode(caption)
        tokens = tokens_caption

        cur_example = InputExample(
            image_feat=image_feature,
            # image_target=image_target,
            caption=tokens,
            image_loc=image_location,
            num_boxes=num_boxes,
            # overlaps=overlaps,
        )

        cur_features = self.convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.region_len
        )

        #
        batch = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            # lm_label_ids,
            # is_next,
            cur_features.image_feat,
            cur_features.image_loc,
            # image_target,
            # image_label,
            cur_features.image_mask,
            image_id.decode()
        )
        return (

            batch

        )

    # def __iter__(self):
    #     print("madar begeryad")
    #     for batch in self.ds.get_data():
    #         # input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask, masked_label, image_id = (
    #         #     batch
    #         # )
    #
    #         # input_ids, input_mask, segment_ids, image_feat, image_loc, image_mask, image_id = (
    #         #     batch
    #         # )
    #
    #         input_ids, input_mask, segment_ids, image_feat, image_loc, image_mask, image_id  = (batch)
    #
    #
    #         batch_size = input_ids.shape[0]
    #         # sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
    #         # sum_count[sum_count == 0] = 1
    #         # g_image_feat = np.sum(image_feat, axis=1) / sum_count
    #         # image_feat = np.concatenate(
    #         #     [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
    #         # )
    #         image_feat = np.array(image_feat, dtype=np.float32)
    #
    #         # g_image_loc = np.repeat(
    #         #     np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
    #         # )
    #         # image_loc = np.concatenate(
    #         #     [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
    #         # )
    #
    #         image_loc = np.array(image_loc, dtype=np.float32)
    #         # g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
    #         # image_mask = np.concatenate([g_image_mask, image_mask], axis=1)
    #
    #         batch = (
    #             input_ids,
    #             input_mask,
    #             segment_ids,
    #             # lm_label_ids,
    #             # is_next,
    #             image_feat,
    #             image_loc,
    #             # image_target,
    #             # image_label,
    #             image_mask,
    #         )
    #
    #         yield tuple([torch.tensor(data) for data in batch] + [image_id])

    def __len__(self):
        return len(self.image_reader)


class BertPreprocessBatch(object):
    def  __init__(
        self,
        caption_path,
        tokenizer,
        bert_model,
        seq_len,
        region_len,
        data_size,
        split="Train",
        encoding="utf-8",
        visual_target=0,
        visualization=False,
        objective=0,
    ):
        self.split = split
        self.seq_len = seq_len
        self.region_len = region_len
        self.tokenizer = tokenizer
        self.visual_target = visual_target
        self.num_caps = data_size
        # self.captions = np.load(caption_path, allow_pickle=True)
        self.captions = json.load(open(caption_path, "r"))
        self.visualization = visualization
        self.objective = objective
        self.bert_model = bert_model

    def __call__(self, data):

        # image_feature_wp, image_target_wp, image_location_wp, num_boxes, image_h, image_w, image_id, caption = (
        #     data
        # )

        image_feature, num_boxes, image_location, _, subjects = (data)
        # image_feature = np.zeros((self.region_len, 2048), dtype=np.float32)
        # image_target = np.zeros((self.region_len, 1601), dtype=np.float32)
        # image_location = np.zeros((self.region_len, 5), dtype=np.float32)

        # calculate the IOU here.
        # overlaps = iou(image_location_wp, image_location_wp)

        num_boxes = int(num_boxes)
        # image_feature[:num_boxes] = image_feature_wp
        # image_target[:num_boxes] = image_target_wp
        # image_location[:num_boxes, :4] = image_location_wp

        # image_location[:, 4] = (
        #     (image_location[:, 3] - image_location[:, 1])
        #     * (image_location[:, 2] - image_location[:, 0])
        #     / (float(image_w) * float(image_h))
        # )

        # image_location[:, 0] = image_location[:, 0] / float(image_w)
        # image_location[:, 1] = image_location[:, 1] / float(image_h)
        # image_location[:, 2] = image_location[:, 2] / float(image_w)
        # image_location[:, 3] = image_location[:, 3] / float(image_h)

        if self.visual_target == 0:
            image_feature = copy.deepcopy(image_feature)
            # image_target = copy.deepcopy(image_target)
        else:
            image_feature = copy.deepcopy(image_feature)
            # image_target = copy.deepcopy(image_feature)

        tokens = []
        assert len(subjects) == 1

        for subject in subjects:
            caption = self.captions[subject]
            tokens_caption = self.tokenizer.encode(caption)
            tokens = tokens_caption

        # tokens = np.array(tokens)
        cur_example = InputExample(
            image_feat=image_feature,
            # image_target=image_target,
            caption=tokens,
            image_loc=image_location,
            num_boxes=num_boxes,
            # overlaps=overlaps,
        )
        # transform sample to features
        cur_features = self.convert_example_to_features(
            cur_example, self.seq_len, self.tokenizer, self.region_len
        )

        cur_tensors = (
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            # cur_features.lm_label_ids,
            cur_features.image_feat,
            cur_features.image_loc,
            # cur_features.image_target,
            # cur_features.image_label,
            cur_features.image_mask,
            # cur_features.masked_label,
            subject,
        )
        return cur_tensors


    def convert_example_to_features(
        self, example, max_seq_length, tokenizer, max_region_length
    ):
        """
        """
        image_feat = example.image_feat
        tokens = example.caption
        image_loc = example.image_loc
        # image_target = example.image_target
        num_boxes = int(example.num_boxes)
        # overlaps = example.overlaps

        self._truncate_seq_pair(tokens, max_seq_length - 2)

        # tokens, tokens_label = self.random_word(tokens, tokenizer, is_next)
        # image_feat, image_loc, image_label, masked_label = self.random_region(
        #     image_feat, image_loc, num_f, is_next, overlaps
        # )

        # concatenate lm labels and account for CLS, SEP, SEP
        # lm_label_ids = [-1] + tokens_label + [-1]
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        segment_ids = [0] * len(tokens)

        input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        # input_ids = input_ids[:1] input_ids[1:]
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        # Zero-pad up to the visual sequence length.
        while len(image_mask) < max_region_length:
            image_mask.append(0)
            # image_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # lm_label_ids.append(-1)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        # assert len(lm_label_ids) == max_seq_length
        assert len(image_mask) == max_region_length
        # assert len(image_label) == max_region_length

        features = InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            # lm_label_ids=np.array(lm_label_ids),
            # is_next=np.array(example.is_next),
            image_feat=image_feat,
            # image_target=image_target,
            image_loc=image_loc,
            # image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            # masked_label=masked_label,
        )
        return features

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

# class DiscourseRelationClassification(Dataset):

#     def __init__(
#             self,
#             task,
#             dataroot,
#             annotations_jsonpath,
#             split,
#             image_features_reader,
#             gt_image_features_reader,
#             tokenizer,
#             bert_model,
#             clean_datasets,
#             padding_index=0,
#             max_seq_length=36,
#             max_region_num=101,
#     ):
#         super().__init__()
#         self.split = split
#         # ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
#         # label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
#         # self.ans2label = cPickle.load(open(ans2label_path, "rb"))
#         # self.label2ans = cPickle.load(open(label2ans_path, "rb"))
#         # self.num_labels = len(self.ans2label)
#         self._max_region_num = max_region_num
#         self._max_seq_length = max_seq_length
#         self._image_features_reader = image_features_reader
#         self._tokenizer = tokenizer
#         self._padding_index = padding_index
#
#         clean_train = "_cleaned" if clean_datasets else ""
#
#         if "roberta" in bert_model:
#             cache_path = os.path.join(
#                 dataroot,
#                 "cache",
#                 task
#                 + "_"
#                 + split
#                 + "_"
#                 + "roberta"
#                 + "_"
#                 + str(max_seq_length)
#                 + clean_train
#                 + ".pkl",
#             )
#         else:
#             cache_path = os.path.join(
#                 dataroot,
#                 "cache",
#                 task + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl",
#             )
#         if not os.path.exists(cache_path):
#             self.entries = _load_dataset(dataroot, split, clean_datasets)
#             self.tokenize(max_seq_length)
#             self.tensorize()
#             cPickle.dump(self.entries, open(cache_path, "wb"))
#         else:
#             logger.info("Loading from %s" % cache_path)
#             self.entries = cPickle.load(open(cache_path, "rb"))
#
#     def tokenize(self, max_length=16):
#         """Tokenizes the questions.
#
#         This will add q_token in each entry of the dataset.
#         -1 represent nil, and should be treated as padding_index in embedding
#         """
#         for entry in self.entries:
#             tokens = self._tokenizer.encode(entry["question"])
#             tokens = tokens[: max_length - 2]
#             tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)
#
#             segment_ids = [0] * len(tokens)
#             input_mask = [1] * len(tokens)
#
#             if len(tokens) < max_length:
#                 # Note here we pad in front of the sentence
#                 padding = [self._padding_index] * (max_length - len(tokens))
#                 tokens = tokens + padding
#                 input_mask += padding
#                 segment_ids += padding
#
#             assert_eq(len(tokens), max_length)
#             entry["q_token"] = tokens
#             entry["q_input_mask"] = input_mask
#             entry["q_segment_ids"] = segment_ids
#
#     def tensorize(self):
#
#         for entry in self.entries:
#             question = torch.from_numpy(np.array(entry["q_token"]))
#             entry["q_token"] = question
#
#             q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
#             entry["q_input_mask"] = q_input_mask
#
#             q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
#             entry["q_segment_ids"] = q_segment_ids
#
#             if "test" not in self.split:
#                 answer = entry["answer"]
#                 labels = np.array(answer["labels"])
#                 scores = np.array(answer["scores"], dtype=np.float32)
#                 if len(labels):
#                     labels = torch.from_numpy(labels)
#                     scores = torch.from_numpy(scores)
#                     entry["answer"]["labels"] = labels
#                     entry["answer"]["scores"] = scores
#                 else:
#                     entry["answer"]["labels"] = None
#                     entry["answer"]["scores"] = None
#
#     def __getitem__(self, index):
#         entry = self.entries[index]
#         image_id = entry["image_id"]
#         question_id = entry["question_id"]
#         features, num_boxes, boxes, _ = self._image_features_reader[image_id]
#
#         mix_num_boxes = min(int(num_boxes), self._max_region_num)
#         mix_boxes_pad = np.zeros((self._max_region_num, 5))
#         mix_features_pad = np.zeros((self._max_region_num, 2048))
#
#         image_mask = [1] * (int(mix_num_boxes))
#         while len(image_mask) < self._max_region_num:
#             image_mask.append(0)
#
#         # shuffle the image location here.
#         # img_idx = list(np.random.permutation(num_boxes-1)[:mix_num_boxes]+1)
#         # img_idx.append(0)
#         # mix_boxes_pad[:mix_num_boxes] = boxes[img_idx]
#         # mix_features_pad[:mix_num_boxes] = features[img_idx]
#
#         mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
#         mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
#
#         features = torch.tensor(mix_features_pad).float()
#         image_mask = torch.tensor(image_mask).long()
#         spatials = torch.tensor(mix_boxes_pad).float()
#
#         question = entry["q_token"]
#         input_mask = entry["q_input_mask"]
#         segment_ids = entry["q_segment_ids"]
#
#         co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
#         target = torch.zeros(self.num_labels)
#
#         if "test" not in self.split:
#             answer = entry["answer"]
#             labels = answer["labels"]
#             scores = answer["scores"]
#             if labels is not None:
#                 target.scatter_(0, labels, scores)
#
#         return (
#             features,
#             spatials,
#             image_mask,
#             question,
#             target,
#             input_mask,
#             segment_ids,
#             co_attention_mask,
#             question_id,
#         )
#
#     def __len__(self):
#         return len(self.entries)

if __name__ == '__main__':
    from pytorch_transformers.tokenization_bert import BertTokenizer
    dataroot = '/Users/sinamalakouti/PycharmProjects/vilbert-multi-task/data/discoursedata'


    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased", do_lower_case= True
    )
    bert_model = "bert-base-uncased"
    seq_len = 36
    data = DiscourseRelationDataset(
        "",
        dataroot,
        tokenizer,
        bert_model,
        seq_len,
        encoding="utf-8",
        visual_target=0,
        batch_size=1,
        shuffle=False,
        num_workers=25,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    )

    from torch.utils.data import DataLoader, Dataset, RandomSampler

    train_sampler = RandomSampler(data)

    loader = DataLoader(
        data,
        sampler=train_sampler,
        batch_size=2,
        num_workers=1,
        pin_memory=True,
    )

    for b in loader:
        print(b[-3])
        print(b[-1])
        print("*********************************************")
        # print(b)


    # for b in data:
    #     print(b)

