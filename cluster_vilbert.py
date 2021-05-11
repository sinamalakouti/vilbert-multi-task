# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import numpy as np
import math

from tensorboardX import SummaryWriter
from tqdm import tqdm
from bisect import bisect
import yaml
from easydict import EasyDict as edict

import pdb
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from vilbert.datasets.discourse_relation_dataset import DiscourseRelationDataset
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader, Dataset, RandomSampler

from pytorch_transformers.optimization import (
    AdamW,
    WarmupConstantSchedule,
    WarmupLinearSchedule,
)

from vilbert.optimization import RAdam
from vilbert.task_utils import (
    LoadDatasets,
    LoadLosses,
    ForwardModelsTrain,
    ForwardModelsVal,
)
# from torch.optim.lr_scheduler import (
#     LambdaLR,
#     ReduceLROnPlateau,
#     CosineAnnealingLR,
#     CosineAnnealingWarmRestarts,
# )

import vilbert.utils as utils
import torch.distributed as dist

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import numpy as np
def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    batch_size = 64
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
    )
    parser.add_argument(
        "--output_dir",
        default="save",
        type=str,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        default="config/bert_base_6layer_6conect.json",
        type=str,
        help="The config file which specified the model details.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=int,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--train_iter_multiplier",
        default=1.0,
        type=float,
        help="multiplier for the multi-task training.",
    )
    parser.add_argument(
        "--train_iter_gap",
        default=4,
        type=int,
        help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--do_lower_case",
        default=True,
        type=bool,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
             "0 (default value): dynamic loss scaling.\n"
             "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--in_memory",
        default=False,
        type=bool,
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--optim", default="AdamW", type=str, help="what to use for the optimization."
    )
    parser.add_argument(
        "--tasks", default="0", type=str, help="discourse : TASK0"
    )
    parser.add_argument(
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument(
        "--vision_scratch",
        action="store_true",
        help="whether pre-trained the image or not.",
    )
    parser.add_argument(
        "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
    )
    parser.add_argument(
        "--lr_scheduler",
        default="mannul",
        type=str,
        help="whether use learning rate scheduler.",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="whether use single stream baseline."
    )
    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--clean_train_sets",
        default=True,
        type=bool,
        help="whether clean train sets for multitask data.",
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )
    parser.add_argument(
        "--task_specific_tokens",
        action="store_true",
        default=False,
        help="whether to use task specific tokens for the multi-task learning.",
    )

    # todo
    args = parser.parse_args()
    with open("vilbert_tasks.yml", "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    task_names = []
    task_lr = []
    task_id = 1
    for i, task_id in enumerate(args.tasks.split("-")):
        task_id = str(1)
        task = "TASK" + task_id
        name = task_cfg[task]["name"]
        task_names.append(name)
        task_lr.append(task_cfg[task]["lr"])
    base_lr = min(task_lr)
    loss_scale = {}
    for i, task_id in enumerate(args.tasks.split("-")):
        task = "TASK" + task_id
        loss_scale[task] = task_lr[i] / base_lr

    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""
    timeStamp = (
            "-".join("discourse")
            + "_"
            + args.config_file.split("/")[1].split(".")[0]
            + prefix
    )
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(
        open("config/" + args.bert_model + "_weight_name.json", "r")
    )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 3
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    # task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = LoadDatasets(
    #     args, task_cfg, args.tasks.split("-"),'train'
    # )
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    labels = ["Visible", 'Subjective', 'Action', 'Story', 'Meta', 'Irrelevant', 'Other']
    train_dataset = DiscourseRelationDataset(
        labels,
        task_cfg[task]["dataroot"],
        tokenizer,
        args.bert_model,
        task_cfg[task]["max_seq_length"],
        encoding="utf-8",
        visual_target=0,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    )

    train_sampler = RandomSampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size= batch_size,
        num_workers=0,
        pin_memory=True,
    )
    # for i in train_loader:
    #     print("hello")
    # todo task_ids , task_num_tiers
    task_ids = ['TASK0']
    task_num_iters = [100]
    task_batch_size = task_cfg['TASK0']["batch_size"]

    print("task_batch_size")
    print(task_batch_size)
    logdir = os.path.join(savePath, "logs")
    tbLogger = utils.tbLogger(
        logdir,
        savePath,
        task_names,
        task_ids,
        task_num_iters,
        args.gradient_accumulation_steps,
    )

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if args.task_specific_tokens:
        print("*********** config.task_specific_tokens = True ************")
        config.task_specific_tokens = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    num_train_optimization_steps = 10
    num_labels = len(labels)
    if args.dynamic_attention:
        config.dynamic_attention = True
    if "roberta" in args.bert_model:
        config.model = "roberta"

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )
    model.double()
    model = model.to(device)
    task_losses = LoadLosses(args, task_cfg, args.tasks.split("-"))

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if "embeddings" in name:
                bert_weight_name_filtered.append(name)
            elif "encoder" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    optimizer_grouped_parameters = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if "vil_" in key:
                lr = 1e-4
            else:
                if args.vision_scratch:
                    if key[12:] in bert_weight_name:
                        lr = base_lr
                    else:
                        lr = 1e-4
                else:
                    lr = base_lr
            if any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.0}
                ]
            if not any(nd in key for nd in no_decay):
                optimizer_grouped_parameters += [
                    {"params": [value], "lr": lr, "weight_decay": 0.01}
                ]

    if default_gpu:
        print(len(list(model.named_parameters())), len(optimizer_grouped_parameters))

    if args.optim == "AdamW":
        optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr, correct_bias=False, weight_decay=1e-4)
    elif args.optim == "RAdam":
        optimizer = RAdam(optimizer_grouped_parameters, lr=base_lr, weight_decay=1e-4)


    startIterID = 0
    global_step = 0
    start_epoch = 0

    if args.resume_file != "" and os.path.exists(args.resume_file):
        checkpoint = torch.load(args.resume_file, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        # warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) + 1
        task_stop_controller = checkpoint["task_stop_controller"]
        tbLogger = checkpoint["tb_logger"]
        del checkpoint

    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model, delay_allreduce=True)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if default_gpu:
        print("***** Running training *****")
        print("  Num Iters: ", task_num_iters)
        print("  Batch size: ", batch_size)
        print("  Num steps: %d" % num_train_optimization_steps)

    target_path = os.path.join(task_cfg[task]["dataroot"], "all_targets_json.json")
    all_targets = json.load(open(target_path, "r"))
    model = model.to(device)
    print(next(model.parameters()).is_cuda)
    kmeans = MiniBatchKMeans(n_clusters=num_labels,random_state = 0,batch_size = batch_size)
    with torch.no_grad():
        model.eval()
        print("********* training *************")
        for step, batch in enumerate(train_loader):
            print("##########  STEP @ {} ##########".format(step))
            batch = tuple(t.to(device=device, non_blocking=True) if type(t) == torch.Tensor else t for t in batch)
            input_ids, input_mask, segment_ids, image_feat, image_loc, image_mask, image_id = (batch)
            true_targets = []
            for id in image_id:
                true_targets.append(np.fromiter(all_targets[id].values(), dtype=np.double))
            true_targets = torch.from_numpy(np.array(true_targets))
            true_targets = true_targets.to(device)
            model.double()
            model = model.to(device)
            pooled_output, discourse_prediction, vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ \
                = model(
                True,
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask,
            )
            kmeans.partial_fit(pooled_output.to('cpu'))


        evaluate(model, device, task_cfg, tokenizer, args, labels)




def evaluate(model,kmeans, device, task_cfg, tokenizer, args, labels):
    model.eval()
    task = "TASK0"
    target_path = os.path.join(task_cfg[task]["test_dataroot"], "all_targets_json.json")
    batch_size = 64
    test_dataset = DiscourseRelationDataset(
        labels,
        task_cfg[task]["test_dataroot"],
        tokenizer,
        args.bert_model,
        task_cfg[task]["max_seq_length"],
        encoding="utf-8",
        visual_target=0,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        cache=5000,
        drop_last=False,
        cuda=False,
        objective=0,
        visualization=False,
    )

    all_targets = json.load(open(target_path, "r"))

    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(
        test_dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    cluster_label_map = np.zeros(len(labels), len(labels))
    with torch.no_grad():
        for batch in test_loader:
            batch = tuple(t.to(device=device, non_blocking=True) if type(t) == torch.Tensor else t for t in batch)
            input_ids, input_mask, segment_ids, image_feat, image_loc, image_mask, image_id = (batch)
            true_targets = []
            for id in image_id:
                true_targets.append(np.fromiter(all_targets[id].values(), dtype=np.double))
            true_targets = torch.from_numpy(np.array(true_targets))
            true_targets = true_targets.to(device)
            model.double()
            model = model.to(device)
            pooled_output, discourse_prediction, vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ \
                = model(
                True,
                input_ids,
                image_feat,
                image_loc,
                segment_ids,
                input_mask,
                image_mask
            )
            preds = kmeans.predict(pooled_output.to('cpu'))

            for clstr_indx, y_true in zip(preds, true_targets):
                cluster_label_map[clstr_indx] += y_true
        print("********** clstr labels is    *******")
        print(cluster_label_map)



if __name__ == "__main__":
    main()
