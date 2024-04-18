
import os
import json
import torch
import argparse
import numpy as np

import utils
from utils import accuracy
from collections import defaultdict
from monai.metrics import compute_roc_auc
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

def binary2multi(input_tensor):
    # input_tensor: [batch]
    if len(input_tensor.shape) == 2 and input_tensor.shape[1] == 1:
        return torch.cat([1.0 - input_tensor, input_tensor], dim=1)
    elif len(input_tensor.shape) == 1:
        return torch.cat([1.0 - input_tensor.unsqueeze(dim=1), input_tensor.unsqueeze(dim=1)], dim=1)
    else:
        raise NotImplementedError

def compute_cls_metrics(targets, preds):
    # compute the classification metrics
    # 'acc', 'auc', 'f1', 'precision', 'recall'
    # targets: [N], preds: [N] in [0,1] for binary classification
    # targets: [N, C], preds: [N] in [0, C-1] for multi-classification task
    if (targets==0).sum() == 0: # without negative samples
        # ap = average_precision_score(targets, preds > 0.5, average='macro')
        targets_th, preds_th = torch.from_numpy(targets), torch.from_numpy(preds)
        acc = accuracy(binary2multi(preds_th), targets_th, topk=(1,))
        return {'acc':round(acc[0].item(), 4)}

    num_class = preds.shape[1] if len(preds.shape) > 1 else 1
    if num_class == 1:
        pcf = precision_recall_fscore_support(targets, preds > 0.5, average='macro')
        precision = pcf[0]
        recall = pcf[1]
        f1 = pcf[2]
        ap = average_precision_score(targets, preds > 0.5, average='macro')
        # ap = average_precision_score(targets, preds, average='macro')
        targets_th, preds_th = torch.from_numpy(targets), torch.from_numpy(preds)
        auc = compute_roc_auc(preds_th, targets_th, average="macro")
        acc, = accuracy(binary2multi(preds_th), targets_th, topk=(1,))
        acc = acc.item()
        # the acc metric in RETFound
        # acc = utils.compute_acc(targets_th, binary2multi(preds_th)) * 100
    else: # for multi-classes
        pcf = precision_recall_fscore_support(targets.argmax(axis=1), preds.argmax(axis=1), average='macro')
        precision = pcf[0]
        recall = pcf[1]
        f1 = pcf[2]
        targets_th, preds_th = torch.from_numpy(targets), torch.from_numpy(preds)
        auc = compute_roc_auc(preds_th, targets_th, average="macro")
        acc, = accuracy(preds_th, targets_th.argmax(dim=1), topk=(1,))
        acc = acc.item()
        # the acc metric in RETFound
        # acc = utils.compute_acc(targets_th.argmax(dim=1), preds_th) * 100
        ap = 0

    metrics = {
        "acc": round(acc, 4),
        "auc":round(auc,4),
        "f1":round(f1.item(),4),
        "precision":round(precision.item(),4),
        "recall":round(recall.item(),4),
        "ap":round(ap, 4)
    }
    return metrics
def performance_single_cls(single_results_path):
    assert os.path.exists(single_results_path), f"cannot find {single_results_path}"
    print(f"read {single_results_path} to get results")
    with open(single_results_path, 'r') as json_file:
        vfm_results:dict = json.load(json_file) # key denotes image path

    targets, preds = [], []
    for img_key, img_val in vfm_results.items():
        target = np.array(img_val['gt'])
        pred = np.array(img_val['pred'])
        if len(target) == 1 and target[0] < 2: # binary
            targets.append(target)
            preds.append(pred)
        else: # multi-class
            assert len(target) > 1, f'Only support one-hot target'
            target = target[np.newaxis, :]
            pred = pred[np.newaxis, :]
            targets.append(target)
            preds.append(pred)
    targets = np.concatenate(targets, axis=0)
    preds = np.concatenate(preds, axis=0)
    print(f"there are {targets.shape[0]} image files")
    # compute the metrics
    metrics: dict = compute_cls_metrics(targets, preds)
    # print(metrics)
    print_table(metrics)

def print_table(metrics:dict):
    acc = round(metrics.get('acc', 0), 2)
    auc = round(metrics.get('auc', 0)*100.0,2)
    f1 = round(metrics.get('f1', 0) * 100.0, 2)
    precision = round(metrics.get('precision', 0)*100.0, 2)
    recall = round(metrics.get('recall', 0) * 100.0, 2)
    ap = round(metrics.get('ap', 0)*100.0, 2)
    print(f"|acc|auc|f1|precision|recall|ap|")
    print(f"|{acc}|{auc}|{f1}|{precision}|{recall}|{ap}|")
