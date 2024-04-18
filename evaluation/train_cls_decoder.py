# Train a decoder for the single-modal classification task
import sys
sys.path.append('./')
import os
import argparse
import json
import copy
import torch
import torch.backends.cudnn as cudnn
import utils
import models
import numpy as np
from models.head import ClsHead
from collections import defaultdict
from pathlib import Path
from torch import nn
# from torchvision import transforms as pth_transforms
from torchvision.transforms import InterpolationMode
from torchvision import datasets
import transforms as self_transforms
# from loader import ImageFolder
from dataset import ClsImgs, ImageFolderDataset
from sklearn.metrics import precision_recall_fscore_support
from monai.metrics import compute_roc_auc
from evaluation_funcs import performance_single_cls

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    args.output_dir = os.path.join(args.output_dir, args.name) # set new output dir with name
    if not os.path.exists(args.output_dir):
        print(f"Create the output_dir: {args.output_dir}")
        os.makedirs(args.output_dir)

    # fix the seed for reproducibility
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")

    train_transform = self_transforms.Compose([
        # self_transforms.RandomResizedCrop(args.input_size),
        self_transforms.Resize(size=(args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
        self_transforms.RandomHorizontalFlip(),
        self_transforms.RandomVerticalFlip(),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    val_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size), interpolation=InterpolationMode.BICUBIC),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    # prepare teh dataset
    if args.dataset_format == 'vfm':
        dataset_train = ClsImgs(root=args.data_path, split='training', transform=train_transform)
        dataset_val = ClsImgs(root=args.data_path, split='test', transform=val_transform) # set split='val' if there are val set in datasets
    elif args.dataset_format == 'ImageNet':
        dir_train = os.path.join(args.data_path, 'train')
        dataset_train = ImageFolderDataset(dir_train, train_transform)
        dir_val = os.path.join(args.data_path, 'test')
        dataset_val = ImageFolderDataset(dir_val, val_transform)
    else:
        raise NotImplementedError

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    if sys.gettrace():
        print(f"Now in debug mode")
        args.num_workers = 0
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    model = models.__dict__[args.arch](
        img_size=[args.input_size],
        patch_size=args.patch_size,
        num_classes=0,
        use_mean_pooling=False)
    embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    # define the decoder
    # linear_classifier = ClsHead(embed_dim=embed_dim * args.n_last_blocks, num_classes=args.num_labels, layers=3)

    if args.avgpool_patchtokens == 0:
        linear_classifier = ClsHead(embed_dim=embed_dim*args.n_last_blocks, num_classes=args.num_labels, layers=3)
    elif args.avgpool_patchtokens == 1:
        linear_classifier = ClsHead(embed_dim=embed_dim, num_classes=args.num_labels, layers=3)

    print(utils.get_parameter_number(linear_classifier)) # compute the parameters
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # set optimizer
    parameters = linear_classifier.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        betas=(0.9, 0.999), weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_f1": 0.}
    if args.load_from: # load the weights to re-start training the model
        utils.restart_from_checkpoint(
            args.load_from,
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler)

    if args.load_from and args.test:
        # test the trained decoder
        if args.dataset_format == 'vfm':
            dataset_test = ClsImgs(root=args.data_path, split='test', transform=val_transform)
        elif args.dataset_format == 'ImageNet':
            dir_test = os.path.join(args.data_path, 'test')
            dataset_test = ImageFolderDataset(dir_test, val_transform)
        else:
            raise NotImplementedError
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        # save the results in json file
        dst_json_path = os.path.join(args.output_dir, f"{args.name}_{args.modality}_results.json")
        model.eval()
        linear_classifier.eval()
        test_stats = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, dst_json_path)
        print('Compute performance of single task: ')
        performance_single_cls(dst_json_path)
        exit()

    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_f1"]

    # the main loop
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.eval()

        linear_classifier.train()
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifier.eval()
            test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)

            all_acc = []
            all_auc = []
            all_f1 = []
            all_precision = []
            all_recall = []
            for key, val in test_stats.items():
                if 'acc' in key:
                    all_acc.append(val)
                elif 'auc' in key:
                    all_auc.append(val)
                elif 'f1' in key:
                    all_f1.append(val)
                elif 'precision' in key:
                    all_precision.append(val)
                elif 'recall' in key:
                    all_recall.append(val)
            all_acc = np.asarray(all_acc).mean()
            all_auc = np.asarray(all_auc).mean()
            all_f1 = np.asarray(all_f1).mean()
            all_precision = np.asarray(all_precision).mean()
            all_recall = np.asarray(all_recall).mean()

            print(f"Mean acc at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_acc:.4f}")
            print(f"Mean auc at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_auc:.4f}")
            print(f"Mean F1 at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_f1:.4f}")
            print(f"Mean Precision at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_precision:.4f}")
            print(f"Mean Recall at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_recall:.4f}")

            log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (test_stats["f1"] >= best_f1):
                # always only save the best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_f1": test_stats["f1"],
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(args.checkpoint_key)))

            best_f1 = max(best_f1, test_stats["f1"])
            print(f'Max F1 so far: {best_f1:.4f}%')
    print("Training of the supervised linear classifier on frozen features completed.\nAnd the best F1-score: {f1:.4f}".format(f1=best_f1))


def train(model, linear_classifier, optimizer, loader, epoch, n, avg_pool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target, extras) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long() # [B]
        if len(target.shape) == 2:
            target = target.squeeze()

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avg_pool == 0:
                output = [x[:, 0] for x in intermediate_output]  # only retain CLS tokens
            elif avg_pool == 1:
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]  # only patch tokens
            output = torch.cat(output, dim=-1)

        output = linear_classifier(output)

        # compute cross entropy loss
        num_class = output.shape[1]
        if num_class > 1:  # for multi-class case
            loss = nn.CrossEntropyLoss()(output, target)
        else: # for binary class case
            loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avg_pool, dst_json_path = None):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    targets, preds, img_paths = [], [], []
    for inp, target, extras in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True).long() # [B]
        if len(target.shape) == 2:
            target = target.squeeze()

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(inp, n)
            if avg_pool == 0:
                output = [x[:, 0] for x in intermediate_output]  # only retain CLS tokens
            elif avg_pool == 1:
                output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]  # only patch tokens
            output = torch.cat(output, dim=-1)

        output = linear_classifier(output)
        num_class = output.shape[1]
        if num_class > 1:  # multi-class case
            loss = nn.CrossEntropyLoss()(output, target)
        else:
            loss = nn.BCEWithLogitsLoss()(output.squeeze(dim=1), target.float())

        # save results
        if num_class > 1:  # multi-class
            preds.append(output.softmax(dim=1).detach().cpu())
            # targets.append(torch.nn.functional.one_hot(target, num_class)) # convert the target into one-hot format
            targets.append(target.detach().cpu())
        else:  # num_labels=1 binary classification
            targets.append(target.detach().cpu())  # [[B, 1]]
            preds.append(output.detach().cpu().sigmoid())
        img_paths += extras['img_path']


        metric_logger.update(loss=loss.item())

    preds_all = torch.cat(preds, dim=0) 
    targets_all = torch.cat(targets, dim=0)
    num_class = preds_all.shape[1]
    if num_class == 1:
        acc1, = utils.accuracy(binary2multi(preds_all.squeeze()), targets_all, topk=(1,))
        acc1 = acc1.item()
        auc = compute_roc_auc(preds_all.squeeze(dim=1), targets_all, average="none")
        pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.cpu().numpy() > 0.5, average='macro')
        precision = pcf[0]
        recall = pcf[1]
        f1 = pcf[2]
    else: # for multi-class cases
        acc1, = utils.accuracy(preds_all, targets_all, topk=(1,))
        acc1 = acc1.item()
        target_onehot = torch.nn.functional.one_hot(targets_all, num_class).float()
        auc_dr_grading_list = compute_roc_auc(preds_all, target_onehot, average="none")
        auc = np.array(auc_dr_grading_list).mean()  # only positives
        pcf = precision_recall_fscore_support(targets_all.numpy(), preds_all.argmax(dim=1).numpy(), average=None)
        precision = pcf[0][1:].mean() # only positive class
        recall = pcf[1][1:].mean()
        f1 = pcf[2][1:].mean()


        # the acc in retfound metrics
        # acc1 = utils.compute_acc(targets_all, preds_all) # targets_all: non one-hot, preds_all: one-hot
    batch_size = 1
    
    metric_logger.meters['acc1'].update(acc1, n=batch_size)
    metric_logger.meters['auc'].update(auc, n=batch_size)
    metric_logger.meters['precision'].update(precision.item(), n=batch_size)
    metric_logger.meters['recall'].update(recall.item(), n=batch_size)
    metric_logger.meters['f1'].update(f1.item(), n=batch_size)

    print(
        '* Acc@1 {top1.global_avg:.4f} loss {losses.global_avg:.4f} AUC {auc.global_avg:.4f} Pre {precision.global_avg:.4f} Recall {recall.global_avg:.4f} F1 {f1.global_avg:.4f}'
        .format(top1=metric_logger.acc1, losses=metric_logger.loss, auc=metric_logger.auc,
                precision=metric_logger.precision, recall=metric_logger.recall, f1=metric_logger.f1))

    # save predictions as json file
    if dst_json_path is not None:
        save_preds(targets, preds, img_paths, dst_json_path)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def save_preds(targets, preds, paths:list,dst_json_path):
    # save the predictions into json file
    results = defaultdict()

    targets_th = torch.cat(targets, dim=0) # [N]
    preds_th = torch.cat(preds, dim=0) # [N, C]
    num_class = preds_th.shape[1]
    if num_class > 1:
        targets_th = torch.nn.functional.one_hot(targets_th, num_class).float() # convert to one-hot


    num = preds_th.shape[0]
    for idx in range(num):
        img_path = paths[idx] # e.g. Sjchoi86/test/Glaucoma_025.png
        # key = img_path.split('.')[0]
        key = img_path
        if key not in results.keys():
            if len(preds_th[idx]) > 1:
                results[key] = {'gt': targets_th[idx].tolist(),
                                'pred': preds_th[idx].tolist()}
            else:
                results[key] = {'gt': [targets_th[idx].item()],
                            'pred':[preds_th[idx].item()]}

    print(f"there are {len(results.keys())} images in test set")
    with open(dst_json_path, "w") as f:
        f.write(json.dumps(results, indent=4))
    print(f"write {dst_json_path} success. ")


def binary2multi(input_tensor):
    # input_tensor: [batch]
    if len(input_tensor.shape) == 2 and input_tensor.shape[1] == 1:
        return torch.cat([1.0 - input_tensor, input_tensor], dim=1)
    elif len(input_tensor.shape) == 1:
        return torch.cat([1.0 - input_tensor.unsqueeze(dim=1), input_tensor.unsqueeze(dim=1)], dim=1)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser('training a classification decoder on pretrained decoder')
    parser.add_argument('--name', type=str, default=None, required=True, help='the trial name')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. """)
    parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int)
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
        'vit_large'], help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--modality', default='Fundus', type=str)
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of training""")
    parser.add_argument('--batch_size_per_gpu', default=512, type=int, help='Per-GPU batch-size')

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/data/', type=str, help='Please specify path to the dataset.')
    parser.add_argument('--dataset_format', default='vfm',choices=['vfm', 'ImageNet'], type=str, help='Please specify path to the dataset.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./results", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=5, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument('--test', action='store_true', help='Whether to run inference only')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        main(args_copy)
