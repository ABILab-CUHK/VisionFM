# Train a decoder for the multi-modal classification task
import sys
sys.path.append('./')
import os
import argparse
import json
import torch
import torch.backends.cudnn as cudnn
import utils
import numpy as np
from pathlib import Path
from torch import nn
from evaluation.dataset import ClsFeats
from models.head import ClsHead
from monai.metrics import compute_roc_auc
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

def train_decoder(args):
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

    # ============ preparing features ... ============
    # datasets = ['FunCls_feat', 'UltraCls_feat', 'FFACls_feat', 'OCTCls_feat', 'SiltLampCls_feat']
    dataset_train = ClsFeats(root=args.data_path, split='training', datasets=args.datasets)
    dataset_val = ClsFeats(root=args.data_path, split='test', datasets=args.datasets)
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

    embed_dim = args.embed_dim
    linear_classifier = ClsHead(embed_dim=embed_dim*4, num_classes=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # =============== set optimizer
    parameters = linear_classifier.parameters()

    optimizer = torch.optim.SGD(
        parameters,
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_f1": 0.}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
        )
    start_epoch = to_restore["epoch"]
    best_f1 = to_restore["best_f1"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        linear_classifier.train()
        train_stats = train(linear_classifier, optimizer, train_loader, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # model.eval()
            linear_classifier.eval()
            test_stats = validate_network(val_loader, linear_classifier)

            all_acc = []
            all_auc =[]
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
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (all_f1 >= best_f1):
                # always only save best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_f1": all_f1,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(epoch)))

            best_f1 = max(best_f1, all_f1)
            print(f'Max f1 so far: {best_f1:.4f}')

    print("Training of the supervised linear classifier on frozen features completed.\nAnd the best F1-score: {f1:.4f}".format(f1=best_f1))

def train(linear_classifier, optimizer, loader, epoch):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        inp = inp.cuda(non_blocking=True)  # [B, 4, 768]
        target = target.cuda(non_blocking=True)

        inp_splits = torch.split(inp, 1, dim=1)
        inp_splits = [item.squeeze(dim=1) for item in inp_splits]
        inputs = torch.cat(inp_splits, dim=-1)  # [B, 4*768]
        output = linear_classifier(inputs)

        # compute the loss for each type of task
        gt_dr_grading = target[:, :5]  # DR Grading,
        gt_dr = target[:, 5] # DR
        gt_glaucoma = target[:, 6] # Glaucoma
        gt_amd = target[:, 7] # AMD
        gt_cata = target[:, 8]  # Cataract
        gt_hyper_retin = target[:, 9] # Hyper Retino
        gt_rvo = target[:, 10] # RVO
        gt_myopia = target[:, 11] # Myopia
        gt_rd = target[:, 12]  # Retinal Detachment

        pred_dr_grading = output[:, :5]
        pred_dr = output[:, 5]
        pred_glaucoma = output[:, 6]
        pred_amd = output[:, 7] 
        pred_cata = output[:, 8]
        pred_hyper_retin = output[:, 9]
        pred_rvo = output[:, 10]
        pred_myopia = output[:, 11]
        pred_rd = output[:, 12]

        loss_dr_grading = nn.CrossEntropyLoss()(pred_dr_grading, gt_dr_grading)
        loss_dr = nn.BCEWithLogitsLoss()(pred_dr, gt_dr)
        loss_glaucoma = nn.BCEWithLogitsLoss()(pred_glaucoma, gt_glaucoma)
        loss_amd = nn.BCEWithLogitsLoss()(pred_amd, gt_amd)
        loss_cata = nn.BCEWithLogitsLoss()(pred_cata, gt_cata)
        loss_hyper_retin = nn.BCEWithLogitsLoss()(pred_hyper_retin, gt_hyper_retin)
        loss_rvo = nn.BCEWithLogitsLoss()(pred_rvo, gt_rvo)
        loss_myopia = nn.BCEWithLogitsLoss()(pred_myopia, gt_myopia)
        loss_rd = nn.BCEWithLogitsLoss()(pred_rd, gt_rd)

        loss = 5 * loss_dr_grading + loss_dr + loss_glaucoma + loss_amd + loss_cata + 5 * loss_hyper_retin + loss_rvo + loss_myopia + loss_rd
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
def validate_network(val_loader, linear_classifier):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    dice_classes = []

    targets, preds = [], []
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = inp.shape[0]

        inp_splits = torch.split(inp, 1, dim=1)
        inp_splits = [item.squeeze(dim=1) for item in inp_splits]
        inputs = torch.cat(inp_splits, dim=-1)

        output = linear_classifier(inputs)

        # save results
        targets.append(target.detach().cpu()) 
        preds.append(output.detach().cpu())

        # compute the metrics
        # compute the loss for each type of task
        gt_dr_grading = target[:, :5]  # DR Grading,
        gt_dr = target[:, 5] # DR
        gt_glaucoma = target[:, 6] # Glaucoma
        gt_amd = target[:, 7] # AMD
        gt_cata = target[:, 8]  # Cataract
        gt_hyper_retin = target[:, 9] # Hyper Retino
        gt_rvo = target[:, 10] # RVO
        gt_myopia = target[:, 11] # Myopia
        gt_rd = target[:, 12]  # Retinal Detachment

        pred_dr_grading = output[:, :5]
        pred_dr = output[:, 5]
        pred_glaucoma = output[:, 6]
        pred_amd = output[:, 7] 
        pred_cata = output[:, 8]
        pred_hyper_retin = output[:, 9]
        pred_rvo = output[:, 10]
        pred_myopia = output[:, 11]
        pred_rd = output[:, 12]

        tasks = ['dr_grading', 'dr', 'glaucoma', 'amd', 'cata', 'hyper_retin', 'rvo', 'myopia', 'rd']
        batch_size_dr_grading = 1 if pred_dr_grading.shape[0] == 0 else pred_dr_grading.shape[0]
        acc_dr_grading, = utils.accuracy(pred_dr_grading, gt_dr_grading.argmax(dim=1), topk=(1,))
        acc_dr_grading = acc_dr_grading.item()
        auc_dr_grading_list = compute_roc_auc(torch.softmax(pred_dr_grading, dim=1).cpu(), gt_dr_grading.cpu(),average="none")
        auc_dr_grading = np.array(auc_dr_grading_list[1:]).mean() 
        pcf = precision_recall_fscore_support(gt_dr_grading.argmax(dim=1).cpu().numpy(), pred_dr_grading.argmax(dim=1).cpu().numpy(), average=None)
        precision_dr_grading = pcf[0][1:].mean()
        recall_dr_grading = pcf[1][1:].mean()
        f1_dr_grading = pcf[2][1:].mean()

        all_metrics = defaultdict()
        for idx in range(1, 9):
            task = tasks[idx]
            globals()[f"batch_size_{task}"] = 1 if locals()[f"pred_{task}"].shape[0] == 0 else locals()[f"pred_{task}"].shape[0]
            globals()[f"acc_{task}"], = utils.accuracy(binary2multi(locals()[f"pred_{task}"]), locals()[f"gt_{task}"], topk=(1,))
            globals()[f"acc_{task}"] = globals()[f"acc_{task}"].item()
            globals()[f"auc_{task}"] = compute_roc_auc(torch.sigmoid(locals()[f"pred_{task}"]), locals()[f"gt_{task}"], average="none")
            pcf = precision_recall_fscore_support(locals()[f"gt_{task}"].cpu().numpy(), torch.sigmoid(locals()[f"pred_{task}"]).cpu().numpy() > 0.5,average=None)
            if pcf[0].shape[0] == 1:
                print(f"gt of {task} only 1 class (usually only contain background class), will set precision, recall and f1 as 0.")
                globals()[f"precision_{task}"] = 0.0
                globals()[f"recall_{task}"] = 0.0
                globals()[f"f1_{task}"] = 0.0
            else:
                globals()[f"precision_{task}"] = pcf[0][1:].item() 
                globals()[f"recall_{task}"] = pcf[1][1:].item()
                globals()[f"f1_{task}"] = pcf[2][1:].item()

        metrics = ['acc', 'auc', 'f1', 'precision', 'recall']
        for metric in metrics:
            print_str = f"{metric} "
            for task in tasks:
                if task == 'dr_grading':
                    metric_logger.meters[f"{metric}_{task}"].update(locals()[f"{metric}_{task}"], n=locals()[f"batch_size_{task}"])
                else:
                    metric_logger.meters[f"{metric}_{task}"].update(globals()[f"{metric}_{task}"], n=globals()[f"batch_size_{task}"])

    print('*ACC:  DR_Grading {dr_grading.global_avg:.4f}  DR {dr.global_avg:.4f} Glaucoma {glaucoma.global_avg:.4f} AMD {amd.global_avg:.4f} Cata {cata.global_avg:.4f} HyperRetin {hyper.global_avg:.4f} RVO {rvo.global_avg:.4f} Myopia {myopia.global_avg:.4f} RD {rd.global_avg:.4f}'
          .format(dr_grading=metric_logger.acc_dr_grading, dr=metric_logger.acc_dr, glaucoma=metric_logger.acc_glaucoma, amd=metric_logger.acc_amd, cata=metric_logger.acc_cata, hyper=metric_logger.acc_hyper_retin, rvo=metric_logger.acc_rvo, myopia=metric_logger.acc_myopia,rd=metric_logger.acc_rd))
    print('*AUC: DR_Grading {dr_grading.global_avg:.4f}  DR {dr.global_avg:.4f} Glaucoma {glaucoma.global_avg:.4f} AMD {amd.global_avg:.4f}  Cata {cata.global_avg:.4f} HyperRetin {hyper.global_avg:.4f} RVO {rvo.global_avg:.4f} Myopia {myopia.global_avg:.4f} RD {rd.global_avg:.4f}'
          .format(dr_grading=metric_logger.auc_dr_grading, dr=metric_logger.auc_dr, glaucoma=metric_logger.auc_glaucoma, amd=metric_logger.auc_amd, cata=metric_logger.auc_cata, hyper=metric_logger.auc_hyper_retin, rvo=metric_logger.auc_rvo, myopia=metric_logger.auc_myopia,rd=metric_logger.auc_rd))
    print('*F1-Sore: DR_Grading {dr_grading.global_avg:.4f}  DR {dr.global_avg:.4f} Glaucoma {glaucoma.global_avg:.4f} AMD {amd.global_avg:.4f}  Cata {cata.global_avg:.4f} HyperRetin {hyper.global_avg:.4f} RVO {rvo.global_avg:.4f} Myopia {myopia.global_avg:.4f} RD {rd.global_avg:.4f}'
          .format(dr_grading=metric_logger.f1_dr_grading, dr=metric_logger.f1_dr, glaucoma=metric_logger.f1_glaucoma, amd=metric_logger.f1_amd, cata=metric_logger.f1_cata, hyper=metric_logger.f1_hyper_retin, rvo=metric_logger.f1_rvo, myopia=metric_logger.f1_myopia,rd=metric_logger.f1_rd))
    print('*Precesion: DR_Grading {dr_grading.global_avg:.4f}  DR {dr.global_avg:.4f} Glaucoma {glaucoma.global_avg:.4f} AMD {amd.global_avg:.4f}  Cata {cata.global_avg:.4f} HyperRetin {hyper.global_avg:.4f} RVO {rvo.global_avg:.4f} Myopia {myopia.global_avg:.4f} RD {rd.global_avg:.4f}'
          .format(dr_grading=metric_logger.precision_dr_grading, dr=metric_logger.precision_dr, glaucoma=metric_logger.precision_glaucoma, amd=metric_logger.precision_amd, cata=metric_logger.precision_cata, hyper=metric_logger.precision_hyper_retin, rvo=metric_logger.precision_rvo, myopia=metric_logger.precision_myopia,rd=metric_logger.precision_rd))
    print('*Recall: DR_Grading {dr_grading.global_avg:.4f}  DR {dr.global_avg:.4f} Glaucoma {glaucoma.global_avg:.4f} AMD {amd.global_avg:.4f}  Cata {cata.global_avg:.4f} HyperRetin {hyper.global_avg:.4f} RVO {rvo.global_avg:.4f} Myopia {myopia.global_avg:.4f} RD {rd.global_avg:.4f}'
          .format(dr_grading=metric_logger.recall_dr_grading, dr=metric_logger.recall_dr, glaucoma=metric_logger.recall_glaucoma, amd=metric_logger.recall_amd, cata=metric_logger.recall_cata, hyper=metric_logger.recall_hyper_retin, rvo=metric_logger.recall_rvo, myopia=metric_logger.recall_myopia,rd=metric_logger.recall_rd))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def binary2multi(input_tensor):
    # input_tensor: [batch]
    return torch.cat([1.0 - input_tensor.unsqueeze(dim=1), input_tensor.unsqueeze(dim=1)], dim=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a classification decoder using pre-extracted features')
    parser.add_argument('--name', type=str, default=None, required=True, help='the trial name')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/data/', type=str, help='Please specify path to the data.')
    parser.add_argument('--datasets', nargs='+', help='set the trained features from modalities')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")

    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=13, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument('--embed_dim', type=int, default=768, help='The choice for embed dim')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_decoder(args)
