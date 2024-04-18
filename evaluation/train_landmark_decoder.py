# Train a landmark detection decoder
import sys
sys.path.append('./')
import os
import argparse
import json
import torch
import torch.backends.cudnn as cudnn
import utils
import models
from pathlib import Path
from torch import nn
import evaluation.transforms as self_transforms
from evaluation.dataset import SegImgs
from models.unetr_head import Unetr_Head

def two_d_softmax(x):
    exp_y = torch.exp(x)
    return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

def nll_across_batch(output, target):
    nll = -target * torch.log(output.double())
    return torch.mean(torch.sum(nll, dim=(2, 3)))

def train_decoder(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # fix the seed for reproducibility
    utils.fix_random_seeds(args.seed)

    args.output_dir = os.path.join(args.output_dir, args.name) # set new output dir with name
    if not os.path.exists(args.output_dir):
        print(f"Create the output_dir: {args.output_dir}")
        os.makedirs(args.output_dir)

    # ============ preparing data ... ============
    mean, std = utils.get_stats(args.modality)
    print(f"use the {args.modality} mean and std: {mean} and {std}")

    train_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size)),
        self_transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        self_transforms.RandomHorizontalFlip(),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    val_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size)),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    dataset_train = SegImgs(root=args.data_path, split='training', transform=train_transform, label_suiffix='npy')
    dataset_val = SegImgs(root=args.data_path,  split='test', transform=val_transform, label_suiffix='npy')
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

    # set the decoder for the landmark detection
    linear_classifier = Unetr_Head(embed_dim=embed_dim, num_classes = args.num_labels, img_dim=args.input_size)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # set optimizer
    parameters = linear_classifier.parameters()
    optimizer = torch.optim.AdamW(
        parameters,
        args.lr,  # linear scaling rule
        betas=(0.9, 0.999), weight_decay=0.05
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_mre": 200.}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
        )
    start_epoch = to_restore["epoch"]
    best_mre = to_restore["best_mre"]

    # the main loop
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        linear_classifier.train()
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            # model.eval()
            linear_classifier.eval()
            test_stats = validate_network(val_loader, model, linear_classifier)
            all_mre = test_stats['metric_landmarks']

            print(f"Mean mre at epoch {epoch} of the network on the {len(dataset_val)} test images: {all_mre:.4f}")
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (all_mre <= best_mre):
                # always only save the best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_mre": all_mre,
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(epoch)))

            best_mre = min(best_mre, all_mre)
            print(f'Min mre so far: {best_mre:.4f}')
    print("Training of the supervised landmark decoder on frozen features completed.\nAnd the best mre: {mre:.4f}".format(mre=best_mre))

def train(model, linear_classifier, optimizer, loader, epoch):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target, _ ) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True) # [B, C, H, W]
        target = target.cuda(non_blocking=True) # [B, C, H, W]

        # forward
        with torch.no_grad():
            n = len(model.blocks)  # get all the layers
            if n == 12:
                selected_levels = [3, 5, 7, 11] # for default vit-base model
            elif n == 24: # for vit-large
                selected_levels = [5, 11, 17, 23]
            else:
                raise NotImplementedError # please set suitable selected_levels
            intermediate_output = model.get_intermediate_layers(inp, n)
            features = [intermediate_output[idx][:, 1:] for idx in selected_levels] # for multi-scale, [B, 196, dim]

        output = linear_classifier(features, inp)  # [B, C, H, W]
        output = two_d_softmax(output)
        loss = nll_across_batch(output, target)

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
def validate_network(val_loader, model, linear_classifier):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target, img_paths in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        batch_size = inp.shape[0]

        # forward
        with torch.no_grad():
            n = len(model.blocks)  # get all the layers
            if n == 12:
                selected_levels = [3, 5, 7, 11] # for default vit-base model
            elif n == 24: # for vit-large
                selected_levels = [5, 11, 17, 23]
            else:
                raise NotImplementedError # please set suitable selected_levels
            intermediate_output = model.get_intermediate_layers(inp, n)
            features = [intermediate_output[idx][:, 1:] for idx in selected_levels] # for multi-scale, [B, 196, dim]

        output = linear_classifier(features, inp)  # [B, C, H, W]
        output = two_d_softmax(output)
        loss = nll_across_batch(output, target)

        # compute the metrics
        metric_landmarks = utils.mre(output, target)
        metric_logger.update(loss=loss.item())
        metric_logger.meters['metric_landmarks'].update(metric_landmarks, n=batch_size)
    print('* Landmarks {landmarks.global_avg:.4f} loss {losses.global_avg:.3f}'
          .format(landmarks=metric_logger.metric_landmarks, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a landmark detection decoder')
    parser.add_argument('--name', type=str, default=None, required=True, help='the trial name')
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
                                                                          'vit_large', 'vit_huge',],help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
                        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate of training.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--data_path', default='', type=str,
                        help='Please specify path to the dataset.')
    parser.add_argument('--modality', default='UBM', type=str, help='The involved modality')
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=3, type=int, help='Number of points for landmark detection task')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_decoder(args)
