# Train a regression decoder with multi-modalities: Fundus and External Eye photographs, for biomarker prediction
import sys
sys.path.append('../')
import os
import argparse
import json
import torch
import torch.backends.cudnn as cudnn
import utils
from pathlib import Path
from torch import nn
from evaluation.dataset import ClsFeats
from models.head import RegHead

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

    # ============ preparing data ... ============
    dataset_train = ClsFeats(root=args.data_path, datasets=args.datasets, split='training')
    dataset_val = ClsFeats(root=args.data_path, datasets=args.datasets, split='test')
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
    linear_classifier = RegHead(embed_dim=embed_dim*4, num_classes=args.num_labels)
    print(utils.get_parameter_number(linear_classifier))
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
    to_restore = {"epoch": 0, "best_mae": 1.0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
        )
    start_epoch = to_restore["epoch"]
    best_mae = to_restore["best_mae"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        linear_classifier.train()
        train_stats = train(linear_classifier, optimizer, train_loader, epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            linear_classifier.eval()
            test_stats = validate_network(val_loader, linear_classifier)

            print(f"MAE at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['mae']:.4f}")
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

            if utils.is_main_process() and (test_stats['mae'] <= best_mae):
                # always only save best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_mae": test_stats['mae'],
                }
                torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(epoch)))
            best_mae = min(best_mae, test_stats['mae'])
            print(f'Best mae so far: {best_mae:.4f}')

    print("Training of the supervised regression decoder completed. \n"
          "The best MAE: {mae:.4f}".format(mae=best_mae))


def train(linear_classifier, optimizer, loader, epoch):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True) # [B, 4, 1280]
        target = target.cuda(non_blocking=True)

        inp_splits = torch.split(inp, 1, dim=1)
        inp_splits = [item.squeeze(dim=1) for item in inp_splits]

        output = linear_classifier(torch.cat(inp_splits, dim=-1))  # [B, C]

        # compute the loss
        loss = nn.SmoothL1Loss()(output, target)

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
    pred_precents = []
    predictions, targets = [], []
    all_extras = []
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        inp_splits = torch.split(inp, 1, dim=1)
        inp_splits = [item.squeeze(dim=1) for item in inp_splits]
        # use features from 4 scales
        output = linear_classifier(torch.cat(inp_splits, dim=-1))  # [B, C]

        # save results
        predictions.append(output.cpu())
        targets.append(target.cpu())
        # compute the metrics

        batch_size = inp.shape[0]
        loss = nn.SmoothL1Loss()(output, target)

        pred_precent = torch.abs(output - target).mean(dim=0) / target.mean(dim=0) # [38]
        pred_precents.append(pred_precent.unsqueeze(dim=0).cpu())
        mae = pred_precent.mean().cpu().numpy()

        # batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['mae'].update(mae.item(), n=batch_size)
    print('MAE: Reg {mae.global_avg:.4f} Loss: {loss.global_avg:.4f}'.format(mae=metric_logger.mae, loss=metric_logger.loss))
    #
    percents = torch.cat(pred_precents).mean(dim=0)
    print(f"Percents on each output: {percents.numpy()}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a regression decoder on top of a VisionFM encoder')
    parser.add_argument('--name', type=str, default=None, required=True, help='the trial name')
    parser.add_argument('--epochs', default=500, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). """)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/data/', type=str, help='Please specify path to the data.')
    parser.add_argument('--datasets', nargs='+', help='choose the data features by specifying their modalities')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./results", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=38, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
    parser.add_argument('--embed_dim', type=int, default=768, help='The choice for embed dim')
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_decoder(args)
