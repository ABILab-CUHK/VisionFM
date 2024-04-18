# the script to extract features
import sys
sys.path.append('./')
import os
import argparse
import copy
import torch
import pickle
import torch.backends.cudnn as cudnn
import utils
import models
# from torchvision import transforms as pth_transforms
import transforms as self_transforms
from evaluation.dataset import ClsImgs, SegImgs

def extract_feats(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # fix the seed for reproducibility
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    mean, std = utils.get_stats(args.modality)
    print(f"Use the {args.modality} mean and std")

    train_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size)),
        self_transforms.RandomHorizontalFlip(),
        self_transforms.RandomVerticalFlip(),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    val_transform = self_transforms.Compose([
        self_transforms.Resize(size=(args.input_size, args.input_size)),
        self_transforms.ToTensor(),
        self_transforms.Normalize(mean, std),
    ])

    if args.mode == 'cls':
        dataset_train = ClsImgs(root=args.data_path, dst_root=args.dst_root,  split='training', transform=train_transform)
        dataset_val = ClsImgs(root=args.data_path, dst_root=args.dst_root, split='test', transform=val_transform)
    else:
        dataset_train = SegImgs(root=args.data_path, dst_root=args.dst_root,  split='training', transform=train_transform)
        dataset_val = SegImgs(root=args.data_path, dst_root=args.dst_root, split='test', transform=val_transform)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
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

    n_last_blocks = 4 # only extract the features of last 4 blocks, the default value
    if args.mode == 'seg':
        n_last_blocks = len(model.blocks) # extract the features of all the blocks

    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load pre-train weights
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs): 
        train_loader.sampler.set_epoch(epoch)
        model.eval()

        # ============ extract features ... ============
        print(f"Epoch {epoch}: Extracting features for train set...")
        extract_features(model, train_loader, n_last_blocks,  epoch, args.mode)
        if epoch == 0:
            print(f"Epoch {epoch}: Extracting features for val set...")
            extract_features(model, val_loader, n_last_blocks,  epoch, args.mode)

@torch.no_grad()
def extract_features(model, data_loader, n, epoch, mode='cls'):
    metric_logger = utils.MetricLogger(delimiter="  ")
    for samples, labs, extras in metric_logger.log_every(data_loader, 10):
        samples = samples.cuda(non_blocking=True)

        feats_dict = {}
        batch_size = samples.shape[0]

        intermediate_output = model.get_intermediate_layers(samples, n) # features of last n blocks in vit

        if mode == 'cls':
            selected_levels = [item for item in range(len(model.blocks))][-4:]
            output = [x[:, 0] for x in intermediate_output] # only retain the CLS token
        elif mode == 'seg':
            if n == 12:
                selected_levels = [3, 5, 7, 11] # for default vit-base model
            elif n == 24: # for vit-large
                selected_levels = [5, 11, 17, 23]
            else:
                raise NotImplementedError # please set suitable selected_levels
            # only retain the patch token in 4 level
            output = [intermediate_output[idx][:, 1:] for idx in selected_levels]


        dst_paths = extras['dst_path']
        img_paths = extras['img_path']
        for batch_idx in range(batch_size):
            dst_path = dst_paths[batch_idx]

            if mode == 'cls':
                feats_dict['img_name'] = '/'.join(dst_path.split('/')[-3:])
            else:
                feats_dict['img_name'] = '/'.join(dst_path.split('/')[-4:])
            feats_dict['feats_idx'] = selected_levels
            feats_dict['feats'] = [feat[batch_idx].cpu().numpy() for feat in output]
            feats_dict['labels'] = labs[batch_idx].numpy()
            feats_dict['img_path'] = img_paths[batch_idx]
            # feats_dict['image'] = samples[batch_idx].cpu().numpy() #

            f_dir, f_name = os.path.split(dst_path)
            if not os.path.exists(f_dir):
                os.makedirs(f_dir)
            pickle_path = os.path.join(f_dir, f_name.split('.')[0] + f'_{epoch}.pickle')
            file = open(pickle_path, 'wb')
            pickle.dump(feats_dict, file)
            file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='vit_base', type=str, choices=['vit_tiny', 'vit_small', 'vit_base',
                                                                          'vit_large', 'vit_huge'], help='Architecture.')
    parser.add_argument('--input_size', type=int, default=224, help='input size')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')

    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')

    parser.add_argument('--data_path', default='./data/FundusClassification/', type=str, help='Please specify path to the dataset.')
    parser.add_argument('--dst_root', type=str, help='The root dir to save the extracted features')
    parser.add_argument('--modality', type=str, default='Fundus', choices=['Fundus', 'OCT', 'FFA', 'SlitLamp', 'UltraSound', 'MRI', 'External', 'UBM'], help='the modality of the dataset')
    parser.add_argument('--mode', type=str, default='cls', choices=['cls', 'seg']) # the extraction for classification and segmentation task

    args = parser.parse_args()

    for checkpoint_key in args.checkpoint_key.split(','):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        extract_feats(args_copy)
