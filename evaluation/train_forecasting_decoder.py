# train a forecasting decoder
import sys
sys.path.append('../')
import os
import argparse
import json
import torch
import torch.backends.cudnn as cudnn
import utils
import models
from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms
from evaluation.dataset import GFDataset
from models.head import ForecastHead
from sklearn.metrics import f1_score

def train_decoder(args):
	utils.init_distributed_mode(args)
	print("git:\n  {}\n".format(utils.get_sha()))
	print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
	cudnn.benchmark = True

	# fix the seed for reproducibility
	utils.fix_random_seeds(args.seed)

	# ============ preparing data ... ============
	mean, std = utils.get_stats(args.modality)
	print(f"Use the {args.modality} mean and std")

	train_transform = pth_transforms.Compose([
		pth_transforms.Resize(size=(args.input_size, args.input_size)),
		pth_transforms.RandomHorizontalFlip(),
		pth_transforms.RandomVerticalFlip(),
		pth_transforms.ToTensor(),
		pth_transforms.Normalize(mean, std),
	])

	val_transform = pth_transforms.Compose([
		pth_transforms.Resize(size=(args.input_size, args.input_size)),
		pth_transforms.ToTensor(),
		pth_transforms.Normalize(mean, std),
	])


	dataset_train = GFDataset(root=args.data_path, split='training', transform=train_transform)
	dataset_val = GFDataset(root=args.data_path, split='test', transform=val_transform)
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
		img_size = [args.input_size],
		patch_size=args.patch_size,
		num_classes=0,
		use_mean_pooling=args.avgpool_patchtokens==1)
	embed_dim = model.embed_dim
	model.cuda()
	print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
	# load weights to evaluate
	utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)

	linear_classifier = ForecastHead(input_dim=embed_dim, max_len=args.max_len)
	linear_classifier = linear_classifier.cuda()
	linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

	# set optimizer
	parameters = linear_classifier.parameters()

	optimizer = torch.optim.SGD(
		parameters,
		args.lr,  # linear scaling rule
		momentum=0.9,
		weight_decay=0, # we do not apply weight decay
	)

	# optimizer = torch.optim.AdamW(
        # parameters, lr=args.lr,
        # betas=(0.9, 0.999), eps=1e-08
    	# )
    	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)


	# Optionally resume from a checkpoint
	to_restore = {"epoch": 0, "best_f1": 1.0}
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
		model.eval()

		linear_classifier.train()
		train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks,
							args.avgpool_patchtokens, args.pos_weight)
		log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
					 'epoch': epoch}
		if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
			linear_classifier.eval()
			test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.pos_weight)

			print(f"Mean F1 at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['f1']:.4f}")
			log_stats = {**{k: v for k, v in log_stats.items()},
						 **{f'test_{k}': v for k, v in test_stats.items()}}

			if utils.is_main_process() and (test_stats['f1'] >= best_f1):
				# always only save best checkpoint till now
				with (Path(args.output_dir) / "log.txt").open("a") as f:
					f.write(json.dumps(log_stats) + "\n")

				save_dict = {
					"epoch": epoch + 1,
					"state_dict": linear_classifier.state_dict(),
					"optimizer": optimizer.state_dict(),
					"best_f1": test_stats['f1'],
				}
				torch.save(save_dict, os.path.join(args.output_dir, "checkpoint_{}_linear.pth".format(epoch)))
			best_f1 = max(best_f1, test_stats['f1'])
			print(f'Best f1 so far: {best_f1:.4f}')

	print("The mean F1: {f1:.4f}".format(f1=best_f1))


def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, pos_weight):
	metric_logger = utils.MetricLogger(delimiter="  ")
	metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
	header = 'Epoch: [{}]'.format(epoch)
	for (inp, target, delta_time) in metric_logger.log_every(loader, 20, header):
		# move to gpu
		inp = inp.cuda(non_blocking=True) 
		target = target.cuda(non_blocking=True)
		delta_time = delta_time.cuda(non_blocking=True)

		# forward
		 with torch.no_grad():
			 intermediate_output = model.get_intermediate_layers(inp, n)
			 if avgpool == 0:
				 output = [x[:, 0] for x in intermediate_output] # each one [B, dim]
			 elif avgpool == 1:
				 output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
			 elif avgpool == 2:
				 output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
			 else:
				 assert False, "Unkown avgpool type {}".format(avgpool)
			 input_feats = torch.cat(output, dim=-1)

		output = linear_classifier(input_feats, delta_time)
		# compute the loss
		loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target.float())

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
def validate_network(val_loader, model, linear_classifier, n, avgpool, pos_weight):
	linear_classifier.eval()
	metric_logger = utils.MetricLogger(delimiter="  ")
	header = 'Test:'
	for inp, target, delta_time in metric_logger.log_every(val_loader, 20, header):
		# move to gpu
		inp = inp.cuda(non_blocking=True)
		target = target.cuda(non_blocking=True)
		delta_time = delta_time.cuda(non_blocking=True)

		# forward
		with torch.no_grad():
			intermediate_output = model.get_intermediate_layers(inp, n)
			if avgpool == 0:
				output = [x[:, 0] for x in intermediate_output]
			elif avgpool == 1:
				output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
			elif avgpool == 2:
				output = [x[:, 0] for x in intermediate_output] + [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
			else:
				assert False, "Unkown avgpool type {}".format(avgpool)
			input_feats = torch.cat(output, dim=-1)

		output = linear_classifier(input_feats, delta_time)

		loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(output, target.float())

		# compute the metrics
		f1 = f1_score(target.cpu().numpy(), output.cpu().numpy())

		batch_size = inp.shape[0]
		metric_logger.update(loss=loss.item())
		metric_logger.meters['f1'].update(f1.item(), n=batch_size)
	print('f1: {f1.global_avg:.4f} Loss: {loss.global_avg:.4f}'.format(f1=metric_logger.f1, loss=metric_logger.loss))
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Train a forecasting decoder')
	parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
		for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""")
	parser.add_argument('--avgpool_patchtokens', default=0, choices=[0, 1, 2], type=int,
						help="""Whether or not to use global average pooled features or the [CLS] token.
		We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).""")
	parser.add_argument('--epochs', default=500, type=int, help='Number of epochs of training.')
	parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
		training (highest LR used during training). """)
	parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
	parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
		distributed training; see https://pytorch.org/docs/stable/distributed.html""")
	parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
	parser.add_argument('--data_path', default='/path/to/data/', type=str, help='Please specify path to the data.')
	parser.add_argument('--seed', default=0, type=int)
	parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
	parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
	parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
	parser.add_argument('--num_labels', default=1, type=int, help='Number of labels for linear classifier')
	parser.add_argument('--load_from', default=None, help='Path to load checkpoints to resume training')
	parser.add_argument('--embed_dim', type=int, default=768, help='The choice for embed dim')
	parser.add_argument('--modality',type=str, default='Fundus', choices=['Fundus', 'OCT', 'MRI'])
	parser.add_argument('--pos_weight', type=int, default=10, help='Weight on the positive class in loss')
	parser.add_argument('--max_len', type=int, default=6500, help='Length of time interval positional encoding')
	parser.add_argument('--arch', default='vit_base', type=str, help='Encoder architecture.')
	parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
	parser.add_argument('--input_size', type=int, default=224, help='input size')
	parser.add_argument('--pretrained_weights', default='', type=str, help='Path to pretrained weights to evaluate.')
	parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
	
	args = parser.parse_args()
	if args.output_dir:
		Path(args.output_dir).mkdir(parents=True, exist_ok=True)
	train_decoder(args)
