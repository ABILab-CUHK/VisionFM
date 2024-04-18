import os, sys
import torch
from torch.utils.data import DataLoader
from imagen_pytorch import Unet, BaseUnet64, Imagen, SRUnet256, ImagenTrainer
from imagen_pytorch.data import Dataset
from argparse import ArgumentParser
from shutil import move

curr_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.join(curr_path, "..")
if root_path not in sys.path:
    sys.path.insert(0, root_path)

DATASET = {
    'mri' : 'MRI',
    'slitlamp' : 'SlitLamp'
}

    
parser = ArgumentParser()
parser.add_argument('id')
parser.add_argument('-bs', '--batch_size', type=int, default=64)
parser.add_argument('-mbs', '--max_batch_size', type=int, default=None)
parser.add_argument('-n', '--num_steps', type=int, default=250000)
parser.add_argument('-d', '--dataset', choices=list(DATASET.keys()), default='fundus')
parser.add_argument('--log_pbtc', type=int, default=10)
parser.add_argument('--save_pbtc', type=int, default=100)
parser.add_argument('--val_pbtc', type=int, default=1000)
parser.add_argument('-u', '--unet', type=int, default=1)



args = parser.parse_args()
args.dataset = DATASET[args.dataset]
if args.max_batch_size is None: args.max_batch_size = args.batch_size


unet1 = BaseUnet64(dim=256, memory_efficient=True)
unet2 = SRUnet256(memory_efficient=True)

# imagen, which contains the unets above (base unet and super resoluting ones)

unets = [unet1, unet2]

channels = 3
imagen = Imagen(
    condition_on_text = False,   # this must be set to False for unconditional Imagen
    unets = unets,
    channels=channels,
    image_sizes = (64, 256),
    timesteps = 1000
)

# generate dataset from the folder and resize images
dataset = Dataset(os.path.join(root_path, 'data/{}'.format(args.dataset)),
                  exts=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'TIF', 'JPG', 'bmp'],
                  image_size=256) # resized image size
print(len(dataset))
# train each unet separately

opt_path = os.path.join(root_path, 'model/{}.pth.tar'.format(args.id))
bkp_path = opt_path.replace(args.id, args.id+'bkp')

trainer = ImagenTrainer(imagen,
                        warmup_steps=10000,
                        split_valid_from_train=False,
                        cosine_decay_max_steps=args.num_steps,
                        only_train_unet_number=args.unet).cuda()

if os.path.isfile(opt_path): trainer.load(opt_path)
trainer.add_train_dataset(dataset, batch_size=args.batch_size)

print(trainer.num_steps_taken(1))
print(trainer.num_steps_taken(2))

for i in range(1, args.num_steps+1):
    loss = trainer.train_step(unet_number=args.unet,
                              max_batch_size=args.max_batch_size)

    if trainer.is_main and i == 1 or i % args.log_pbtc == 0:
        print(f'{args.unet} {i} {loss}')
        
    if i % args.save_pbtc == 0:
        if trainer.is_main and os.path.isfile(opt_path): move(opt_path, bkp_path)
        trainer.save(opt_path)

if i % args.save_pbtc != 0:
    if trainer.is_main and os.path.isfile(opt_path): move(opt_path, bkp_path)
    trainer.save(opt_path)
