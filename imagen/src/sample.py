import os, sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from imagen_pytorch import Unet, BaseUnet64, Imagen, SRUnet256, ImagenTrainer
from imagen_pytorch.data import Dataset
from argparse import ArgumentParser
from filelock import FileLock
import json

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
parser.add_argument('-bs', '--batch_size', type=int, default=2048)
parser.add_argument('-n', '--amount', type=int, default=100)
parser.add_argument('-d', '--dataset', choices=list(DATASET.keys()), default='fundus')
parser.add_argument('--log_pbtc', type=int, default=10)
parser.add_argument('-u', '--unet', type=int, default=None)
parser.add_argument('-s', '--server', default='c')

args = parser.parse_args()
args.dataset = DATASET[args.dataset]

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
                  image_size=256) # resized image size

# train each unet separately

model_path = os.path.join(root_path, 'model/{}.pth.tar'.format(args.id))

assert os.path.isfile(model_path)

trainer = ImagenTrainer(imagen).cuda()
trainer.load(model_path)

trainer.add_train_dataset(dataset, batch_size=args.batch_size)

print(trainer.num_steps_taken(1))
print(trainer.num_steps_taken(2))

iters = max(args.amount // args.batch_size, 1)

opt_path = os.path.join(root_path, 'output/{}'.format(args.dataset))
if not os.path.exists(opt_path): os.mkdir(opt_path)

lock_path = os.path.join(opt_path, '../{}.lock'.format(args.dataset))
locker = FileLock(lock_path)

log_path = os.path.join(opt_path, '../{}.txt'.format(args.dataset))

with locker:
    if os.path.isfile(log_path):
        with open(log_path, 'r') as f: start = int(f.read())
    else:
        start = 0
        
    end = start + args.amount
    with open(log_path, 'w') as f:
        f.write(str(end))
            
for i in range(start, end):
    imgs = trainer.sample(batch_size=args.batch_size, stop_at_unet_number=args.unet, return_pil_images=True)
    for j, img in enumerate(imgs):
        img.save(os.path.join(opt_path, '{}{}.png'.format(args.server, i*args.batch_size+j)))
