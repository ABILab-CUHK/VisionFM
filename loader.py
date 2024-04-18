# the datasets used in the pretraining
import random
import math
import os
import glob
import numpy as np
from collections import defaultdict
from utils import pil_loader
from torch.utils.data import Dataset

class PretrainData(Dataset):
    '''
    The dataset class to load the images used in the pretraining stages:
    root: the root dir path for the images
    modality: the modality of loaded dataset, supporting 8 modalities (Fundus, OCT, External Eye, UBM, B-Ultrasound, MRI, Silt Lamp, FFA)
    transform: the involved the transformation.

    For each modality, its corresponding dir should be root/modality, e.g. /data/Fundus/XXX.jpg.
    .
    ├── /XXX/Fundus/
    │   ├── 1.jpg
    │   └── ...
    ├── /XXX/OCT/
    │   ├── 1.png
    │   └── ....
    └── ....
    '''
    def __init__(self, root:str, modality:str, transform=None, loader=pil_loader):
        self.data_root = root
        self.modality = modality
        assert self.modality in ['Fundus', 'OCT', 'External', 'UBM', 'Ultrasound', 'MRI', 'Silt-Lamp', 'FFA'], \
            f"Unsupported modality: {self.modality}"
        print(f"The {self.modality} data will be loaded in the pretraining stage")
        self.transform = transform
        self.loader = loader

        # read all the images
        self._entries = []
        self.dataset_dir = os.path.join(self.data_root, self.modality)
        assert os.path.exists(self.dataset_dir), f'the {self.dataset_dir} does not exists.'
        self.get_entries()

    def get_entries(self):
        # read all the image paths under the dataset dir
        self._entries:list = glob.glob(os.path.join(self.dataset_dir, "*"))
        print(f"The total images is: {len(self._entries)}")

    def get_label(self, img_path):
        # get the label based on the image path
        path2label = defaultdict() # can be loaded from the json file, where the key is the image name
        if img_path in path2label.keys():
            return path2label[img_path]
        else:
            return -1
    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index):
        try:
            img_path = self._entries[index]
            image = self.loader(img_path)
        except Exception as e:
            print(f"cannot load {img_path} due to the error: {e}")
            print(f"will randomly load another one.")
            index = random.randint(0, self.__len__())
            img_path = self._entries[index]
            image = self.loader(img_path)

        ############# reading labels #############
        # In most of the case, the label will be -1 due to the lacking of labels in pretraining images
        target = self.get_label(img_path)
        ##################

        if self.transform is not None:
            image = self.transform(image)
        return image, target

class PretrainMask(PretrainData):
    # the wrapper for the pretraining dataset with different masks on the input image
    def __init__(self, *args, patch_size, pred_ratio, pred_ratio_var, pred_aspect_ratio,
                 pred_shape='block', pred_start_epoch=0, **kwargs):
        super(PretrainMask, self).__init__(*args, **kwargs)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
                                           len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
                                                   len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                                        self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index):
        output:list = super(PretrainMask, self).__getitem__(index) # [image list, label]

        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W

            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2 # 16 for 224
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)

            else:
                # no implementation
                assert False

            masks.append(mask)

        return output + (masks,)

