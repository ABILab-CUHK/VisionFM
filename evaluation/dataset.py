# the dataset classes used in downstream tasks
import os
import glob
import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
import utils
from utils import pil_loader, npy_loader
from torchvision import datasets

class SegImgs(Dataset):
    """
    The dataset for the downstream segmentation task.
    root: str, the dataset root dir
    dst_root: str, the save dir for the extracted features, used in feature extraction stage.
    split: str, the training split mode, training or test
    transform: the involved transformations, the data augmentation
    loader: func, the loader to load the image, default is the pil_loader
    few_shot: int, select the {few_shot} images from the training set.
    label_suiffix: str, the suffix of files in `labels` directory, default is png

    The folder structure should be:
    ├── /XXX/VesselSegmentation/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── images/
    │   │   │   │   ├── 1.jpg
    │   │   │   │   └── ...
    │   │   │   └── labels/
    │   │   │       ├── 1.png
    │   │   │       └── ...
    │   │   └── test/
    │   │       └── ...
    │   ├── dataset_B/
    │   │   └── ...
    │   └── ...
    └── ...

    The range of pixel values in images in labels directory should be [0, C-1], where C is the number of classes.

    the file name of label should be same as that of the corresponding image file.
    For example, label path: /xxx/VesselSegmentation/dataset_A/training/masks/1.png, image path: /xxx/VesselSegmentation/dataset_A/training/images/1.png
    Otherwise, you need to manually assign their corresponding relationships.
    """
    def __init__(self, root:str, split, dst_root:str=None, transform=None, loader=pil_loader, few_shot=-1, label_suiffix='png'):
        self.root:str = root
        self.dst_root = dst_root # the dst root dir for extracted features
        self.split = split
        assert self.split in ["training", "test"], f"unsupported split mode: {self.split}"
        self.transform = transform
        self.loader = loader
        self.label_suffix = label_suiffix

        self.few_shot = few_shot

        self._entries = []
        self._entries_cache = []

        # read all the images
        dataset_dir = self.root
        assert os.path.exists(dataset_dir), f'cannot find the dataset dir: {dataset_dir}'
        # find the sub-datasets under this task
        sub_datasets:list = utils.get_sub_dirs(dataset_dir)
        self.get_entries(sub_datasets)
        random.shuffle(self._entries)

        if self.few_shot != -1 and self.split == "training":
            self._entries = random.sample(self._entries, 1)
            print(f"select {self.few_shot} images for the few-shot setting")
        print(f"there are {len(self._entries)} images in the {self.split} set.")

    def get_entries(self, sub_datasts:list):
        # read all the images
        for sub_dataset in sub_datasts:
            print(f"reading images at {sub_dataset}")
            img_dir = os.path.join(sub_dataset, self.split, 'images')
            assert os.path.exists(img_dir), f"cannot find the {img_dir}"
            img_files = glob.glob(os.path.join(img_dir, "*"))
            print(f"there are {len(img_files)} image files in {img_dir}")
            self._entries += img_files
        print(f"the total number of images is: {len(self._entries)}")

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        img_path:str = self._entries[idx]
        try:
            img = self.loader(img_path)
        except BaseException as e:
            print(f"cannot load {img_path} due to the error: {e}")
            print(f"will randomly load another one.")
            index = random.randint(0, self.__len__())
            img_path = self._entries[index]
            img = self.loader(img_path)

        if self.dst_root is None:
            dst_path = None
        else:
            dst_path = img_path.replace(self.root, self.dst_root)

        label_path = img_path.replace('images', 'labels')
        if self.label_suffix != 'png':
            label_path = label_path.replace('.png', f'.{self.label_suffix}')
        assert os.path.exists(label_path), f"cannot find the {label_path}. The corresponding relationships between images and labels may be needed to check"

        if self.label_suffix == 'png':
            label = self.loader(label_path).convert('L')
        elif self.label_suffix == 'npy':
            label:np.ndarray = npy_loader(label_path) # [H, W, C]
            # convert ndarray to tensor for the following transformation
            label = torch.from_numpy(label).permute(2, 0, 1) # [H, W, C] -> [C, H, W]

        if self.transform is not None:
            img, label = self.transform(img, label)

        if self.dst_root is not None:
            extras = {'img_path': img_path, 'dst_path':dst_path}
        else:
            extras = {'img_path': img_path}
        return img, label, extras

class ClsImgs(Dataset):
    """
    The dataset for classification task.
    root: str, the dataset root dir
    split: str, the training split mode, training or test
    transform: the involved transformations, the data augmentation
    loader: func, the loader to load the image, default is the pil_loader
    few_shot: int, select the {few_shot} images from the training set.

    The folder structure should be:
    .
    ├── /XXX/FundusClassification/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── 1.jpg
    │   │   │   └── ...
    │   │   ├── test/
    │   │   │   ├── 2.jpg
    │   │   │   └── ...
    │   │   ├── training_labels.txt
    │   │   └── test_labels.txt
    │   ├── dataset_B/
    │   │   └── ....
    │   └── ....
    ├── /XXX/OCTClassification/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── 1.jpg
    │   │   │   └── ...
    │   │   ├── test/
    │   │   │   ├── 2.jpg
    │   │   │   └── ...
    │   │   ├── training_labels.txt
    │   │   └── test_labels.txt
    │   ├── dataset_B/
    │   │   └── ....
    │   └── ....
    where  the `training_labels.txt` contains the image path and its corresponding labels:
    # training_labels.txt
    training/1.jpg;0
    ....
    """
    def __init__(self, root:str, split, dst_root:str=None, transform=None, loader=pil_loader, few_shot=-1):
        self.root:str = root
        self.dst_root = dst_root # the root to save the extracted feats
        self.split = split
        assert self.split in ["training", "test"], f"unsupported split mode: {self.split}"
        self.transform = transform
        self.loader = loader
        self.few_shot = few_shot

        self._entries = []
        self._entries_cache = []

        # read all the images
        dataset_dir = self.root
        assert os.path.exists(dataset_dir), f'cannot find the dataset dir: {dataset_dir}'
        # find the sub-datasets under this task
        sub_datasets:list = utils.get_sub_dirs(dataset_dir)
        self.get_entries(sub_datasets)

        random.shuffle(self._entries)

        if self.few_shot != -1 and self.split == "training":
            self._entries = random.sample(self._entries, 1)
            print(f"select {self.few_shot} images for the few-shot setting")
        print(f"there are {len(self._entries)} images in the {self.split} set.")

    def get_entries(self, sub_datasts:list):
        # read all the images
        for sub_dataset in sub_datasts:
            print(f"reading images at {sub_dataset}")
            label_file = os.path.join(sub_dataset, f'{self.split}_labels.txt')
            self.read_txt(label_file, sub_dataset)
        print(f"there are {len(self._entries)} images in total")

    def read_txt(self, file_path, sub_dir):
        assert os.path.exists(file_path)
        with open(file_path, "r") as f:
            data = f.readlines()
        print(f"there are {len(data)} imgs in the {file_path}")
        for item in data:
            item = item.strip() # remove the space
            img_path, labels = item.split(';') # e.g. img_path: training/1.jpg
            src_path = os.path.join(sub_dir, img_path)
            assert os.path.exists(src_path), f"the {src_path} does not exists."

            if self.dst_root is None:
                dst_path = None
            else:
                dst_path = os.path.join(sub_dir.replace(self.root, self.dst_root), img_path)
            labels_list = labels.strip().split(',')
            labels_list = [float(label) for label in labels_list]
            self._entries.append((src_path, dst_path, labels_list))

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index: int):
        img_path, dst_path, labels = self._entries[index]
        # read image and label
        try:
            img = self.loader(img_path)
        except BaseException as e:
            print(e)
            print(f"will change the index to load next image")
            index = random.randint(0, len(self._entries)-1)
            img_path, dst_path, labels = self._entries[index]
            img = self.loader(img_path)

        if self.transform is not None:
            img = self.transform(img) # Tensor: [3, H, W]
        labels = torch.Tensor(labels) # labels is a list to contain the labels of different classification tasks

        if self.dst_root is not None:
            extras = {'img_path': img_path, 'dst_path':dst_path}
        else:
            extras = {'img_path': img_path}
        return img, labels, extras

class ClsFeats(Dataset):
    """
    The dataset for the classification decoder by reading extracted features.
    root: str, the dataset root dir
    split: str, the training split mode, training or test

    The folder structure should be:
    .
    ├── /XXX/FundusClsFeat/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── 1.pickle
    │   │   │   └── ...
    │   │   └── test/
    │   │       ├── 2.pickle
    │   │       └── ...
    │   ├── dataset_B/
    │   │   └── ....
    │   └── ....
    ├── /XXX/OCTClsFeat/
    │   ├── dataset_A/
    │   │   └── ...
    │   └── ...
    └── ...
    """
    def __init__(self, root:str, split:str, datasets:list):
        self.root = root
        self.split = split
        assert self.split in ["training", 'test']
        # the involved datasets (features)
        # self.datasets = ['FunCls_feat','UltraCls_feat', 'FFACls_feat', 'OCTCls_feat', 'SiltLampCls_feat']
        self.datasets = datasets

        print(f"there are {len(self.datasets)} in {self.root}: {self.datasets}")
        self._entries = []
        self.get_entries()

    def get_entries(self):
        for dataset in self.datasets:
            dataset_dir = os.path.join(self.root, dataset)
            sub_dirs = os.listdir(dataset_dir)
            for sub_dir in sub_dirs:
                dir_curr = os.path.join(dataset_dir, sub_dir, self.split)
                img_files = glob.glob(dir_curr + "/*")
                print(f"there are {len(img_files)} in {dir_curr}")
                self._entries += img_files
        print(f"the total involved images: {len(self._entries)}")

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, index: int):
        data_path = self._entries[index] # the pickle file
        data:dict = pickle.load(open(data_path, 'rb'))
        feats:list = data["feats"] # len=4, [[768, ]]
        labels = data["labels"] 

        # to tensor
        feats_th = [torch.from_numpy(item).unsqueeze(dim=0) for item in feats]
        feats_th = torch.cat(feats_th, dim=0) # [4, 768]
        labels_th = torch.from_numpy(labels) 

        return feats_th, labels_th

class ImageFolderDataset(datasets.ImageFolder):
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        extras = {'img_path': path}
        return sample, target, extras


class GFDataset(Dataset):
    def __init__(self, root, split, transform=None):

        with open(os.path.join(root, split + '.txt')) as f:
            self.data = f.readlines()
            self.data = [d.rstrip() for d in self.data]

        self.transform = transform

    def __getitem__(self, index):
        entry = self.data[index]
        img_path, label, time_diff = entry.split(', ')
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, int(label), int(time_diff)


    def __len__(self):
        return len(self.data)

    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
