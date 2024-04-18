from torchvision import transforms
import random
import torch
import numpy as np
from utils import get_stats
from torchvision.transforms import functional as tvF

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for t in self.transforms:
                image = t(image)
            return image
        else:
            for t in self.transforms:
                image, target = t(image, target)
            return image, target

# create custom class transform
class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img, target=None):
        # img and gt are PIL.Image
        # assert img.size == gt.size
        # fix parameter
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        # return the image with the same transformation
        img_transformed = tvF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        if target is None:
            return img_transformed
        gt_transformed = tvF.resized_crop(target, i, j, h, w, self.size, transforms.InterpolationMode.NEAREST)
        return img_transformed, gt_transformed

class RandomCrop(transforms.RandomCrop):
    def __call__(self, img, target=None):
        i, j, h, w = self.get_params(img, self.size)

        img_transformed = tvF.crop(img, i, j, h, w)
        if target is None:
            return img_transformed
        gt_transformed = tvF.crop(target, i, j, h, w)
        return img_transformed, gt_transformed

class RandomAffine(transforms.RandomAffine):
    def __call__(self, img, target=None):

        if random.random() > 0.5: # 0.5 probality to apply this transform
            return img if target is None else img, target

        fill = self.fill
        if isinstance(img, torch.Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * tvF.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = tvF.get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)

        img_transformed = tvF.affine(img, *ret, interpolation=transforms.InterpolationMode.BILINEAR, fill=fill, center=self.center)
        if target is None:
            return img_transformed
        gt_transformed = tvF.affine(target, *ret, interpolation=transforms.InterpolationMode.NEAREST, fill=fill, center=self.center)
        return img_transformed, gt_transformed

class ColorJitter(transforms.ColorJitter):
    def __call__(self, img, target=None):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = tvF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = tvF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = tvF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = tvF.adjust_hue(img, hue_factor)
        if target is None:
            return img
        else:
            return img, target

class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if target is None:
            if random.random() < self.flip_prob:
                image = tvF.hflip(image)
            return image
        else:
            if random.random() < self.flip_prob:
                image = tvF.hflip(image)
                target = tvF.hflip(target)
            return image, target

class RandomVerticalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if target is None:
            if random.random() < self.flip_prob:
                image = tvF.vflip(image)
            return image
        else:
            if random.random() < self.flip_prob:
                image = tvF.vflip(image)
                target = tvF.vflip(target)
            return image, target

class ToTensor(object):
    def __call__(self, image, target=None):
        image = tvF.to_tensor(image)
        if target is None:
            return image
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Resize:
    def __init__(self, size, interpolation=transforms.InterpolationMode.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image, target=None):
        img_transformed = tvF.resize(image, size=self.size, interpolation=self.interpolation)
        if target is None:
            return img_transformed
        gt_transformed = tvF.resize(target, size=self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return img_transformed, gt_transformed

class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = tvF.center_crop(image, self.size)
        if target is None:
            return image
        target = tvF.center_crop(target, self.size)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = tvF.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target

class NormalizeMix(object):
    def __init__(self):
        pass

    def __call__(self, image, modality:str):
        mean, std = get_stats(modality)
        image = tvF.normalize(image, mean=mean, std=std)
        return image