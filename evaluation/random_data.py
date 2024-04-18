# randomly generate toy data 
import os
import random

import cv2
import tqdm
import argparse
import numpy as np

def check_dir(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def pretrain_data(args):
    """
    randomly generating the images used for the pretraining data.
    For each modality, its corresponding dir should be root/modality, e.g. /data/Fundus/XXX.jpg.
    .
    ├── /dst_dir/Fundus/
    │   ├── 1.png
    │   └── ...
    ├── /dst_dir/OCT/
    │   ├── 1.png
    │   └── ....
    └── ....
    '''
    In this func, we will only generate the Fundus modality as the example.
    """
    dst_root = args.dst_dir
    modalities = ['Fundus', 'OCT', 'External', 'UltraSound', 'FFA', 'SiltLamp'] 
    for modality in modalities:
        # modality = 'Fundus'
        target_dir = os.path.join(dst_root, modality)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        print(f"Will generate {args.num} {modality} images at {target_dir}")

        for idx in tqdm.tqdm(range(args.num)):
            img = np.random.randint(0, 255, size=[args.img_size, args.img_size, 3]) # [H, W, 3]
            img_path = os.path.join(target_dir, "{:0>8}.png".format(idx))
            cv2.imwrite(img_path, img)
            # print(f"save {img_path} done. ")
        print(f"Generate {modality} data done at {target_dir}")
def segmentation_data(args):
    """
    This func will generate random images for the retinal vessel segmentation task.
    The folder structure should be:

    ├── /dst_dir/VesselSegmentation/ # all dataset should be the same task
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── images/
    │   │   │   │   ├── 1.png
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
    """
    dst_root = args.dst_dir
    task = 'VesselSegmentation'
    target_dir = os.path.join(dst_root, task, 'dataset_A')
    check_dir(target_dir)
    print(f"Will generate {task} images at {target_dir}")

    splits = ['training', 'test']
    for split in splits:
        print(f"Generate {split} set...")
        for idx in tqdm.tqdm(range(args.num)):
            img_dir = os.path.join(target_dir, split, 'images')
            check_dir(img_dir)
            mask_dir = os.path.join(target_dir, split, 'labels')
            check_dir(mask_dir)
            img = np.random.randint(0, 255, size=[args.img_size, args.img_size, 3])  # [H, W, 3]
            # mask = np.random.randint(0, 2, size=[args.img_size, args.img_size], dtype=np.uint8) # [H, W], the binary mask
            mask = np.random.randint(0, 5, size=[args.img_size, args.img_size], dtype=np.uint8) # [H, W], the multi-class mask
            # save images
            cv2.imwrite(os.path.join(img_dir, "{:0>8}.png".format(idx)), img)
            cv2.imwrite(os.path.join(mask_dir, "{:0>8}.png".format(idx)), mask)
    print(f"Generate {task} data done in {target_dir}")

def landmark_data(args):
    """
    This func will generate random images and labels for the landmark detection task.
    The folder structure should be:

    ├── /dst_dir/LandmarkDetection/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── images/
    │   │   │   │   ├── 1.png
    │   │   │   │   └── ...
    │   │   │   └── labels/
    │   │   │       ├── 1.npy # Attention the suffix
    │   │   │       └── ...
    │   │   └── test/
    │   │       └── ...
    │   ├── dataset_B/
    │   │   └── ...
    │   └── ...
    └── ...
    The channels of xxx.npy in labels directory should be equal to the number of landmarks.
    """
    dst_root = args.dst_dir
    task = 'LandmarkDetection'
    target_dir = os.path.join(dst_root, task, 'dataset_A')
    check_dir(target_dir)
    print(f"Will generate {task} images at {target_dir}")

    splits = ['training', 'test']
    for split in splits:
        print(f"Generate {split} set...")
        for idx in tqdm.tqdm(range(args.num)):
            img_dir = os.path.join(target_dir, split, 'images')
            check_dir(img_dir)
            mask_dir = os.path.join(target_dir, split, 'labels')
            check_dir(mask_dir)
            img = np.random.randint(0, 255, size=[args.img_size, args.img_size, 3])  # [H, W, 3]
            mask = np.zeros([args.img_size, args.img_size, 3], dtype=np.uint8)# [H, W, 3], represent 3 points
            for i in range(3): # set the target point
                mask[np.random.randint(0, args.img_size), np.random.randint(0, args.img_size), i] = 1

            # save images
            cv2.imwrite(os.path.join(img_dir, "{:0>8}.png".format(idx)), img)
            np.save(os.path.join(mask_dir, "{:0>8}.npy".format(idx)), mask)
    print(f"Generate {task} data done in {target_dir}")

def classification_data(args):
    """
    This func will generate random images and labels for the single modal classification task.
    The folder structure should be:
    ├── /XXX/FundusClassification/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── 1.png
    │   │   │   └── ...
    │   │   ├── test/
    │   │   │   ├── 2.jpg
    │   │   │   └── ...
    │   │   ├── training_labels.txt
    │   │   └── test_labels.txt
    │   ├── dataset_B/
    │   │   └── ....
    │   └── ....

    The `training_labels.txt` and `test_labels.txt` contains the image path and its corresponding labels:
    # in training_labels.txt for DR grading task
    training/1.jpg;1
    training/2.jpg;2
    ....

    This script would generate random images and labels for Fundus based DR grading task.
    """
    dst_root = args.dst_dir

    task = 'FundusClassification'
    target_dir = os.path.join(dst_root, task, 'dataset_A')
    check_dir(target_dir)
    print(f"Will generate {task} images at {target_dir}")

    splits = ['training', 'test']
    for split in splits:
        print(f"Generate {split} set...")
        # the labels text
        label_path = os.path.join(target_dir, f"{split}_labels.txt")
        label_write = open(label_path, 'w')
        for idx in tqdm.tqdm(range(args.num)):
            img_dir = os.path.join(target_dir, split)
            check_dir(img_dir)
            img = np.random.randint(0, 255, size=[args.img_size, args.img_size, 3])  # [H, W, 3]

            # save images
            f_name = "{:0>8}.png".format(idx)
            labels_str = str(np.random.randint(0, 5)) # for DR grading task
            # labels_str = str(np.random.randint(0, 1)) # for binary classification task
            label_write.write(f"{split}/{f_name};{labels_str}\n")
            cv2.imwrite(os.path.join(img_dir, f_name), img)
        label_write.close()
    print(f"Generate {task} data done in {target_dir}")

def multi_classification_data(args):
    """
    This func will generate random images and labels for the multi-modal classification task.
    The folder structure should be:
    ├── /XXX/FundusClassification/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── 1.png
    │   │   │   └── ...
    │   │   ├── test/
    │   │   │   ├── 2.png
    │   │   │   └── ...
    │   │   ├── training_labels.txt
    │   │   └── test_labels.txt
    │   ├── dataset_B/
    │   │   └── ....
    │   └── ....
    ├── /XXX/OCTClassification/
    │   ├── dataset_A/
    │   │   ├── training/
    │   │   │   ├── 1.png
    │   │   │   └── ...
    │   │   ├── test/
    │   │   │   ├── 2.png
    │   │   │   └── ...
    │   │   ├── training_labels.txt
    │   │   └── test_labels.txt
    │   ├── dataset_B/
    │   │   └── ....
    │   └── ....
    ....

    The `training_labels.txt` and `test_labels.txt` contains the image path and its corresponding label:
    # class list: ['Healthy', 'DR-1', 'DR-2', 'DR-3', 'DR-4', 'DR', 'Glaucoma', 'AMD', 'Cataract', 'Hypertensive Retinopathy', 'Retinal Vein Occlusion', 'Myopia', 'Retinal Detachment']
    # in training_labels.txt
    # Attention: The labels are suggested to use a multilabel style (not one-hot) to facilitate recognition on an image labelled with multiple diseases and also simplify the use of graded and non-graded data.
    training/1.jpg;0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
    training/2.jpg;0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
    ...

    You can set an arbitrary number of labels by yourself. Remember to modify the corresponding dataset classes to suit your modifications.
    In this code, we will generate labels of 13 values.
    """
    dst_root = args.dst_dir
    tasks = ['FundusClassificationMulti', 'OCTClassificationMulti']
    for task in tasks:
        target_dir = os.path.join(dst_root, task, 'dataset_A')
        check_dir(target_dir)
        print(f"Will generate {task} images at {target_dir}")

        splits = ['training', 'test']
        for split in splits:
            print(f"Generate {split} set...")
            # the labels text
            label_path = os.path.join(target_dir, f"{split}_labels.txt")
            label_write = open(label_path, 'w')
            for idx in tqdm.tqdm(range(args.num)):
                img_dir = os.path.join(target_dir, split)
                check_dir(img_dir)
                img = np.random.randint(0, 255, size=[args.img_size, args.img_size, 3])  # [H, W, 3]
                # save images
                f_name = "{:0>8}.png".format(idx)
                # if 'Multi' in task: # for multi-modal classification task
                labels = ['0' for item in range(13)]
                labels[np.random.randint(0, 5)] = '1'
                if labels[0] == '0': labels[5] = '1' # DR

                labels_str = ','.join(labels)
                label_write.write(f"{split}/{f_name};{labels_str}\n")
                cv2.imwrite(os.path.join(img_dir, f_name), img)
            label_write.close()
        print(f"Generate {task} data done in {target_dir}")

def metric_reg(args):
    """
    This func will generate random images and labels for a regression task, e.g., predicting biomarker values from a given image.
    The folder structure should be:
    .
    ├── /XXX/FundusRegression/
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
    ├── /XXX/ExternalRegression/
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
    ...
    The corresponding `training_labels.txt` and `test_labels.txt` are organized as follows (38 values):
    ```
    # in training_labels.txt
    training/1.jpg;38.8,2.5,37.0,11.4,8.9,0.05,0.4,0.13,1.1,46.6,157.0,3.87,31.3,32.8,337.0,..., 3.4,4.13,0.93,2.62,3.17
    ```
    """
    dst_root = args.dst_dir
    tasks = ['FundusRegression', 'ExternalRegression']
    for task in tasks:
        target_dir = os.path.join(dst_root, task, 'dataset_A')
        check_dir(target_dir)
        print(f"Will generate {task} images at {target_dir}")

        splits = ['training', 'test']
        for split in splits:
            print(f"Generate {split} set...")
            # the labels text
            label_path = os.path.join(target_dir, f"{split}_labels.txt")
            label_write = open(label_path, 'w')
            for idx in tqdm.tqdm(range(args.num)):
                img_dir = os.path.join(target_dir, split)
                check_dir(img_dir)
                img = np.random.randint(0, 255, size=[args.img_size, args.img_size, 3])  # [H, W, 3]

                # save images
                f_name = "{:0>8}.png".format(idx)
                labels = [str(100*random.random()) for item in range(38)] # for the regression task
                labels_str = ",".join(labels)
                
                label_write.write(f"{split}/{f_name};{labels_str}\n")
                cv2.imwrite(os.path.join(img_dir, f_name), img)
            label_write.close()
        print(f"Generate {task} data done in {target_dir}")

def parse_args():
    parser = argparse.ArgumentParser('Generate the random image data')
    # Add an argument for the task to be performed
    parser.add_argument('--task', type=str, default='pretrain', choices=['pretrain', 'multi_cls','segmentation', 'single_cls','landmark', 'metric_reg'], help='generate toy data for a type of task')
    # Add an argument for the destination directory to save the generated images
    parser.add_argument('--dst_dir', type=str, default='./', help='save dir for the generated images')
    # Add an argument for the number of generated images
    parser.add_argument('--num', type=int, default=500, help='the number of the generated images')
    # Add an argument for the image size of the generated images
    parser.add_argument('--img_size', type=int, default=512, help='the image size of generate images')
    # Parse the arguments
    args = parser.parse_args()
    # Return the arguments
    return args

if __name__ == "__main__":
    args = parse_args()
    assert args.num >  0, f"the number of the generated images should be greater than 0"

    print(f"Will generate the random datasets for {args.task}")
    if args.task == 'pretrain':
        pretrain_data(args) # Generate toy pre-training data
    elif args.task == 'multi_cls': # Generate toy data for multi-modal classification task dataset
        multi_classification_data(args)
    elif args.task == 'single_cls': # Generate toy data for single-modal classification task dataset
        classification_data(args)
    elif args.task == 'segmentation': # Generate toy data for segmentation task
        segmentation_data(args)
    elif args.task == 'landmark': # Generate toy landmark detection data
        landmark_data(args) 
    elif args.task == 'metric_reg': # Generate toy regression data
        metric_reg(args)

    pass
