# VisionFM: A Vision Foundation Model for Generalist Ophthalmic Artificial Intelligence

The official implementation of VisionFM - a multimodal multitask vision foundation model, pre-trained using 3.4 million ophthalmic images from over 0.5 million subjects to enable generalist ophthalmic artificial intelligence. VisionFM is able to process eight common ophthalmic imaging modalities including fundus photography, optical coherence tomography (OCT), fundus fluorescein angiography (FFA), slit lamp, B-scan ultrasound, external eye imaging, MRI, and ultrasound biomicroscopy (UBM), and can be applied to solve various ophthalmic AI tasks such as ocular disease recognition, disease progression prediction, segmetation and detection of disease phenotypes and anatomical landmarks, as well as systemic biomarker and disease prediction. Functionality of VisionFM can be further extended beyond the current tasks by self-supervised pre-training of new imaging modalities and supervised fine-tuning of new clinical tasks, with the potential of addressing diverse, global ophthalmic diseases and different clinical challenges.  

## Latest News

- [2024/11] :tada: Congrats! VisionFM has been published in [NEJM AI](https://ai.nejm.org/doi/full/10.1056/AIoa2300221).
- [2024/05] The fine-tuning code has been released, along with fine-tuned weights on eight public multiclass disease recognition datasets


## 0. Install environment

Create the environment with conda commands:
```shell
conda create -n vfm python=3.8
conda activate vfm
```

Install the dependencies:
```shell
git clone https://github.com/ABILab-CUHK/VisionFM.git
cd VisionFM
pip install -r requirments.txt
```
## 1. Finetuning
If you want to utlize our weights to fine-tune on your data, please refer to this [instruction](./Fine-tuning/README.md).

## 2. Pretraining
In this step, you can pretrain your own VisionFM encoders on your data. Please follow the instructions below to start pretraining. 

### 2.1. Preparing the pretraining dataset

In our study, we used `8` modalities: `Fundus, OCT, External Eye, UBM, B-Ultrasound, MRI, Silt Lamp, and FFA`.
For each modality, e.g. Fundus, its data path should be `/xxx/Fundus/`, which contains all the Fundus images
 with the same or different suffix:

```
.
├── /dst_dir/Fundus/
│   ├── 1.jpg
│   ├── 2.png
│   └── ....
├── /dst_dir/OCT/
│   ├── 1.png
│   ├── 2.jpg
│   └── ...
└── ...
 
```

you can run the following command to generate random images if you do not have fundus photographs at hand:
```shell
cd evaluation
python random_data.py --task pretrain --dst_dir ../dataset/pretrain_random
```

### 2.2. Pretraining the VisionFM encoders

1. Train `vit-base` on the modality `Fundus`:

```shell
# run the following commands to train the Fundus encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29500 main_pretrain.py \
--local-rank=0 \
--data_path ./dataset/pretrain_random \
--modality Fundus \ 
--norm_last_layer true \
--epochs 400 \
--batch_size_per_gpu 12 \
--shared_head true \
--out_dim 8192 \
--output_dir ./results \
--global_crops_scale 0.32 1.0 \
--pred_ratio 0 0.3 \
--pred_ratio_var 0 0.2 \
--name Train_Random_Fundus \
--load_pretrain > train_fundus.log 2>&1 &
# or 
bash train_vitb_fundus.sh # contain the same command

# Attention: the defaualt batch size is 128, batch_size=12 is only for debugging.
```

By changing modalities, different VisionFM encoders can be trained.

## 3. Training decoders for downstream tasks

### 3.1. Download the pretrained weights of VisionFM
Please download the corresponding model weight based on the modality that you want to carry out research.

| Modality   | Google Drive                                                                                      |
|------------|---------------------------------------------------------------------------------------------------|
| Fundus     | [Download](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view?usp=sharing) |
| OCT        | [Download](https://drive.google.com/file/d/1o6E-ine2QLx2pxap-c77u-SU0FjxwypA/view?usp=sharing) |
| FFA        | [Download](https://drive.google.com/file/d/128izBUNV00Ojb9w9Dq3GhBvhWqzU-mla/view?usp=sharing) |
| Ultrasound | [Download](https://drive.google.com/file/d/1IlD0snowxdEVvxmiIBZGR0D9uOcrCT2D/view?usp=sharing) |
| External Eye  | [Download](https://drive.google.com/file/d/16zGHTD4ZcGAYW382kKHBw3TU6D1OtvTD/view?usp=sharing) |
| Silt Lamp   | [Download](https://drive.google.com/file/d/1pemWDkGoZYlqLQ6ooFINktyk8xnv9wY_/view?usp=sharing) |
| MRI        | [Download](https://drive.google.com/file/d/1fcfylnOWhfnZHBAKT9pQPufyS5ZYCXu0/view?usp=sharing) |
| UBM        | [Download](https://drive.google.com/file/d/1q2fVOgFBnWNu1BsXaza1A-OIcCiifNUQ/view?usp=sharing) |


### 3.2.Training a classification decoder [Multi-modality]

The `Multi-modality` means the decoder is trained on different modalities simultaneously. Considering the existence of different encoders (each modality has its own encoder),
We adopt a two-stage pipeline: `pre-extracting features from VisionFM encoders of different modalities` and `training the decoder using the aggragated image features`:

For the first step, we need to extract image features based on their modalities. For example, we can extract the Fundus and OCT features through Fundus and OCT encoders respectively.
Then, for the second step, we can start training the decoder using the combined features extracted from these two modalities.

#### 3.2.1. Preparing the dataset
Please organize your dataset into the following directory structure (we call this directory structure style as `vfm`):
```
.
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
...
```

The corresponding `training_labels.txt` and `test_labels.txt` are organized as follows as an example:
```
# class list: ['Healthy', 'DR-1', 'DR-2', 'DR-3', 'DR-4', 'DR', 'Glaucoma', 'AMD', 'Cataract', 'Hypertensive Retinopathy', 'Retinal Vein Occlusion', 'Myopia', 'Retinal Detachment']
# in training_labels.txt
# Attention: The labels are suggested to use a multilabel style (not one-hot) to facilitate recognition on an image labelled with multiple diseases and also simplify the use of graded and non-graded data.
training/1.jpg;0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0
training/2.jpg;0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0
```

You can download our preprocessed public [dataset](https://drive.google.com/file/d/1QShoYrkhZetF41vmFuf6ds3I1W05YONk/view?usp=drive_link) (containing IDRiD and OCTID datasets) to start the training. 
After unzipping the downloaded dataset, please organize the dataset using the following structure:
```text
./dataset/ProcessedDatasets/MultiModalCls/IDRiD
./dataset/ProcessedDatasets/MultiModalCls/OCTID
```

Or you can generate random dataset using the following command:
```shell
python evaluation/random_data.py --task multi_cls --dst_dir ./dataset/multi_cls_random
```

#### 3.2.2.  Feature extraction
The following command extracts image features using the pretrained VisionFM encoders:

```shell
#cd evaluation
#extract the Fundus and OCT features through Fundus and OCT encoders respectively
#CUDA_VISIBLE_DEVICES=0 nohup python evaluation/extract_features.py \
# extract Fundus features
CUDA_VISIBLE_DEVICES=1,2 nohup python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/multi_cls_random/FundusClassificationMulti/ \
--modality Fundus \
--dst_root ./dataset/multi_cls_random/FunClsFeat/ > extract_feats.log 2>&1 &

# extract OCT features
CUDA_VISIBLE_DEVICES=1,2 nohup python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_OCT_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/multi_cls_random/OCTClassificationMulti/ \
--modality OCT \
--dst_root ./dataset/multi_cls_random/OCTClsFeat/ > extract_feats.log 2>&1 &

# for the provided preprocessed datasets, you should set the following param:
--data_path ./dataset/ProcessedDatasets/MultiModalCls/FundusClassificationMulti/
--dst_root ./dataset/ProcessedDatasets/MultiModalCls/FunClsFeat/
# or
--data_path ./dataset/ProcessedDatasets/MultiModalCls/OCTClassificationMulti/
--dst_root ./dataset/ProcessedDatasets/MultiModalCls/OCTClsFeat/
```

#### 3.2.3. Training a decoder based on extracted multimodal features
Then, train the classification decoder by the following command:
```shell
#cd evaluation
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_multi_decoder.py \
--name train_debug \ 
--output_dir ./results \
--datasets FunClsFeat OCTClsFeat \
--data_path ./dataset/multi_cls_random/ \
--batch_size_per_gpu 8192 > train_cls_multi.log 2>&1 &
```

### 3.3. Training a classification decoder [Single-modality]

This task mainly focuses on single modality, such as Fundus based DR grading task.

#### 3.3.1. Preparing the dataset
Please organize your dataset into the following directory structure (we call this directory structure style as `vfm`):
```
.
├── /XXX/FundusClassification/ # all dataset should be the same task with same class definition
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
...
```

The `training_labels.txt` and `test_labels.txt` contains the image path and its corresponding labels:
```text
# in training_labels.txt
training/1.jpg;1
training/2.jpg;2
```

You can download our preprocessed [dataset](https://drive.google.com/file/d/1QShoYrkhZetF41vmFuf6ds3I1W05YONk/view?usp=drive_link) (containing the processed IDRiD and OCTID to start the training. 
After unzipping the downloaded dataset, you should organize the dataset using the following structure:
```text
./dataset/ProcessedDatasets/SingleModalCls/FundusClassification/IDRiD 
./dataset/ProcessedDatasets/SingleModalCls/OCTClassification/OCTID
```

Or you can generate random dataset by using the following command:
```shell
python evaluation/random_data.py --task single_cls --dst_dir ./dataset/single_cls_random # for DR grading task
```

Except for the mentioned directory structure (called vfm), you can also use the following directory structure (ImageNet format):
```test
├── data folder
    ├──train
        ├──class_a
        ├──class_b
        ├──class_c
    ├──val
        ├──class_a
        ├──class_b
        ├──class_c
    ├──test
        ├──class_a
        ├──class_b
        ├──class_c
```
If your datasets are organized as this structure, please set `--dataset_format ImageNet`.


#### 3.3.2. Training the decoder
Then, train the decoder for the classification task by the following command:
```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/single_cls_random/FundusClassification/ \
--num_labels 5 \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

#Attention: set --num_labels 1 for the binary classification task

# for processed dataset: Fundus based DR grading
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \ 
--data_path ./dataset/ProcessedDatasets/SingleModalCls/FundusClassification  \
--num_labels 5 \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

# for dataset with ImageNet format: Fundus based DR grading
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--dataset_format ImageNet \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/xxx/  \
--num_labels 5 \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

# for processed dataset: OCT based binary classification (Health, DR)
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_OCT_weights.pth \
--output_dir ./results \
--data_path ./dataset/ProcessedDatasets/SingleModalCls/OCTClassification  \
--num_labels 1 \
--modality OCT \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &

# after the training of decoder is completed, you can use the following command to evaluate the trained decoder
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 evaluation/train_cls_decoder.py \
--name single_cls_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/ProcessedDatasets/SingleModalCls/FundusClassification  \
--num_labels 5 \
--load_from ./results/single_cls_debug/checkpoint_teacher_linear.pth\
--test \
--batch_size_per_gpu 32 > train_single_cls.log 2>&1 &


# you can also load the RETFound weights by adding two additional params:
--arch vit_large \
--checkpoint_key model \
```

### 3.4.Training a segmentation decoder [Single-modality]
In segmentation task, we train different decoders for different tasks and modalities.

#### 3.4.1. Preparing the dataset
Please organize your dataset into the following directory structure (we call this directory structure style as `vfm`):
```
├── /dst_dir/VesselSegmentation/ # all dataset should be the same task, e.g., fundus vessel segmentation
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
The range of pixel values in images in the labels directory should be [0, C-1], where C is the number of classes. 
```
You can download our preprocessed public [dataset](https://drive.google.com/file/d/1QShoYrkhZetF41vmFuf6ds3I1W05YONk/view?usp=drive_link) (contain DRIVE dataset for vessel segmetnation) to start the training. 
After unzipping the downloaded dataset, please organize the dataset as the following structure:
```text
./dataset/ProcessedDatasets/SingleModalSeg/VesselSegmentation/DRIVE 
```
Or you can generate a random dataset by using the following command:
```shell
python evaluation/random_data.py --task segmentation --dst_dir ./dataset/seg_random
```

#### 3.4.2. Training the decoder
Then, train the decoder for the segmentation task by the following command:
```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 evaluation/train_seg_decoder.py \
--name single_seg_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--input_size 512 \
--modality Fundus \
--num_labels 5 \
--output_dir ./results \
--data_path ./dataset/seg_random/VesselSegmentation/ \
--batch_size_per_gpu 5 > train_seg.log 2>&1 &

# for the provided preprocessed datasets, you should set the following param:
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29509 evaluation/train_seg_decoder.py \
--name single_seg_debug \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--input_size 512 \
--modality Fundus \
--num_labels 1 \
--output_dir ./results \
--data_path ./dataset/ProcessedDatasets/SingleModalSeg/VesselSegmentation/ \
--batch_size_per_gpu 5 > train_seg.log 2>&1 &
```


### 3.5. Training a landmark detection decoder [Single-modality]
In this task, we train a decoder to detect the anterior chamber angle (ACA) landmarks on UBM images. 

#### 3.5.1. Preparing the dataset
Please organize the dataset into the same directory structure as that of segmentation tasks (the suffix of labels should be .npy).

Or you can generate a random dataset by using the following command:
```shell
python evaluation/random_data.py --task landmark --dst_dir ./dataset/landmark_random
```

#### 3.5.2. Training the decoder
Then, train the decoder for the landmark detection task by using the following command:
```shell
CUDA_VISIBLE_DEVICES=0 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29508 evaluation/train_landmark_decoder.py \
--name train_landmark \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--output_dir ./results \
--data_path ./dataset/landmark_random/LandmarkDetection \
--batch_size_per_gpu 32  > train_landmark.log 2>&1 &
```

### 3.6. Training a biomarker prediction decoder [Multi-modality]
In our experiments, we train the decoder on Fundus and External images to predict biomarker prediction. 

#### 3.6.1. Preparing the dataset
Please organize the dataset into the following directory structure (we call this directory structure style as `vfm`), which is the same as classification task:
```
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
```
The corresponding `training_labels.txt` and `test_labels.txt` are organized as follows (38 values):
```
# in training_labels.txt
training/1.jpg;38.8,2.5,37.0,11.4,8.9,0.05,0.4,0.13,1.1,46.6,157.0,3.87,31.3,32.8,337.0,..., 3.4,4.13,0.93,2.62,3.17
```
Or you can generate random dataset by using the following command:
```shell
python evaluation/random_data.py --task metric_reg --dst_dir ./dataset/metric_random
```

#### 3.6.2. Extracting features
First, extract the image features using the following commands:
```shell
# extract the features for Fundus images
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_Fundus_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/metric_random/FundusRegression \
--modality Fundus \
--dst_root ./dataset/metric_random/FunRegFeat/ > extract_feats.log 2>&1

#extract the features for External images
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=29503 evaluation/extract_features.py \
--pretrained_weights ./pretrain_weights/VFM_External_weights.pth \
--batch_size_per_gpu 768 \
--data_path ./dataset/metric_random/ExternalRegression \
--modality External \
--dst_root ./dataset/metric_random/ExternalRegFeat/ > extract_feats.log 2>&1
```

#### 3.6.3. Training the decoder using extracted features
Then, train the biomarker prediction decoder using the following command:
```shell
CUDA_VISIBLE_DEVICES=0,1 nohup python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=29504 evaluation/train_metric_reg_multi_decoder.py  \
--name train_metric_reg_multi \
--output_dir ./results \
--datasets FunRegFeat ExternalRegFeat \
--data_path ./dataset/metric_random/ \
--batch_size_per_gpu 4096 > train_metric_reg_multi.log 2>&1 &

```


### 3.7. Training a glaucoma progression forcasting decoder [Single-modality]

#### 3.7.1 Preparing the data
Please arganize the data into the following structure:
```

├── /dataset/glaucoma_forecasting/
│   ├── training/
│   │   ├── 1.jpg
│   │   └── ...
│   ├── test/
│   │   ├── 1.jpg
│   │   └── ...
│   ├── training_labels.txt
│   └── test_labels.txt

```
The corresponding `training_labels.txt` and `test_labels.txt` are organized as follows (path/to/image, label, time interval):
```
# in training_labels.txt
./dataset/glaucoma_forecasting/training/1.jpg, 0, 309
# in test_labels.txt
./dataset/glaucoma_forecasting/test/1.jpg, 0, 690
```


#### 3.7.2 Training the decoder
The glaucoma forecasting decoder can be trained using the following command:

```shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch evaluation/train_forecasting_decoder.py  --data_path ./dataset/glaucoma_forecasting/ --pretrained_weights /path/to/checkpoint.pth --n_last_blocks 4 --avgpool_patchtokens 1 --input_size 224 --checkpoint_key teacher --output_dir ./results/glaucoma_forecasting --num_labels 2 --lr 0.001 --batch_size_per_gpu 128 --epochs 100
```


## 4. VisionFM private evaluation data and synthetic images

Synthetic images and a subset of our private evaluation data can be accessed. Please download [the data request and agreement form](resource/visionfm_dataset_agreement_form.pdf), sign and email it to visionfm-datasets@googlegroups.com

## Citation
If you find this repository useful, please consider citing this paper:
```text
@article{qiu2024development,
  title={Development and validation of a multimodal multitask vision foundation model for generalist ophthalmic artificial intelligence},
  author={Qiu, Jianing and Wu, Jian and Wei, Hao and Shi, Peilun and Zhang, Minqing and Sun, Yunyun and Li, Lin and Liu, Hanruo and Liu, Hongyi and Hou, Simeng and others},
  journal={NEJM AI},
  volume={1},
  number={12},
  pages={AIoa2300221},
  year={2024},
  publisher={Massachusetts Medical Society}
}
```
## LICENSE
This project is released under a license that permits use for research and educational purposes only. Commercial use of this model is not allowed. Please ensure that you comply with the terms of this license when using the model. For more information, refer to the LICENSE file included in this repository.

