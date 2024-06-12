

## Datasets

### 1. Crack500 dataset
The Crack500 dataset contains 500 images of size around 2000 × 1500 pixels taken by cell phones on main campus of Temple University. And each image was cropped into 16 non-overlapped image regions and only the region containing more than 1000 pixels of crack is kept. Through this way, the training data consists of 1,896 images, validation data contains 348 images, test data contains 1124 images. Download the Crack500 dataset from [this](https://github.com/fyangneil/pavement-crack-detection) link.
```
|-- Crack500
    |-- train
        |-- images
        |   |-- 20160222_081011_1_361.jpg
            ......
        |-- masks
        |   |-- 20160222_081011_1_361.png
            ......
    |-- test
        |-- images
        |   |-- 20160222_080933_361_1.jpg
            ......
        |-- masks
        |   |-- 20160222_080933_361_1.png
            ......
    |-- val
        |-- images
        |   |-- 20160222_080850_1_361.jpg
            ......
        |-- masks
        |   |-- 20160222_080850_1_361.png
            ......
```

### 2. DeepCrack dataset
The DeepCrack dataset is consist of 537 RGB color images with manually annotated segmentations. The images were divided into two main subsets: a training set with 300 images and a testing set with 237 ones. You can download the Deepcrack dataset from [this](https://github.com/yhlleo/DeepCrack) link.
```
|-- DeepCrack
    |-- train
        |-- images
        |   |-- 7Q3A9060-1.jpg
            ......
        |-- masks
        |   |-- 7Q3A9060-1.png
            ......
    |-- test
        |-- images
        |   |-- 11125-1.jpg
            ......
        |-- masks
        |   |-- 11125-1.png
            ......
```

## Training
```python
python train_crack.py
```
It will run training for model and save the best weights for the validation set.

## Testing
```python
python evaluate_crack.py --output <path_to_dir>
```
It will represent performance measures and will saves related results in `results` folder.


## Model weights
You can download the learned weights for each dataset in the following table.

|Dataset|Google Drive|Baidu Yun|
|:----|:----:|:----:|
|Crack500 |[link](https://drive.google.com/drive/folders/1QACf6O9TmtEVfqQeNldZJoG5UTAg27uQ?usp=share_link)|[link](https://pan.baidu.com/s/13nG4HepvFDTqUDlOUbH3Zw?pwd=e9b9)|
|Deepcrack |[link](https://drive.google.com/drive/folders/1QACf6O9TmtEVfqQeNldZJoG5UTAg27uQ?usp=share_link)|[link](https://pan.baidu.com/s/13nG4HepvFDTqUDlOUbH3Zw?pwd=e9b9)|

## Acknowledgment
CT-CrackSeg is based on following repos. We thanks for their great works:<br/>
- [TMUnet](https://github.com/rezazad68/TMUnet)<br/>
- [DeepSegmentor](https://github.com/yhlleo/DeepSegmentor)<br/>

```
