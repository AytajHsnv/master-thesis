## Crack segmentation for concrete pipes
In this work, the configuration of cameras for novel inspection
systems and the semantic segmentation of cracks in pipes are undertaken. The
convolutional transformer architectures TransMUNetand DeepLabV3+ are
used as the segmentation model. Additionally, a Dilated Residual Block and a
Boundary Awareness Module are included to capture details of cracks and boundary
features, respectively. The TransMUNet model has already been evaluated using
two publicly available datasets: Crack500 and DeepCrack [2]. Furthermore, all
DeepLabV3+ models have been trained with the Crack500 and DeepCrack datasets,
in addition to TransMUNet. In addition, experiments were conducted to identify
optimal settings for photographing the cracks in the pipes. In the experimentation
process, a series of images were captured in a variety of settings, which were then
utilized as a dataset to test the model’s performance on the concrete pipes. The
optimal settings, determined by the experiments, include the distance from the
camera to the wall of the pipe and the illumination angle. The initial testing phase
revealed that the DeepLabV3+ model with a ResNet-50 backbone exhibited an
IoU (Intersection-over-Union) of approximately 31% and 24% for the dry and wet
conditions of the concrete surfaces, respectively. However, the performance was  

## Datasets
## Weights

### 1. Crack500 dataset
The Crack500 dataset contains 500 images of size around 2000 × 1500 pixels taken by cell phones on main campus of Temple University. And each image was cropped into 16 non-overlapped image regions and only the region containing more than 1000 pixels of crack is kept. Through this way, the training data consists of 1,896 images, validation data contains 348 images, test data contains 1124 images. Download the Crack500 dataset from [this](https://github.com/fyangneil/pavement-crack-detection) link. Here images and masks are in the same folder, however they are distingushed by file endings (jpg and png, respectively). 


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
### 3. Custom dataset
Custom made dataset is availabe by request. This dataset is used to test the semantic segmentation model and obtain optimal illumination angle.
Angles are 10, 20, 45, 75 and 90 degree. Working distances start from 12.5 cm  until 60 cm. 

## Weights
You can upload them to the checkpoints folder and edit the file names, hyperparameters in config_crack.yml. 


## Training
```python
python train_crack.py
```
It will run training for model and save the best weights for the validation set.

## Testing
```python
python evaluate_crack.py --output <path_to_dir>
```
It will represent performance measures and will saves related results in `results` folder. Plots will be added to the same directory as evaluate_crack.py. Test set of of the DeepCrack and Crack500 do not have angles so, in order to use test set of these dataset, following part should be commented out. 
data_path = config['path_to_testdata']
DIR_IMG  = os.path.join(data_path)
DIR_MASK = os.path.join(data_path)
img_names  = [path.name for path in Path(DIR_IMG).glob('*.jpg')]
mask_names = [path.name for path in Path(DIR_MASK).glob('*.png')]
img_names= natsorted(img_names)
mask_names=natsorted(mask_names)


## Acknowledgment
This paper is based on following repos:<br/>
- [TransMUNet](https://github.com/HqiTao/CT-crackseg)<br/>
- [DeepLabV3+](https://github.com/VainF/DeepLabV3Plus-Pytorch)<br/>

```
