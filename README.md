# PointNet
PointNet implementation on Windows x86 (pyTorch version, most of codes are from Fxia22).

charlesq34: https://github.com/charlesq34/pointnet ////// https://arxiv.org/abs/1612.00593

Fxia22's wonderful work: https://github.com/fxia22/pointnet.pytorch.

#### How long do I need to train the mosel? ####
- Within 40 min for classification task (?? I do not remember LoL). Segmentation will be much faster!

#### How to run ####
enter the project directory\utils
python train_classification.py --dataset=data_directory --nepoch=10
python train_segmentation.py --dataset=data_directory --nepoch=5 --outf=path_to_save_trained_model

#### Other chatter ####
- This code can runs on a Windows PC with GPU enabled. No need for Linux, no need for virtual machine, no need for OpenCV tools, etc.

- From the last part of the .pdf report file, you can find a lot of useful solutions I made during the code reproduction. I believed it would save you a lot of time.

- Have pytorch installed in advanced. I used anaconda-environments to manage the modules and PyCharm as IDE.

- The dataset used: http://stanford.edu/~rqi/pointnet/

- Change the path to make sure the code can find the pointnet module and the dataset. // Follow the error reported.

- Can perform classification and segmentation tasks

#### ABOUT VISUALIZATION TOOL ####
- The visualization tools (to visualize segmentation result) given by charlesq34 can only run on Linux (but it was written in C++, so I managed to build a .dll so it can now run on a Win).

- If the visualization tool did not work for you, see the report, it will tell you how to make the .dll file [really simple]


#### IMPORTANT ####
- Reduce the batch_size if lack of memory issue is reported.
- When changeing the batch_size, REMEMBER to check the value in train_classification.py line139. You need to understand why I changed the code here (compared to Fxia22).
- Wish you good luck

The segmentation result visualization:
![img](https://github.com/saaries/PointNet/blob/master/resources_for_report/seg_result.gif)

Some segmentation results:
![img](https://github.com/saaries/PointNet/blob/master/resources_for_report/stich.png)
