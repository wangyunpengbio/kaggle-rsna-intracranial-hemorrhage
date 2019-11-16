# RSNA Intracranial Hemorrhage Detection

This is the project for [RSNA Intracranial Hemorrhage Detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) hosted on Kaggle in 2019. The code was mostly from [appian42](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage). However, I have changed the augmentation methods, learning rate and network backbone, ensembling three different models and achieveing about 0.065 on Public Leaderboard. In stage2, this code can get into the silver medal area.(58th, Private Leaderboard:0.05517)

PS: In fact, the key point of entering the gold medal area is to use level information and do post-processing. Because doctors annotate layers by layers, they often mark the same label in successive layers. The information of the upper and lower layers of the target is very important for the prediction of the target label. And the post-processing I tried is just network architecture refinement and post-processing based on the logical relationship between `any` column and other `phenotype` columns, which didn't improve the score in my experiment. This is the gap between me and the gold medal area players.

## Requirements

- Python 3.6.7
- [Pytorch](https://pytorch.org/) 1.1.0
- [NVIDIA apex](https://github.com/NVIDIA/apex) 0.1 (for mixed precision training)
- [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) 0.5.1

The details of python packages can be seen in the `requirements.txt`.

## Performance (Single model)

| Backbone | Image size | LB |
----|----|----
| se\_resnext50\_32x4d | 512x512 | 0.070 - 0.072 |
| efficientnet-b3 | 512x512 | 0.071 - 0.074 |
| resnet34 | 512x512 | 0.073 - 0.076 |

## Windowing

For this challenge, windowing is important to focus on the matter, in this case the brain and the blood. There are good kernels explaining how windowing works.

- [See like a Radiologist with Systematic Windowing](https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing) by David Tang
- [RSNA IH Detection - EDA](https://www.kaggle.com/allunia/rsna-ih-detection-eda) by Allunia

We used three types of windows to focus and assigned them to each of the chennel to construct images on the fly for training.

| Channel | Matter | Window Center | Window Width |
----------|--------|---------------|---------------
| 0 | Brain | 40 | 80 |
| 1 | Blood/Subdural | 80 | 200 |
| 2 | Soft tissues | 40 | 380 |


## Preparation

Please put `./input` directory in the root level and unzip the file downloaded from kaggle there. All other directories such as `./cache`, `./data`, `./model` will be created if needed when `./bin/preprocess.sh` is run.


## Preprocessing

Please make sure you run the script from parent directory of `./bin`.

~~~
$ sh ./bin/preprocess.sh
~~~

- `preprocess.sh` does the following at once.
- dicom_to_dataframe.py reads dicom files and save its metadata into the dataframe. 
- create_dataset.py creates a dataset for training.
- make_folds.py makes folds for cross validation. 


## Training

~~~
$ sh ./bin/train001.sh
$ sh ./bin/train001-resnet34.sh 
$ sh ./bin/train001-efficientnet-b3.sh
~~~

- `train.001.sh uses` se\_resnext50\_32x4d from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) for training. It will be converged after 3 epoches.
- `train001-resnet34.sh` uses resnet34 from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) for training. It will be converged after 4 epoches.
- `train001-efficientnet-b3.sh` uses efficientnet-b3 from [efficientnet-pytorch](https://github.com/lukemelas/EfficientNet-PyTorch) for training. It will be converged after 3 epoches.

Actually, you should run all of the `train*` in the `./bin` fold.

## Predicting

~~~
$ sh ./bin/predict001-ep2.sh
$ sh ./bin/predict001-resnet34.sh
$ sh ./bin/predict001-ep2-efficientnet-b3.sh
~~~

 `predict001-ep2.sh` + `predict001-resnet34.sh` + `predict001-ep2-efficientnet-b3.sh` does the predictions and makes a submission file for scoring on Kaggle. Actually, you should run all of the `predict*` in the `./bin` fold.

## Ensembling

`testmerge.py` will firstly average over each well-trained model(average over 5 folds). Then it will use weight average three models(resnet34, resnext50, efficientnet-b3). The weights are [0.15,0.5,0.35] or [0.3,0.4,0.3].
