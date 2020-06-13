# DMGNN
This repository contains the implementation of the CVPR2020 paper named: Dynamic Multiscale Graph Neural Networks for 3D Skeleton-Based Human Motion Prediction. [Paper](http://openaccess.thecvf.com/content_CVPR_2020/html/Li_Dynamic_Multiscale_Graph_Neural_Networks_for_3D_Skeleton_Based_Human_CVPR_2020_paper.html)

![image](https://github.com/limaosen0/DMGNN/blob/master/img/pipeline.png)

Abstract: We propose novel dynamic multiscale graph neural networks (DMGNN) to predict 3D skeleton-based human motions. The core idea of DMGNN is to use a multiscale graph to comprehensively model the internal relations of a human body for motion feature learning. This multiscale graph is adaptive during training and dynamic across network layers. Based on this graph, we propose a multiscale graph computational unit (MGCU) to extract features at individual scales and fuse features across scales. The entire model is action-category-agnostic and follows an encoder-decoder framework. The encoder consists of a sequence of MGCUs to learn motion features. The decoder uses a proposed graph-based gate recurrent unit to generate future poses. Extensive experiments show that the proposed DMGNN outperforms state-of-the-art methods in both short and long-term predictions on the datasets of Human 3.6M and CMU Mocap. We further investigate the learned multiscale graphs for the interpretability.

# Module Requirement
* Python 3.6
* Pytorch 1.0
* pyyaml
* argparse
* numpy
* h5py

# Environments
After downloading the codes of DMGNN, please run the following commands for the environment preparation.

Run
```
cd torchlight, python setup.py install, cd ..
```

# Training and Testing

To train a model for a specific task, e.g. short-term prediction on CMU Mocap, first
```
cd cmu-short
```
and then, just run
```
python main.py prediction -c ../config/CMU/short/train.yaml
```
Some model hyper-parameters or training configurations could be change in the file of '../config/CMU/short/train.yaml'. During training, the model shows the validation results and finally outputs the lowest prediction error.

And, we can also test the model given the saved model file. First, we need to change the Line-2 in '../config/CMU/short/train.yaml' as the path of saved model. Then, run
```
python main.py prediction -c ../config/CMU/short/test.yaml
```

Additionally, you can also download the saved model for short-term prediction on Human3.6M (as an example): [Model Link](https://pan.baidu.com/s/1ybEjEqlu9yTG6g-qsroB4g) (Baidu Cloud). And the password is: knht. Then, put the whole folder in './h36m-short'. Just run
```
python main.py prediction -c ../config/H36M/short/test.yaml
```
you can get the test results.

# Acknowledgement
Thanks for the framework provided by 'yysijie/st-gcn', which is source code of the published work [ST-GCN](https://aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17135) in AAAI-2018. The github repo is [ST-GCN code](https://github.com/yysijie/st-gcn). We borrow the framework and interface from the code.

We also thank for the pre-processed data provided by the works of Res-sup. ([paper](http://openaccess.thecvf.com/content_cvpr_2017/html/Martinez_On_Human_Motion_CVPR_2017_paper.html),[code](https://github.com/una-dinosauria/human-motion-prediction)) and Convolutional Seq2seq model ([paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Li_Convolutional_Sequence_to_CVPR_2018_paper.html),[code](https://github.com/chaneyddtt/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics)).

# Citation
If you use this code, please cite our paper:
```
@InProceedings{Li_2020_CVPR,
author = {Li, Maosen and Chen, Siheng and Zhao, Yangheng and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
title = {Dynamic Multiscale Graph Neural Networks for 3D Skeleton Based Human Motion Prediction},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
