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

# Environments

# Training and Testing

# Acknowledgement

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
