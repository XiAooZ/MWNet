# MWNet
Official code of **Tracking spatial temporal details in ultrasound long video via wavelet analysis and memory bank**.

Contributors: [Chenxiao Zhang](https://github.com/XiAooZ), Runshi Zhang

# Network Architecture
## MWNet:
![MWNet](imgs/NetworkArchitecture.png)

## High-frequency-aware feature fusion (HFF):

<img src="https://github.com/XiAooZ/MWNet/blob/main/imgs/HFF.png" alt="HFF" width=70%>

## Long-short term memory bank:

<img src="https://github.com/XiAooZ/MWNet/blob/main/imgs/LongShortMemoryBank.png" alt="LSMemory" width="50%">

# Train and Test
MWNet are tested on 4 datasets: Thyroid Nodule, VTUS, CAMUS and TG3K datasets. After downloading the datasets listed below, training of MWNet can be performed by following the instructions in `train.py`, while evaluation of the trained model can be conducted by referring to the guidelines in `eval.py`.

Pretrained checkpoints can be downloaded in [Baidu Netdisk](https://pan.baidu.com/s/19cHe1BdZgKZ31eM7-Wxy4A?pwd=a43i) or [Beihang Netdisk]()

# Datasets
The **VTUS dataset** is available for download in the [Vivim project](https://github.com/scott-yjyang/Vivim)

The **CAMUS dataset** is available for download in [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)

The **TG3K dataset** is availble for download in [TRFE-Net project](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)

# Code Details
Module **Memory-based wavelet convolution (MWConv)** and **Long short-term memory bank** can be found in [mmsegmentation/mmseg/models/backbones/MWConv.py](https://github.com/XiAooZ/MWNet/blob/main/mmsegmentation/mmseg/models/backbones/MWConv.py). 

Module **high frequency feature fusion (HFF)** can be found in [mmsegmentation/mmseg/models/decode_heads/hff_head.py](https://github.com/XiAooZ/MWNet/blob/main/mmsegmentation/mmseg/models/decode_heads/hff_head.py)

# Requirements
We trained our models depending on Pytorch 2.1.1 and Python 3.8.

# Contact
If you have any questions, feel free to contact [zhang_cx@buaa.edu.cn](mailto:zhang_cx@buaa.edu.cn)

# Reference and Acknowledgments
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

[mmpretrain](https://github.com/open-mmlab/mmpretrain)

[WTConv](https://github.com/BGU-CS-VIL/WTConv.git) 

[DAWN](https://github.com/mxbastidasr/DAWN_WACV2020).



