# SnowFormer: Scale-aware Transformer via Context Interaction for Single Image Desnowing
**<font size=5>Authors:</font>** **Sixiang Chen<span>&#8224;</span>, Tian Ye<span>&#8224;</span>, Yun Liu<span>&#8224;</span>, Erkang Chen\*, Jun Shi, Jingchun Zhou**

+ **<span>&#8224;</span>**  &ensp;represents equal contributions.
+ **\***  &ensp;represents corresponding author.

[SnowFormer: Scale-aware Transformer via Context Interaction for Single Image Desnowing](https://arxiv.org/abs/2208.09703)

> **Abstract:** *Single image desnowing is a common yet challenging task. The complex snow degradations and diverse degradation scales demand strong representation ability. In order for the desnowing network to see various snow degradations and model the context interaction of local details and global information, we propose a powerful architecture dubbed as SnowFormer. First, it performs Scale-aware Feature Aggregation in the encoder to capture rich snow information of various degradations. Second, in order to tackle with largescale degradation, it uses a novel Context Interaction Transformer Block in the decoder, which conducts context interaction of local details and global information from previous scale-aware feature aggregation in global context interaction. And the introduction of local context interaction improves recovery of scene details. Third, we devise a Heterogeneous Feature Projection Head which progressively fuse features from both the encoder and decoder and project the refined feature into the clean image. Extensive experiments demonstrate that the proposed SnowFormer achieves significant improvements over other SOTA methods. Compared with SOTA single image desnowing method HDCW-Net, it boost the PSNR metric by 9.2dB on the CSD testset. Moreover, it also achieves a 5.13dB increase in PSNR compared with general image restoration architecture NAFNet, which verifies the strong representation ability of our SnowFormer
for snow removal task.*

##
<table>
  <tr>
    <td><p align='center'>
    <img src="https://github.com/Ephemeral182/SnowFormer/blob/master/image/SnowFormer1_1.jpg#pic_center" width="80%" ></img></td>
  </tr>
  <tr>
    <td><p align="center"><b>Overall Framework of SnowFormer</b></p></td>
  </tr> 
</table>

<table>
  <tr>
    <td> <img src = "https://github.com/Ephemeral182/SnowFormer/blob/master/image/PSNR_param.png" width="450"> </td>
    <td> <img src = "https://github.com/Ephemeral182/SnowFormer/blob/master/image/PSNR_param_srrs.png" width="450"> </td>
  </tr>
  <tr>
    <td><p align="center"><b>PSNR v.s. Parameters on CSD Dataset</b></p></td>
    <td><p align="center"> <b>PSNR v.s. Parameters on SRRS Dataset</b></p></td>
  </tr>
</table>

<table>
  <tr>
    <td><p align='center'>
    <img src="https://github.com/Ephemeral182/SnowFormer/blob/master/image/result.png#pic_center" width="900" ></img></td>
  </tr>
  <tr>
    <td><p align="center"><b>Results on CSD, SRRS, Snow100K datasets</b></p></td>
  </tr> 
</table>

## Installation
Our SnowFormer is built in Pytorch1.12.0, we train and test it ion Ubuntu20.04 environment (Python3.8, Cuda11.6).

For installing, please follow these intructions.
```
conda create -n py38 python=3.8
conda activate py38
conda install pytorch=1.12 
pip install opencv-python tqdm tensorboardX ....
```
## Dataset
We sample the 2000 images from every desnowing dataset for the test stage, including CSD, SRRS and Snow100K. We provide the download link of datasets, and the password is **ephe**.

<table>
  <tr>
    <th align="left">Dataset</th>
    <th align="center">CSD</th>
    <th align="center">SRRS</th>
    <th align="center">Snow100K</th>
  </tr>
  <tr>
    <td align="left">Link</td>
    <td align="center"><a href="https://pan.baidu.com/s/133JxR-sCICZyIXAZSxS_GQ">Download</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1vTBmacVkwp6qtjofMC4-Qg">Download</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1TStTvd4GGF60HYTahvyxpA">Download</a></td>
  </tr>
 </table>

## Testing Stage
You can download pre-trained model of CSD from [Pre-trained model](https://pan.baidu.com/s/1zAUlrniCGgrONk0_8CBadw) &nbsp; **Password:ephe** and save it in model_path.

Run the following commands:
```python
python3  test.py --dataset_type CSD --dataset_CSD dataset_CSD --model_path model_path
```
The rusults are saved in ./out/dataset_type/

## TODO List
- [ ] Checkpoints of SRRS and Snow100K
- [ ] Train.py

## Citation 
```Bibtex
@article{chen2022snowformer,
  title={SnowFormer: Scale-aware Transformer via Context Interaction for Single Image Desnowing},
  author={Chen, Sixiang and Ye, Tian and Liu, Yun and Chen, Erkang and Shi, Jun and Zhou, Jingchun},
  journal={arXiv preprint arXiv:2208.09703},
  year={2022}
}
```
## Contact
If you have any questions, please contact the email 282542428@qq.com, ephemeral182@gmail.com or 201921114013@jmu.edu.cn.
