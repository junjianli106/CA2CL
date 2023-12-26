## CA2CL: Cluster-Aware Adversarial Contrastive Learning for Pathological Image Analysis


This is a PyTorch implementation of the CA2CL


### Preparation

Download the NCT-CRC-HE-100K and CRC-VAL-HE-7K：[NCT dataset](https://zenodo.org/record/1214456).

### Unsupervised Training

To do unsupervised pre-training of a ResNet-18 model on NCT-CRC-HE-100K in an 2*A100(40G) gpu machine, run:
```
CUDA_VISIBLE_DEVICES=0,1 python pretrain.py 
```
the NCT pretrained checkpoint is available at [checkpoint](https://drive.google.com/file/d/14YcAzF3Foi5qF_fPxVV85aPOiCkuGdlr/view?usp=drive_link)

### Fine-tuning

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 1-gpu machine, run:
```
python ft_le.py --dataset_path '../data/Kather_Multi_Class/' \
                --model_name 'resnet18' \
                --model_path './save/NCT/pretrain/resnet18/CA2CL_NCT.tar' \
                --ft_epoch 40 \ 
                --lr 0.0003 \ 
                --seed 1 \ 
                --only_fine_turning \ 
                --labeled_train 0.01 \ 
                --gpu_index 3
```


Fine-tuning results on CRC-VAL-HE-7K  (seed=1):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">label of train<br/>1%</th>
<th valign="bottom">label of train<br/>10%</th>
<th valign="bottom">label of train<br/>50%</th>
<th valign="bottom">label of train<br/>100%</th>
<!-- TABLE BODY -->
<tr><td align="center">acc</td>
<td align="center">0.958</td>
<td align="center">0.967</td>
<td align="center">0.968</td>
<td align="center">0.970</td>

</tr>
<!-- TABLE BODY -->
<tr><td align="center">f1</td>
<td align="center">0.941</td>
<td align="center">0.954</td>
<td align="center">0.957</td>
<td align="center">0.960</td>
</tr>
<!-- TABLE BODY -->
<tr><td align="center">logs</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1hZ0l13WNMdmSG1mjzDE-s8QNiBEBJ6zl?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1hZ0l13WNMdmSG1mjzDE-s8QNiBEBJ6zl?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1hZ0l13WNMdmSG1mjzDE-s8QNiBEBJ6zl?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1hZ0l13WNMdmSG1mjzDE-s8QNiBEBJ6zl?usp=drive_link">download</a></td>
</tr>
</tbody></table>

### Linear Classification

With a pre-trained model, to train a supervised linear classifier on frozen features/weights in an 8-gpu machine, run:
```
python ft_le.py --dataset_path '../data/Kather_Multi_Class/' 
                --model_name 'resnet18'
                --model_path './save/NCT/pretrain/resnet18/CA2CL_NCT.tar' \
                --ft_epoch 100
                --lr 0.01
                --seed 1
                --only_linear_eval
                --labeled_train 0.01
                --gpu_index 3
```


Linear classification results on CRC-VAL-HE-7K (seed=1):
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">label of train<br/>1%</th>
<th valign="bottom">label of train<br/>10%</th>
<th valign="bottom">label of train<br/>50%</th>
<th valign="bottom">label of train<br/>100%</th>
<!-- TABLE BODY -->
<tr><td align="center">acc</td>
<td align="center">0.967</td>
<td align="center">0.969</td>
<td align="center">0.969</td>
<td align="center">0.968</td>

</tr>
<!-- TABLE BODY -->
<tr><td align="center">f1</td>
<td align="center">0.953</td>
<td align="center">0.959</td>
<td align="center">0.958</td>
<td align="center">0.956</td>
</tr>
<!-- TABLE BODY -->
<tr><td align="center">logs</td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BwM_F7ZDSXZfe0Ne7kWRR0KUdFHRXyrt?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BwM_F7ZDSXZfe0Ne7kWRR0KUdFHRXyrt?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BwM_F7ZDSXZfe0Ne7kWRR0KUdFHRXyrt?usp=drive_link">download</a></td>
<td align="center"><a href="https://drive.google.com/drive/folders/1BwM_F7ZDSXZfe0Ne7kWRR0KUdFHRXyrt?usp=drive_link">download</a></td>
</tr>
</tbody></table>


### Transferring to Object Detection
We follow the evaluation setting in MoCo when trasferring to object detection.
1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)..
2. You can download our pre-processed datasets from [GlaS-coco-format](https://drive.google.com/file/d/1tFmQqDwyfOMJhKO9jjyg7c0V0aJkXR4W/view?usp=drive_link) and [CRAG-coco-format](https://drive.google.com/file/d/1kGg-0f2eV3n36UBmiBzertxJcco5QdLv/view?usp=drive_link).

3. Put dataset under "benchmarks/detection/datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.
4. the converted backbone weights is available at [checkpoint](https://drive.google.com/file/d/1Xd8aMQZMDW1V4fe-ZjdqGnDqFlBLeeoD/view?usp=drive_link), 
put dataset under "benchmarks\detection\converted_weights" directory,
5. run `run_ft.sh`  runs of fine-tuning and evaluation on GlaS dataset.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.