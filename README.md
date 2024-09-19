<div style="text-align: justify">
 
# SLIViT: a unified AI framework for analysing 3D biomedical imaging data

<br><br><img src="visuals/SLIViT.png" width="900px"/><br><br>

SLIViT is a data-efficient deep-learning framework that accurately measures disease-related risk factors in volumetric
biomedical imaging scans, such as magnetic resonance imaging (MRI), optical coherence tomography (OCT), ultrasound, and Computed Tomography (CT).

Below you may find step-by-step instructions on how to pre-train, fine-tune, and evaluate SLIViT. Please refer to 
<a href="https://www.researchsquare.com/article/rs-3044914/latest"><!--<a href="https://doi.org/10.1038/s41551-024-01257-9">-->our manuscript</a> 
for further details and feel free to <a href="mailto:orenavram@gmail.com,berkin1997@g.ucla.edu?subject=A%20SLIViT%20question"> reach
out</a> regarding any concerns/issues you are experiencing with SLIViT.

#TL;DR 
TODO

# Usage instructions
Running SLIViT is straightforward, as detailed below. But first, please ensure you have a cozy conda environment set up with all the necessary packages installed.

## Setting up the environment
First, go ahead and clone the repository. Once that's done, let’s set up your conda environment:
```bash
git clone https://github.com/cozygene/SLIViT
conda create --name slivit python=3.8
```
Next up, activate your conda environment and install the necessary packages:
```bash
conda activate slivit
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
cd SLIViT
pip install -r requirements.txt
```

[//]: # (Now, <a href="https://drive.google.com/drive/folders/1SmmVeGaM7DU2pmLRM-4HVVWb6E8iSwtP?usp=sharing">download</a> the pre-trained backbone and the fine-tuned SLIViT models. Once downloaded, please move the files into the checkpoints folder.)

Is your environment all ready to go? Awesome! You can either take SLIViT for a spin by training it yourself, or just grab our trained checkpoints right <a href="https://drive.google.com/drive/folders/1SmmVeGaM7DU2pmLRM-4HVVWb6E8iSwtP?usp=sharing">here</a>. Heads up—our model runs smoothly on PyTorch, and this repository is fully equipped to harness PyTorch’s GPU powers (no TensorFlow here 😉).

Curious about more advanced features? Just run the help command:
```bash
python pretrain.py -h
```

Happy computing! 🚀
## Pre-training SLIViT's backbone 
```bash
python pretrain.py --dataset <dataset type {kermany,chestmnist,custom}> --out_dir <out path> --meta_csv <path to a meta file csv> --label2d <(comma separated )label column name(s)>
```

### The 2D OCT (Kermany) dataset
<img src="visuals/OCT2D.png" width="600px"/><br>
<small>(Figure sourced from Kermany, et al., 2018 [1])</small>

Download the dataset <a href="https://data.mendeley.com/datasets/rscbjbr9sj/3">here</a>. After downloading the data, please update the paths in meta/kermany.csv to reflect the locations of your downloaded videos (you can use utils/get_kermany_csv.py for this purpose). To pre-train SLIViT on the Kermany dataset set --dataset to kermany and `label2d` to `Drusen,CNV,DME,Normal`.


### The 2D X-ray (ChestMNIST) dataset
<img src="visuals/Xray2D.png" width="450px"/><br>
<small>(Figure borrowed from Wang, et al., 2017 [2])</small>

The MNIST datasets will be automatically downloaded through the class API. To get started, simply set `--dataset` to `chestmnist`.


### A custom 2D dataset
You can also create your own dataloader for any other 2D dataset of your choice (you can start with our template in datasets/CustomDataset2D.py) and use it to pretrain SLIViT by setting `--dataset` to `custom`. 


## Fine-tuning SLIViT

```bash
python fine_tune.py --dataset <dataset type {oct,ultrasound,mri,ct,custom}> --fe_path <path to a pretrained convnext-t backbone> --out_dir <out path> --meta_csv <path to a meta file csv> --test_csv <path to an external test csv file> --label3d <label column name in csv>
```

TODO: should I mention evaluate here? 
If you prefer to evaluate slivit from the same dataset for training and testing, set the test proportion to be greater than zero in `--split_ratio` which by default is configured for an external test set (`0.85,0.15,0`). For example, set `--split_ratio` to `0.7,0.15,0.15` to set (respectively) the sample proportions for training, validation, and testing. 

### The OCT (Houston) dataset
The 3D OCT datasets utilized in this study are not publicly available due to institutional data-use policies and patient privacy concerns. However, we have provided the Dataset class used in our research for your reference. You can use this to fine-tune SLIViT on your own dataset.

### The ultrasound video (EchoNet) dataset
<img src="visuals/ultrasound.gif" width="750px"/><br>
The EchoNet Ultrasound videos are available <a href="https://stanfordaimi.azurewebsites.net/datasets/834e1cd1-92f7-4268-9daa-d359198b310a">here</a>. After downloading the data, please update the paths in `meta/echonet.csv` to reflect the locations of your downloaded videos (you can use `utils/get_echonet_csv.py` for this purpose). To fine-tune SLIViT on the EchoNet dataset set `--dataset` to `ultrasound`.

[//]: # (```bash)

[//]: # (python slivit_train.py --dataset3d ultrasound --meta_csv ./Dsets/ultrasound_meta.csv --bbpath ./checkpoints/kermany_convnext_tiny_feature_extractor.pth --nObb_feat 4 --nslc 32 --depth 5 --dim 256 --heads 32 )

[//]: # (```)


### The 3D MRI (United Kingdom Biobank) dataset
<img src="visuals/mri.gif" width="750px"/><br>
The UKBB MRI dataset is available <a href="https://www.ukbiobank.ac.uk">here</a>. Once you have downloaded the data, please create an appropriate meta file, or simply update the paths in `meta/ukbb.csv` to reflect the locations of your downloaded scans. To fine-tune SLIViT on the EchoNet dataset set `--dataset` to `mri`.

### The 3D CT (NoduleMNIST) dataset
<img src="visuals/ct.gif" width="450px"/><br>

The MNIST datasets will be automatically downloaded through the class API. To get started, simply set `--dataset` to `ct`. 

### Custom 3D
Ready to fine-tune SLIViT on your own dataset? Just set `--dataset` to `custom` (after you’ve tailored a Dataset class to fit your needs; you can start with our template in `datasets/CustomDataset3D.py`) and include the right meta file to get things rolling!

## Evaluating SLIViT
```bash
python fine_tune.py --dataset <dataset type {oct,ultrasound,mri,ct,custom}> --fe_path <path to a pretrained convnext-t backbone> --out_dir <out path> --meta_csv <path to a meta file csv> --label3d <label column name in csv>
```

If you prefer to use an external dataset for testing, set the test proportion to zero in the split ratio (for example, using a split of 0.85 for training and 0.15 for validation, as shown below). Additionally, provide a corresponding meta file to properly configure the dataset:
```bash
--split_ratio 0.85,0.15,0 --test_csv <path to an external test csv file>
```


### The OCT (Houston) dataset
The 3D OCT datasets used in this study aren’t available publicly due to institutional policies and patient privacy concerns. However, you can still use the fine-tuned model we’ve provided in the `checkpoints` directory to evaluate it on your own dataset.


### Nodule MNIST
```bash
python evaluate.py --dataset3d nodulemnist --checkpoint ./checkpoints/slivit_noduleMNIST --depth 5 --dim 64 --nslc 28 --mlp_dim 64 --heads 10
```
### UKBB

```bash
python evaluate.py --dataset3d ukbb --meta_csv ./datasets/ukbb_meta.csv --checkpoint ./checkpoints/slivit_ukbb --metric r2 --pathology PDFF --depth 5 --dim 256 --nslc 36 --heads 36
```
### Ultrasound

```bash
python evaluate.py --dataset3d ultrasound --meta_csv ./datasets/ultrasound_meta.csv --checkpoint ./checkpoints/slivit_ultrasound --pathology EF_b --depth 5 --dim 256 --nslc 32 --heads 32 --mlp_dim 256
```
### Custom 3D

```bash
python evaluate.py --dataset3d custom --meta_csv /path/to/generated/meta.csv --bbpath /path/to/finetuned/convnext_bb.pth --task TaskType --pathology Pathology
```
- ```--dataset3d``` is the dataset for 3D fine-tuning ( `nodulemnist`, `ukbb`, `ultrasound` ,`custom` ) 
- ```--meta_csv``` is the path to the created ```meta.csv``` file
- ```--pathology``` is pathology for 3D fine-tuning
- ```--nObb_feat``` is the number of classes the backbone was pre-trained on ( Kermany: `4` , ChestMNIST: `14` )
- ```--task``` is the 3D Fine-tuning task (classification or regression)
- ```--metric``` is the score metric for evaluation ( `roc-auc`, `pr-auc`, `r2` ) 
- ```--checkpoint``` is the path to fine-tuned slivit
- ```--nslc``` is the number of slices to use for 3D Fine-tuning
- ```--depth``` is the Vision Transformer depth
- ```--heads``` is the number of heads for multihead attention
- ```--dim``` specifies the dimension for encoding transformer input
- ```--mlp_dim``` denotes the multi-layer perceptron dimension for ViT

## Data Availability

The 2D OCT dataset was downloaded from https://www.kaggle.com/datasets/paultimothymooney/kermany2018. The 3D OCT B-scan
data are not publicly available due to institutional data use policy and concerns about patient privacy. However, they
are available from the authors upon reasonable request and with permission of the institutional review board. The
echocardiogram dataset is available at https://echonet.github.io/dynamic/index.html#dataset. The MRI dataset is
available at https://www.ukbiobank.ac.uk. The 3D CT, the 2D CT, and the 2D X-ray datasets are available
at https://medmnist.com/.

## Credits

We kindly request users to cite the corresponding paper upon using our code, checkpoints, or conclusions in any
capacity. Proper credit supports and recognizes the original creators' efforts.

```
@article{avram2023slivit,
  title={SLIViT: a general AI framework for clinical-feature diagnosis from limited 3D biomedical-imaging data},
  author={Avram, Oren and others},
  journal={InReview preprint https://www.researchsquare.com/article/rs-3044914/latest},
  year={2023}
}
```

## Acknowledgement

Parts of the code are taken from the <a href="https://github.com/lucidrains/vit-pytorch/tree/main"> vit-pytorch</a>
repository. The figures depicting Kermany and the X-ray dataset are sourced from [1] and [2] respectively.

## References

[1] Kermany DS, Goldbaum M, Cai W, Valentim CCS, Liang H, Baxter SL, McKeown A, Yang G, Wu X, Yan F, Dong J, Prasadha MK, Pei J, Ting MYL, Zhu J, Li C, Hewett S, Dong J, Ziyar I, Shi A, Zhang R, Zheng L, Hou R, Shi W, Fu X, Duan Y, Huu VAN, Wen C, Zhang ED, Zhang CL, Li O, Wang X, Singer MA, Sun X, Xu J, Tafreshi A, Lewis MA, Xia H, Zhang K. Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. Cell. 2018 Feb 22;172(5):1122-1131.e9. doi: 10.1016/j.cell.2018.02.010. PMID: 29474911.

[2] Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. (2017). ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). IEEE.
</div>
