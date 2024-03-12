<div align="justify">


# SLIViT: a general AI framework for clinical-feature diagnosis from limited 3D biomedical-imaging data

<br><br><img src="SLIViT.png" width="900px"/><br><br>

SLIViT is a data-efficient deep-learning framework that accurately measures disease-related risk factors in volumetric biomedical imaging, such as magnetic resonance imaging (MRI) scans, optical coherence tomography (OCT) scans, ultrasound videos, and Computed Tomography (CT) scans.

SLIViT preprocesses volumes into 2D images and then combines two 2D-based deep-vision architectures: (1) a ConvNeXt backbone module that extracts a feature map for the slices (i.e., 2D frames of a volume), and (2) a ViT module that integrates this feature map into a single diagnosis prediction. SLIViT's feature-map extractor is initialized by pre-trained weights. These weights are obtained by pre-training a ConvNeXt first on ImageNet and then on a 2D biomedical-imaging dataset. The pre-trained network can be fine-tuned for different diagnostic tasks in volumetric-biomedical-imaging-data using a relatively small training set (few hundreds of samples).

A checkpoint of the pre-trained backbone is available <a href="TODO">here</a> although one may wish to train the model from scratch. A step-by-step procedure that includes pre-training, fine-tuning, and evaluating SLIViT is provided, along with a specific case study using 3D CT volumes (classifying nodule malignancy in the NoduleMNIST dataset). 

### Usage instructions
#### Setting up the environment
Before using SLIViT, please make sure to have an appropriate Python environment with the relevant packages (listed in requirements.txt) properly installed:
```bash
conda create --name slivit --file requirements.txt
```
##Pre-training

### 2D OCT pre-training ( Kermany Dataset )

Download the <a href="https://www.kaggle.com/datasets/paultimothymooney/kermany2018">Kermany dataset</a>.

```bash
python bb_train.py --meta_csv kermany_meta.csv --pathologies Drusen,CNV,DME,Normal --out_dir /output/dir/to/save_pretrained_model/ 
```



### 2D Custom Dataset Pre-Training
To commence pre-training with 2D Medical Scans, split the data into `train`, `validation`, `test` sets. Create three folders named `train`, `val`, `test`, and move the scans accordingly.

Generate the  ```meta.csv``` file for 2D pre-training data as illustrated below:

|F_Name | Path | Pathology-1  |  Pathology-2   |  Pathology-3  | Pathology-4  | 
|--- | --- | --- | --- |--- |--- |
| CNV-6116901-21.jpeg  | /data_dir/train/CNV/CNV-6116901-21.jpeg| 1.0 | 0.0   |  0.0  |   0.0| 
| DME-4616882-33.jpeg  |   /data_dir/test/DME/DME-4616882-33.jpeg| 0.0 | 1.0  |   0.0   |  0.0| 
| CNV-7907754-23.jpeg  |  /data_dir/test/CNV/CNV-7907754-23.jpeg | 1.0 | 0.0   |  0.0    | 0.0| 
| NORMAL-3757443-31.jpeg | /data_dir/val/NORMAL/NORMAL-3757443-31.jpeg | 0.0  |0.0  |   0.0   |  1.0
| NORMAL-6434323-2.jpeg |  /data_dir/train/NORMAL/NORMAL-6434323-2.jpeg  |0.0|  0.0   |  0.0 |    1.0|
| NORMAL-910422-8.jpeg | /data_dir/val/NORMAL/NORMAL-910422-8.jpeg | 0.0 | 0.0  |   0.0   |  1.0|
| DRUSEN-8086850-28.jpeg | /data_dir/train/DRUSEN/DRUSEN-8086850-28.jpeg | 0.0 | 0.0   |  1.0   |  0.0|

In the above table, `F_Name` denotes the file name for 2D medical scan files, `Path` indicates the directory to these files, and `Pathology-1`, `Pathology-2`, `Pathology-3`, and `Pathology-4` represent binary classes for the respective pathologies.

After creating ```meta.csv``` file, ConvNeXt backbone can be trained with following bash script:

```bash
python bb_train.py --meta_csv /path/to/meta.csv --pathologies Pathology-1,Pathology-2,Pathology-3,Pathology-4 --out_dir /output/dir/to/save_pretrained_model/ --b_size 16 --gpu_id 1 --n_cpu=32
```
- ```--meta_csv``` is the directory to the created ```meta.csv```.
- ```--pathologies``` is a comma-separated list of pathologies for pre-training.
- ```--out_dir```  is the output directory for saving the pre-trained backbone.
- ```--b_size``` denotes the batch size for training.
- ```--out_dir``` pecifies the GPU ID for training.
- ```--n_cpu``` indicates the number of CPUs for data loading.

#### Fine-tuning

#### Evaluating

Please refer to <a href="https://www.researchsquare.com/article/rs-3044914/latest">our manuscript</a> for further details and feel free to <a href="mailto:orenavram@gmail.com,berkin1997@g.ucla.edu?subject=A%20SLIViT%20question"> reach out</a> regarding any concerns/issues you are experiencing with SLIViT.

### Data Availability
The 2D OCT dataset was downloaded from https://www.kaggle.com/datasets/paultimothymooney/kermany2018. The 3D OCT B-scan data are not publicly available due to institutional data use policy and concerns about patient privacy. However, they are available from the authors upon reasonable request and with permission of the institutional review board. The echocardiogram dataset is available at https://echonet.github.io/dynamic/index.html#dataset. The MRI dataset is available at https://www.ukbiobank.ac.uk. The 3D CT, the 2D CT, and the 2D X-ray datasets are available at https://medmnist.com/.

### Credits
We kindly request users to cite the corresponding paper upon using our code, checkpoints, or conclusions in any capacity. Proper credit supports and recognizes the original creators' efforts.

```
@article{avram2023slivit,
  title={SLIViT: a general AI framework for clinical-feature diagnosis from limited 3D biomedical-imaging data},
  author={Avram, Oren and others},
  journal={InReview preprint https://www.researchsquare.com/article/rs-3044914/latest},
  year={2023}
}
```

### Acknowledgement
Parts of the code are taken from the <a href="https://github.com/lucidrains/vit-pytorch/tree/main"> vit-pytorch</a> repository.
</div>
