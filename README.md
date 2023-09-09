# SLIViT: a general AI framework for medical-feature diagnosis from limited 3D biomedical-imaging data


<img src="SLIViT.png" width="900px"/>


SLIViT is a deep-learning framework that accurately measures disease-related risk factors in volumetric biomedical imaging, such as magnetic resonance imaging (MRI) scans, optical coherence tomography (OCT) scans, and ultrasound videos. 

Before running SLIViT.ipynb, please make sure to have an appropriate Python environment with the relevant packages (listed in requirements.txt) properly installed:
```bash
conda create --name slivit --file requirements.txt
```

A pre-trained backbone is provided (see backbone.pth.zip) although one may wish to train the model from scratch.

A code snippet with an example of loading a pre-trained SLIViT model and running inference on a given volume is provided at the end of the SLIViT.ipynb notebook.

#Acknowledgements
Part of codes are taken from the <a href="mailto:orenavram@gmail.com,berkin1997@g.ucla.edu?subject=A%20SLIViT%20question"> reach out</a>. Thanks to Caron et al. for the codes of DINO.

Feel free to <a href="https://github.com/lucidrains/vit-pytorch/tree/main"> vit-pytorch</a>.
