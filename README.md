# WMHpypes
Nipype implementation of WMH segmentation pipelines.

## Interfaces

* ###  sysu_media
the winning method in MICCAI 2017 WMH segmentation challenge orginal work repository: ([wmh_ibbmTum](https://github.com/hongweilibran/wmh_ibbmTum))

## Installation

### As a python library (pip)
```
conda create -n wmhpypes -c conda-forge pip
conda activate wmhpypes
git clone https://github.com/0rC0/WMHpypes.git
cd WMHpypes
pip install -r requirements.txt
pip install .
```

### As a python library (anaconda)
```
git clone https://github.com/0rC0/WMHpypes.git
cd WMHpypes
conda env create -f conda_env_cpu.yml
conda activate wmhpypes
pip install .
```

### As a Docker container
``` 
git clone https://github.com/0rC0/WMHpypes.git
cd WMHpypes
# for the GPU implementation see also https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
docker build -f Dockerfile_gpu -t wmhpypes_gpu .
```

## Usage

### As a python library

See `Quickstart` Jupyter notebooks in the `example` directory

### As a Docker container
``` 
docker run -v $PWD:/data --gpus all wmhpypes_gpu:latest -f '/data/test/*' -w '/data/WMHpypes/models/*.h5' -o '/data'
```

# Please cite
If you use the package please cite the original author's [paper](https://arxiv.org/pdf/1802.05203.pdf):

```
Gaubert, M., Dell’Orco, A., Lange, C., Garnier-Crussard, A., Zimmermann, I., Dyrba, M., ... & Max, K. (2023). Performance evaluation of automated white matter hyperintensity segmentation algorithms in a multicenter cohort on cognitive impairment and dementia. Frontiers in psychiatry, 13, 1010273.

Li, Hongwei & Jiang, Gongfa & Wang, Ruixuan & Zhang, Jianguo & Wang, Zhaolei & Zheng, Wei-Shi & Menze, Bjoern. (2018). Fully Convolutional Network Ensembles for White Matter Hyperintensities Segmentation in MR Images. NeuroImage. 183. 10.1016/j.neuroimage.2018.07.005. 
```
