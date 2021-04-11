# WMHpype
Nipype based implementation of a WMH segmentation pipeline

## Install
* `conda create -n WMHpype scikit-learn=0.17.1 python=3.5 #` The old sklearn version is necessary because the pretrained classifier are made with that version
* `pip install nipype nilearn`  
* `conda install -c vfonov pyezminc=1.2`

## Example usage
```
python2 ./scripts/wmhsp.py -e PT -c RF -i test.csv -m ./MNI152_T1_1mm_brain_mask.mnc -o $PWD -t $PWD -p ./Trained_Classifiers/ -d Y -n test.csv
```