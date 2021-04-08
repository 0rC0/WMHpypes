# WMHpype
Nipype based implementation of a WMH segmentation pipeline

## Requirements
* python2
* scikit-learn <= 0.21 (`conda install scikit-learn=0.17`)
* numpy
* scipy  
* minc-toolkit-v2 (`conda install -c bic-mni minc-toolkit-v2`)
* pyezminc (`conda install -c vfonov pyezminc=1.2`)

## Example usage
```
python2 ./scripts/wmhsp.py -e PT -c RF -i test.csv -m ./MNI152_T1_1mm_brain_mask.mnc -o $PWD -t $PWD -p ./Trained_Classifiers/ -d Y -n test.csv
```