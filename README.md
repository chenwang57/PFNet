# PFNet: Large-scale Traffic Forecasting with Progressive Spatio-Temporal Fusion
===
This is a TensorFlow implementation of PFNet.

## Requirements
***
* TensorFlow-gpu==2.5.0
* numpy==1.19.5
* networkx==2.6.3
* einops==0.3.2

## Data
***
The original dataset is under the folder `original_data`, and the pre-processed dataset is under the folder `input`.

## Run
***
### Train Details
Before training this model, make sure the two following settings are modified in `run.py`:
    DATASET = 'LondonHW'        # LondonHW or ManchesterHW
    DURATION = 60
where DURATION is the constant of the forecasting horizon, such as 15, 30, 60. 

### Test Details


## Citation
***
Please cite the following paper, if you find the repository or the paper useful.

    ......
