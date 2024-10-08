# PFNet: Large-scale Traffic Forecasting with Progressive Spatio-Temporal Fusion

This is a TensorFlow implementation of PFNet.

**Important Notice: Email Update to** chenwang99@buaa.edu.cn

## Requirements

* TensorFlow-gpu==2.5.0
* numpy==1.19.5
* networkx==2.6.3
* einops==0.3.2

## Data

The original data is under the folder `original_data/`, and the pre-processed dataset is under the folder `input/`.

## Run

### Train Details
***
Before training this model, make sure the three following settings are modified in `run.py`:

    MODE = 'train'              # train or test
    DATASET = 'LondonHW'        # LondonHW or ManchesterHW
    DURATION = 60

where `DURATION` is the constant of the forecasting horizon, such as 15, 30, and 60. 
After that, you can run `python run.py` to start training PFNet. The result will be generated in the `experiments` folder, including the `tensorboard-logs` folder, best model parameters, the result of prediction, ground truth, and the running log file.

### Test Details
***
Before testing the PFNet, you should modify the `MODE` variable as follows:

    MODE = 'test'              # train or test

Besides, go to the bottom of the `run.py` file, and comment out the following code:

    history = trainer.fit()
    
After uncommenting the test code `trainer.evaluate(is_pretrained=True, model_path='./experiments/LondonHW_15/2022-05-12-22-33-53/best_model')`, and changing the path `LondonHW_15/2022-05-12-22-33-53` of the test model, you can run `python run.py` to perform the test operation.

## Citation

Please cite the following paper, if you find the repository or the paper useful.

    @article{wang2023pfnet,
        title={PFNet: Large-Scale Traffic Forecasting With Progressive Spatio-Temporal Fusion},
        author={Wang, Chen and Zuo, Kaizhong and Zhang, Shaokun and Lei, Hanwen and Hu, Peng and Shen, Zhangyi and Wang, Rui and Zhao, Peize},
        journal={IEEE Transactions on Intelligent Transportation Systems},
        year={2023},
        publisher={IEEE}
    }
