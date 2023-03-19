# SASA
syntax-aware self-attention amr-parser

## Installation

As a prerequisite, the following requirements should be satisfied. Please install by yourself. (The code works fine with `python` 3.7 and `torch` 1.8.0.):
* `python`: >= 3.6
* [`pytorch`](https://github.com/pytorch/pytorch): >= 1.6



`SASA` can be installed from source, you can download it and run it in command line or IDE (i.e. Pycharm):
```shell script
unzip SASA-AMR.zip SASA-AMR
cd /your_path/SASA-AMR/
pip install -r requirements.txt
pip install -e .
mkdir data/AMR/temp/
```



## Dataset
Experiments are mainly conducted on [AMR2.0](https://catalog.ldc.upenn.edu/LDC2017T10) and [AMR3.0](https://catalog.ldc.upenn.edu/LDC2020T02). Sample data has been provided in our code.

## Train
Modify the dataset path in the configuration file `configs/config.yaml` before training.

The `save_dir` in the file `train.py` is the directory under `runs/` where the checkpoint files are saved. Modify it if you need.

Run the following command
```shell script
python bin/train.py --config configs/config.yaml
```
and results are in `runs/`

## Evaluate
To evaluate trained model, you can use the following command:
```shell script
python bin/predict_amrs.py \
    --datasets <AMR-ROOT>/*.txt \
    --dep-file <AMR-ROOT>/*.json \
    --gold-path data/AMR/tmp/sample/gold.amr.txt \
    --pred-path data/AMR/tmp/sample/pred.amr.txt \
    --checkpoint runs/<checkpoint>.pt \
    --beam-size 5 \
    --batch-size 500 \
    --device cuda \
    --config configs/config.yaml \
    --penman-linearization --use-pointer-tokens
```

To reproduce our paper's results, you will also need to run the [BLINK](https://github.com/facebookresearch/BLINK) 
entity linking system on the prediction amr file. Please refer to [Spring](https://github.com/SapienzaNLP/spring) for further details. 

To have comparable Smatch scores you will also need to use the scripts available at https://github.com/mdtux89/amr-evaluation

## Pretrained Checkpoints

You can download the pretrained model in BaiduNetDisk, and put them in the output directory (`runs/`):
```
URL：https://pan.baidu.com/s/1XWVeKmnnY2kPJLXW-8nxRA 
Password：SASA 
```

## Acknowledgements
Our code is based on [Spring](https://github.com/SapienzaNLP/spring). Thanks for their high quality open codebase.
