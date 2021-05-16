# INLCUDE - Isolated Indian Sign Language Recognition

This repository contains code for training models on [INCLUDE](https://zenodo.org/record/4010759) dataset

# Dependencies

Install the dependencies through the following command

```bash
>> pip install -r requirements.txt
```



## Steps
- Download the INCLUDE dataset
- Run `generate_keypoints.py` to save keypoints from Mediapipe Hands and Blazepose for train, validation and test videos. 
```bash
>> python generate_keypoints.py --include_dir <path to downloaded dataset> --save_dir <path to save dir> --dataset <include/include50>
```
- Run `runner.py` to train a machine learning model on the dataset
```bash
>> python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <location to saved keypoints>
```
- Use the `--use_pretrained` flag to either perform only inference using pretrained model or resume training with the pretrained model. 
```bash
>> python runner.py --dataset <include/include50> --use_augs --model transformer --data_dir <location to saved keypoints> --use_pretrained <evaluate/resume_training>
```
- To get predictions for videos from a pretrained model, run the following command.
```bash
>> python evaluate.py --data_dir <dir with videos>
```

## Citation

```
@inproceedings{10.1145/3394171.3413528,
author = {Sridhar, Advaith and Ganesan, Rohith Gandhi and Kumar, Pratyush and Khapra, Mitesh},
title = {INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition},
year = {2020},
isbn = {9781450379885},
publisher = {Association for Computing Machinery},
doi = {10.1145/3394171.3413528},
numpages = {10},
series = {MM '20}
}
```

