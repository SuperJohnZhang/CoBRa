# CoBRa
The is the github page for ECCV submission ``Learning Counterfactual Thoughts for Bias-Robust Vision-Language Reasoning''

## Updates
1. Inital Update \[2024/03/06\]

## Dataset Download
| Updated on      | Questions and Annotations | Figures | Question Count | Figure Count |
| ----------- | :----: | :----: | :----: | :----: |
| Mar 06, 2024     | [CoBRa.csv](./CoBRa.csv) | [CoBRa.zip](https://drive.google.com/file/d/1xCRiI2kXdgmHSSfLlOPjBt4qMNDZLQqi/view?usp=sharing)         | 64604  | 14151 |


## Environment Setup
The following softwares are needed
1. Python >= 3.8
2. Cuda >= 11.3
3. Pytorch >= 12.0
4. numpy, pyyaml, openai, opencv-python, pillow, tqdm, transformers

## CoCT
### Train the TLM
The code is developped on top of implementations of ``[TLM](https://github.com/facebookresearch/XLM)''.
Please check the repo for MLM+TLM parameters
```
python train_TLM.py
```

### Prompt the LVLMs
check ./scripts