# SAP-SAM
 respository for ACM MM 2024 paprt "Fine-grained Semantic Alignment with Transferred Person-SAM for Text-based Person Retrieval"

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

![Overview of SAP-SAM](fig/region_alignment_framework.pdf "Overview of SAP-SAM") 

# 🚀 QuickStart
For a quick start, you can proceed directly to the training phase in Step 5 after installing the necessary dependencies and organizing the data as instructed in Step 4. The processed data is provided in the link of Step 4, and the folder also includes our processed text.

# 🛠️ Start Step by Step
## Step 01
In this step, we obtain the textual descriptions corresponding to the human-parsing dataset.  
First, download the ATR dataset [here](https://github.com/lemondan/HumanParsing-Dataset) and unzip it. Then run the following program and wait for the run to complete. 
```bash
cd s01_atr_transfer

CUDA_VISIBLE_DEVICES=0 python generate_atr_description.py --data_path /path/to/atr_dataset
```
Finally divide the dataset into training, testing and validation sets.

Or you can directly use the phrases we've generated (which are stored in that folder).

(Now, you can directly use the API of MLLM to complete these steps.)

## Step 02

Name the three datasets obtained in the previous step as `atr_item_descriptions_train.json`, `atr_item_descriptions_test.json`, and `atr_item_descriptions_dev.json` and organize them as follows.

```
|-- path/to/atr_dataset/
|   |-- <JPEGImages>/
|       |-- 997_1.jpg
|       |-- 997_2.jpg
|       |-- 997_3.jpg
|       |-- ...
|   |-- <SegmentationClassAug>/
|       |-- 997_1.png
|       |-- 997_2.png
|       |-- 997_3.png
|       |-- ...
|   |-- atr_item_descriptions_train.json
|   |-- atr_item_descriptions_test.json
|   |-- atr_item_descriptions_dev.json
|   |-- atr_label.txt
```

Then download the [BERT model](https://huggingface.co/bert-base-uncased/tree/main) and the [SAM model](https://huggingface.co/facebook/sam-vit-base/tree/main) from huggingface respectively.

Then please run the following code and wait for training.

```bash
cd s02_personsam_training

CUDA_VISIBLE_DEVICES=0 python tuning_sam_on_atr_description.py --data_path /path/to/atr_dataset --sam_path sam_path --language_model_path bert_path --batch_size 6 --num_epochs 20
```

You can download our trained model [here](https://pan.baidu.com/s/1SUeR_7YozaWqNGZSpI74Lw?pwd=s6mp)(Code: s6mp). If you have better computing resources or optimization techniques, you can make this model even better.

## Step 03
First prepare each of the three datasets according to the following steps.
### 3.1 Prepare TBPR Datasets
#### 3.1.1 CUHK-PEDES
Download the dataset from [here](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) and organize the dataset as follows:
```bash
|-- dataset_path/
|   |-- <CUHK-PEDES>/
|       |-- imgs
            |-- cam_a
            |-- cam_b
            |-- ...
|       |-- reid_raw.json
```

#### 3.1.2 ICFG-PEDES
Download the dataset from [here](https://github.com/zifyloo/SSAN) and organize the dataset as follows:
```bash
|-- dataset_path/
|   |-- <ICFG-PEDES>/
|       |-- imgs
            |-- test
            |-- train 
|       |-- ICFG_PEDES.json
```

#### 3.1.3 RSTPReid
Download the dataset from [here](https://github.com/NjtechCVLab/RSTPReid-Dataset) and organize the dataset as follows:
```bash
|-- dataset_path/
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- data_captions.json
```

### 3.2 Split Sentence
Then please download the ChatGLM2-6B model [here](https://huggingface.co/THUDM/chatglm2-6b/tree/main). Then run the following code separately.

```bash
cd s03_chatglm_split

CUDA_VISIBLE_DEVICES=0 python generate_attributes.py --data_path /path/to/dataset_path --dataset "CUHK-PEDES" --chatglm_path /path/to/chatglm --step 100000

CUDA_VISIBLE_DEVICES=0 python generate_attributes.py --data_path /path/to/dataset_path --dataset "ICFG-PEDES" --chatglm_path /path/to/chatglm --step 100000

CUDA_VISIBLE_DEVICES=0 python generate_attributes.py --data_path /path/to/dataset_path --dataset "RSTPReid" --chatglm_path /path/to/chatglm --step 100000
```

Finally, put these results into the corresponding datasets respectively.

## Step 04
Run the following code

```bash
cd s04_generate_relationshaips

CUDA_VISIBLE_DEVICES=0 python generate_mask.py --data_path /path/to/dataset_path --dataset "CUHK-PEDES" --bert_path /path/to/bert_path --sam_path /path/to/sam_path --trained_sam /path/to/trained_sam_in_s02 --step 1000000
```

Similar to Step 03, you need to run this step on all three datasets, achieved by changing the `--dataset` flag.

In addition, you can also directly download the segmented images we have generated [here](https://pan.baidu.com/s/1Jw2AyXAkl0q4ui4yw_CEqg?pwd=4c7y)(Code: 4c7y), and find the processed json format files in the current folder.

Finally, the organizational dataset is shown below.

```bash
|-- dataset_path/
|   |-- <CUHK-PEDES>/
|       |-- imgs
            |-- cam_a
            |-- cam_b
            |-- ...
|       |-- segs
            |-- 0__CUHK01_0363004_0.png
            |-- 0__CUHK01_0363004_1.png
            |-- ...
|       |-- reid_raw.json
|       |-- CUHK-PEDES_data_final_68126.json
|       |-- CUHK-PEDES_score_dict_68125.json
|   |-- <ICFG-PEDES>/
|       |-- imgs
            |-- test
            |-- train 
|       |-- segs
            |-- 0__test_0627_0627_010_05_0303afternoon_1591_0_0.png
            |-- 0__test_0627_0627_010_05_0303afternoon_1591_0_1.png
            |-- ...
|       |-- ICFG_PEDES.json
|       |-- ICFG-PEDES_data_final_34674.json
|       |-- ICFG-PEDES_score_dict_34673.json
|   |-- <RSTPReid>/
|       |-- imgs
|       |-- segs
            |-- 0_0000_c14_0031_0.png
            |-- 0_0000_c14_0031_1.png
            |-- ...
|       |-- data_captions.json
|       |-- RSTPReid_data_final_37010.json
|       |-- RSTPReid_score_dict_37009.json

```

## Step 05
Finally, run the following code to train the model

```bash
cd SAP-SAM-TBPR

CUDA_VISIBLE_DEVICES=4 python train.py --name irra --batch_size 64 --MLM --loss_names 'sdm+mlm_part+matching' --dataset_name 'CUHK-PEDES' --root_dir './' --num_epoch 60 --part_seg --img_aug --part_mask_prob 0.35
```

# 🙏 Acknowledgement
We sincerely appreciate for the contributions of [SSAN](https://github.com/zifyloo/SSAN),  [LGUR](https://github.com/ZhiyinShao-H/LGUR), [IRRA](https://github.com/anosorae/IRRA) and [Segment Anything](https://github.com/facebookresearch/segment-anything).

# Citing

```
@inproceedings{wang2024fine,
  title={Fine-grained Semantic Alignment with Transferred Person-SAM for Text-based Person Retrieval},
  author={Wang, Yihao and Yang, Meng and Cao, Rui},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={5432--5441},
  year={2024}
}
```
