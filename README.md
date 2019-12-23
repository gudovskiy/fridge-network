# Reasoning model for Fridge demo

This is the code used to train models for *smart fridge* paper and demo:

**[Smart Home Appliances: Chat with Your Fridge](https://arxiv.org/pdf/1912.09589.pdf)**

This project forks [MAC](https://github.com/stanfordnlp/mac-network) project and modifies it for *smart fridge* application. You can use this code to reproduce our models.

## Requirements
- Tensorflow (originally has been developed with 1.3 but should work for later versions as well).
- We have performed experiments on Nvidia P100 GPU.
- See [`requirements.txt`](requirements.txt) for the required python3 packages and run `pip3 install -r requirements.txt` to install them.

## Pre-processing
Before training the model, we first have to generate the FRIDGR dataset and extract features for the images:

### Dataset
To generate dataset, use [FRIDGR dataset repository](https://github.com/gudovskiy/fridge-dataset/):
```bash
mv FRIDGR_SPLIT_scenes.json ./FRIDGR_v0.1/scenes/
mv FRIDGR_SPLIT_questions.json ./FRIDGR_v0.1/data/
mv images/* ./FRIDGR_v0.1/images/SPLIT/
```

The final command moves the generated dataset into the proper `data` directory, where we will put all the data files we use during training.

### Feature extraction
Extract ResNet-101 features for the FRIDGR train, val, and test images with the following commands:
```bash
CUDA_VISIBLE_DEVICES=0, python3 extract_features.py --input_image_dir ./images/train --output_h5_file ./data/train.h5 --batch_size 100
CUDA_VISIBLE_DEVICES=1, python3 extract_features.py --input_image_dir ./images/val   --output_h5_file ./data/val.h5 --batch_size 100
CUDA_VISIBLE_DEVICES=2, python3 extract_features.py --input_image_dir ./images/test  --output_h5_file ./data/test.h5 --batch_size 100
```

## Training 
To train the model, run the following command:
```bash
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs0" --train --batchSize 64 --testedNum 10000 --epochs 25 --netLength 4 --gpus 0 @configs/args.txt
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs2" --train --batchSize 64 --testedNum 10000 --epochs 40 --netLength 6 --gpus 1 @configs/args2.txt
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs3" --train --batchSize 64 --testedNum 10000 --epochs 40 --netLength 6 --gpus 2 @configs/args3.txt
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs4" --train --batchSize 64 --testedNum 10000 --epochs 40 --netLength 6 --gpus 3 @configs/args4.txt
```

First, the program preprocesses the FRIDGR questions. It tokenizes them and maps them to integers to prepare them for the network. It then stores a JSON with that information about them as well as word-to-integer dictionaries in the `./FRIDGR_v0.1/data` directory.

Then, the program trains the model. Weights are saved by default to `./weights/{expName}` and statistics about the training are collected in `./results/{expName}`, where `expName` is the name we choose to give to the current experiment. 

### Notes
- The number of examples used for training and evaluation can be set by `--trainedNum` and `--testedNum` respectively.
- You can use the `-r` flag to restore and continue training a previously pre-trained model. 
- We recommend you to try out varying the number of MAC cells used in the network through the `--netLength` option to explore different lengths of reasoning processes.
- Good lengths for FRIDGR are in the range of 4-16 (using more cells tends to converge faster and achieves a bit higher accuracy, while lower number of cells usually results in more easily interpretable attention maps). 

### Model variants
We have explored several variants of our model. We provide a few examples in `configs/args2-4.txt`. For instance, you can run the first by: 
```bash
python3 main.py --expName "experiment1" --train --batchSize 64 --testedNum 10000 --epochs 40 --netLength 6 @configs/args2.txt
```
- [`args2`](config/args2.txt) uses a non-recurrent variant of the control unit that converges faster.
- [`args3`](config/args3.txt) incorporates self-attention into the write unit.
- [`args4`](config/args4.txt) adds control-based gating over the memory.

See [`config.py`](config.py) for further available options (Note that some of them are still in an experimental stage).

## Evaluation
To evaluate the trained model, and get predictions and attention maps, run the following: 
```bash
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs0" --finalTest --batchSize 64 --testedNum 10000 --netLength 4 --gpus 3 --getPreds --getAtt -r @configs/args.txt
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs2" --finalTest --batchSize 64 --testedNum 10000 --netLength 6 --gpus 4 --getPreds --getAtt -r @configs/args2.txt
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs3" --finalTest --batchSize 64 --testedNum 10000 --netLength 6 --gpus 5 --getPreds --getAtt -r @configs/args3.txt
python3 main.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs4" --finalTest --batchSize 64 --testedNum 10000 --netLength 6 --gpus 6 --getPreds --getAtt -r @configs/args4.txt
```
The command will restore the model we have trained, and evaluate it on the validation set. JSON files with predictions and the attention distributions resulted by running the model are saved by default to `./preds/{expName}`.

- In case you are interested in getting attention maps (`--getAtt`), and to avoid having large prediction files, we advise you to limit the number of examples evaluated to 5,000-20,000.

## Visualization
After we evaluate the model with the command above, we can visualize the attention maps generated by running:
```bash
python3 visualization.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs4" --tier val

python3 visualization.py --expName "fridgr_FeatCls_EmbRandom_CfgArgs4" --tier test --imageBasedir ./images --dataBasedir ./FRIDGR_v0.1

```
(Tier can be set to `train` or `test` as well). The script supports filtering of the visualized questions by various ways. See [`visualization.py`](visualization.py) for further details.
