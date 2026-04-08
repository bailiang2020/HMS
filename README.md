

This repository contains the code used for our multimodal fake news detection experiments based on Qwen3-VL. It includes data preprocessing, training, probing, ablations, and attention analysis for the `DGM4` and `Weibo` benchmarks.

## Environment Setup

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` includes CUDA-specific wheels. If your local environment differs from the one used in our experiments, you may need to adjust those packages accordingly.

## Model Preparation

Download the Qwen3-VL model and place it under:

```text
./models/Qwen/Qwen3-VL-2B-Instruct
```

All training and probing scripts assume this path by default.

## DGM4

### 1. Download and organize the raw data

Download and unpack the DGM4 dataset from Hugging Face (`rshaojimmy/DGM4`) and place the raw files under:

```text
./data/raw/DGM4
```

### 2. Preprocess the dataset

Run:

```bash
python data/prepare/DGM4.py --truncate_train --truncate_n 40000
```

The processed files will be written to:

```text
data/processed/DGM4/raw/
```

This command also creates `train_first40000.pkl`, which is used by the provided training scripts.

### 3. Training scripts

#### Baselines

Train with all multimodal data:

```bash
bash scripts/sft/DGM4/multi/train.sh
```

Train with text-only data:

```bash
bash scripts/sft/DGM4/text/train.sh
```

Train with image-only data:

```bash
bash scripts/sft/DGM4/img/train.sh
```

Train with the mixed schedule used in our experiments :

```bash
# 100 samples
bash scripts/sft/DGM4/mix/mix_0.0025.sh

# 400 samples
bash scripts/sft/DGM4/mix/mix_0.01.sh

# 2000 samples
bash scripts/sft/DGM4/mix/mix_0.05.sh
```

#### Our method

First obtain the reference head-role file:

```bash
bash scripts/sft/DGM4/probe/probe.sh
```

Then run our method:

```bash
# 100 samples
bash scripts/sft/DGM4/ours/mix_0.0025.sh

# 400 samples
bash scripts/sft/DGM4/ours/mix_0.01.sh

# 2000 samples
bash scripts/sft/DGM4/ours/mix_0.05.sh
```

#### Ablations

HMS only:

```text
scripts/sft/DGM4/ablation/hms/
```

UKR only:

```text
scripts/sft/DGM4/ablation/ukr/
```

Random heads:

1. Generate a low-overlap random head assignment with:

```text
scripts/utils/generate_random_head_roles.py
```

2. Update HEAD_ROLES_FILE and run the scripts under:

```text
scripts/sft/DGM4/ablation/random_heads/
```

### 4. Attention analysis

To obtain attention distributions, update the relevant parameters in:

```text
src/utils/attn_analysis.py
```

and run the analysis from there.

For attention-head masking experiments, example scripts are provided under:

```text
scripts/sft/DGM4/mask/
```

## Weibo

### 1. Download and organize the raw data

Download the Weibo dataset and place the raw files under:

```text
./data/raw/weibo
```

The original data source is:

- [MFAN repository](https://github.com/drivsaf/MFAN/tree/main)


### 2. Preprocess the dataset

1. Merge `nonrumor_images` and `rumor_images` into a single `images` directory.
2. Repeatedly run the preprocessing script if necessary to make sure all images are downloaded successfully:

```bash
python data/prepare/weibo.py
```

The processed files will be written to:

```text
data/processed/weibo/raw/
```

The processed annotation JSON files used by the provided scripts has been placed under:

```text
data/processed/weibo/annotation/
```

### 3. Training scripts

Train with all multimodal data:

```bash
bash scripts/sft/weibo/multi/train.sh
```

Train with the mixed setting:

```bash
bash scripts/sft/weibo/mix/mix.sh
```

Run our method:

```bash
# 100 samples
bash scripts/sft/weibo/ours/mix.sh
```

Before running our method, first obtain the reference head-role file:

```bash
bash scripts/sft/weibo/probe/probe.sh
```

## Notes

1. Some scripts are configured for evaluation by default. When you want to train from scratch, comment out the following arguments in the corresponding script if they are enabled:

```bash
--test_only
--img_only
--text_only
--lora_path ...
```

2. For multimodal evaluation, make sure no `*_only` flag is enabled.
3. For single-modality evaluation, enable the appropriate `--img_only` or `--text_only` flag.
4. For methods that depend on probing results, make sure `HEAD_ROLES_FILE` points to the correct JSON file. We also provide our head-role JSON files in the corresponding result directories.
5. The scripts are written with explicit local experiment settings such as GPU IDs and ports. Adjust them as needed for your machine.

## Directory Summary

```text
config/              DeepSpeed configurations
data/                Raw data, preprocessing, and processed files
results/             Saved outputs, analysis files, and reference JSONs
scripts/             Training, probing, masking, and utility scripts
src/                 Core training and model code
```
