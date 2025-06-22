# T5 Fine-tune

This repository provides scripts to fine-tune a T5 model for text summarization using the Hugging Face Transformers library.

## Step-by-Step Instructions

### 1. Clone the Repository
```sh
!git clone https://github.com/longdnk/T5-finetune.git
```

### 2. Set CUDA Devices (if using GPU)
```sh
!export CUDA_VISIBLE_DEVICES=0,1
```

### 3. Install Requirements
```sh
!pip install --upgrade -r /kaggle/working/T5-finetune/requirements.txt
```

### 4. Run Training Script
You can run the training script with sample arguments as follows:

```sh
python train.py \
  --model=t5-small \
  --batch_size=4 \
  --num_procs=4 \
  --epochs=20 \
  --max_length=512 \
  --dataset=gopalkalpande/bbc-news-summary \
  --logging_steps=200 \
  --eval_steps=200 \
  --save_steps=200 \
  --save_total_limit=2 \
  --warmup_steps=500 \
  torch_empty_cache_steps=200 \
  --bf16=True
```

You can adjust the arguments as needed for your use case.

---

- For more details, see the code in `train.py`.
- Make sure your environment has access to GPUs for best performance.

## Features
- [x] Logging with tensorboard.
- [x] Multi GPU training.
- [ ] Config params.
- [ ] Run with MLFlow.
- [ ] Online show tensorboard