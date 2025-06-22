import torch
import pprint
import evaluate
import numpy as np
import argparse

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="T5 Fine-tuning Script")
    parser.add_argument(
        "--model", type=str, default="t5-small", help="Model name or path"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size per device"
    )
    parser.add_argument(
        "--num_procs", type=int, default=4, help="Number of processes for tokenization"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=512, help="Max sequence length"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gopalkalpande/bbc-news-summary",
        help="Dataset name or path",
    )
    parser.add_argument("--logging_steps", type=int, default=200, help="Logging steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps")
    parser.add_argument(
        "--save_steps", type=int, default=200, help="Save checkpoint steps"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Max number of checkpoints to keep",
    )
    parser.add_argument("--fp16", type=bool, default=False, action="store_true", help="Use fp16 mixed precision")
    parser.add_argument("--bf16", type=bool, default=False, action="store_true", help="Use bf16 mixed precision")
    return parser.parse_args()


args = parse_args()

MODEL = args.model
BATCH_SIZE = args.batch_size
NUM_PROCS = args.num_procs
EPOCHS = args.epochs
MAX_LENGTH = args.max_length
DATASET = args.dataset
LOGGING_STEPS = args.logging_steps
EVAL_STEPS = args.eval_steps
SAVE_STEPS = args.save_steps
SAVE_TOTAL_LIMIT = args.save_total_limit
FP16 = args.fp16
BF16 = args.bf16
OUT_DIR = "results-" + MODEL


def load_dataset():
    print("===Start load dataset===")
    dataset = load_dataset(DATASET, split="train")
    full_dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    dataset_train = full_dataset["train"]
    dataset_valid = full_dataset["test"]
    print(dataset_train)
    print(dataset_valid)
    print("===Load dataset done===")
    return dataset_train, dataset_valid


print("===Init tokenizer and process data===")
tokenizer = T5Tokenizer.from_pretrained(MODEL)


# Function to convert text data into model inputs and targets
def preprocess_function(examples):
    inputs = [f"summarize: {article}" for article in examples["Articles"]]
    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
    )

    # Set up the tokenizer for targets
    targets = [summary for summary in examples["Summaries"]]
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=MAX_LENGTH, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


dataset_train, dataset_valid = load_dataset()
# Apply the function to the whole dataset
tokenized_train = dataset_train.map(
    preprocess_function, batched=True, num_proc=NUM_PROCS
)
tokenized_valid = dataset_valid.map(
    preprocess_function, batched=True, num_proc=NUM_PROCS
)
print("===Done init tokenizer and process data===")

print("===Init create model===")
model = T5ForConditionalGeneration.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"===Load model on device: {device}===")
model.to(device)
print("===Model done with===")
print(model.eval())

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")
print("===Done create model step===")

print("===Add compute metrics===")
rouge = evaluate.load("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions[0], eval_pred.label_ids

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        rouge_types=["rouge1", "rouge2", "rougeL"],
    )

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


print("===Done add compute metrics===")

print("===Training model step===")


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


print("===Add training arguments===")
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=OUT_DIR,
    eval_strategy="steps",
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    report_to="tensorboard",
    learning_rate=1e-4,
    dataloader_num_workers=4,
    fp16=FP16,
    bf16=BF16,
)
print("===Done add training arguments===")

print("===Create trainer===")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
)
print("===Done create trainer===")

print("===Start training model===")
history = trainer.train()
print("===End training model===")