import os
import numpy as np
import pandas as pd
import wandb

from argparse import ArgumentParser
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, \
    Trainer, TrainingArguments
from pprint import pprint


ACCURACY = PRECISION = RECALL = F1 = None
TRAINING_ARGS = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "evaluation_strategy": "no",
    "save_strategy": "no",
    "logging_steps": 20,
    "output_dir": "./results/"
}


def get_args():
    parser = ArgumentParser(description="Train transformer on a GLUE task")
    parser.add_argument("--task", type=str, help="Any GLUE binary classification task")
    parser.add_argument("--model", type=str, help="Path to transformer model in Huggingface Hub")
    parser.add_argument("--seed", type=int, help="Random seed used for training")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--log_file", type=str, default="results.csv", help="Filename for storing results")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to enable wandb logging")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                        help="Entity parameter for initializing wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, 
                        help="Project parameter for initializing wandb logging")

    return parser.parse_args()


def compute_metrics(eval_pred, average="binary"):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_dict = {}
    for metric in [ACCURACY, PRECISION, RECALL, F1]:
        metric_dict.update(metric.compute(predictions=predictions, references=labels))
    return metric_dict


# Log metrics into pandas DataFrame
def log_metrics(metrics, filename, task, model, seed):
    metrics_df = pd.DataFrame({
        "model": [model],
        "task": [task],
        "seed": [seed],
        "train_accuracy": [metrics["train_accuracy"]],
        "test_accuracy": [metrics["test_accuracy"]],
        "train_f1": [metrics["train_f1"]],
        "test_f1": [metrics["test_f1"]]
    })

    if os.path.isfile(filename):
        metrics_df = pd.concat([pd.read_csv(filename), metrics_df])
    
    metrics_df.to_csv(filename, index=False)


def main():
    args = get_args()

    # Load model and tokenizer from Huggingface Hub
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding=True, truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).cuda()

    # Load and preprocess dataset
    dataset = load_dataset("glue", args.task)
    if args.task == "mrpc":
        dataset = dataset.map(lambda x: {"sentence": x["sentence1"] + " <s> " + x["sentence2"]},
                              batched=False, load_from_cache_file=False)

    tokenized = dataset.map(lambda examples: tokenizer(examples["sentence"]), batched=True, 
                            load_from_cache_file=False).remove_columns(["idx", "sentence"])

    # Load metrics
    global ACCURACY, PRECISION, RECALL, F1
    ACCURACY = load_metric("accuracy")
    PRECISION = load_metric("precision")
    RECALL = load_metric("recall")
    F1 = load_metric("f1")
    
    # Fix training settings
    settings = TRAINING_ARGS.copy()
    settings["num_train_epochs"] = args.epochs
    settings["seed"] = args.seed

    # Create wandb run if needed
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=[args.model, args.task],
        )
    
    # Start training
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**settings), 
        train_dataset=tokenized["train"],
        data_collator=collator,
        compute_metrics=compute_metrics
    )
    trainer.train()

    # Collect and log metrics
    final_metrics = trainer.evaluate(tokenized["train"], metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(tokenized["validation"], metric_key_prefix="test"))
    log_metrics(final_metrics, args.log_file, args.task, args.model, args.seed)
    if args.use_wandb:
        wandb.log(final_metrics)
        run.finish()
    else:
        print("Final metrics:")
        pprint(final_metrics)


if __name__ == "__main__":
    main()