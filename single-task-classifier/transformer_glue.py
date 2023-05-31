# classic
import os
import numpy as np
import pandas as pd
import logging
import random

import torch

# tracking
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

# transformers
from datasets import load_dataset, load_metric
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, EarlyStoppingCallback)


ACCURACY = PRECISION = RECALL = F1 = None

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def get_args():
    '''Load args from config.yaml'''
    config = instantiate(OmegaConf.load("config.yaml"))
    return config


def save_config(cfg: DictConfig, output_dir: str):
    config_file = os.path.join(output_dir, "config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def compute_metrics(eval_pred, average="binary"):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_dict = {}
    for metric in [ACCURACY, PRECISION, RECALL, F1]:
        metric_dict.update(metric.compute(
            predictions=predictions, references=labels))
    return metric_dict


def log_metrics(metrics, filename, task, model, seed):
    '''
    Log metrics into pandas DataFrame
    '''
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


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    args = cfg
    
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logging.info(OmegaConf.to_yaml(args))
    seed_everything(args.TRAINING_ARGS.seed)
    logging.info(
        f"Current hydra folder:{hydra_cfg['runtime']['output_dir']}")

    device = torch.device(
        args.device) if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"Device: {device}")
    # Load model and tokenizer from Huggingface Hub
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding=True, truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model).to(device)

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
    settings = args.TRAINING_ARGS.copy()
    settings['output_dir'] = hydra_cfg['runtime']['output_dir']

    # Create wandb run if needed
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=[args.model, args.task],
            name=f"{args.model}_task_{args.task}_epochs_{args.TRAINING_ARGS.num_train_epochs}_seed_{args.TRAINING_ARGS.seed}"
        )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

    # Start training
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**settings),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks
    )
    trainer.train()

    # Collect and log metrics
    final_metrics = trainer.evaluate(
        tokenized["train"], metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(
        tokenized["validation"], metric_key_prefix="test"))
    log_metrics(final_metrics, args.log_file,
                args.task, args.model, args.TRAINING_ARGS.seed)
    if args.use_wandb:
        wandb.log(final_metrics)
        run.finish()
    else:
        logging.info("Final metrics:")
        logging.info(final_metrics)
    save_config(OmegaConf.to_container(
        args, resolve=True), args.TRAINING_ARGS.output_dir)


if __name__ == "__main__":
    main()
