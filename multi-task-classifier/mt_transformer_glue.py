# classic
import os
import datetime
import gc
import math
import operator
from argparse import ArgumentParser
from functools import wraps
from pathlib import Path
import random
from random import sample
import contextlib
import sys
from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
import logging

# additional
import pandas as pd
import scipy
from scipy import stats

# torch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.nn import BCELoss, MSELoss, CrossEntropyLoss

# transformers
from transformers import BertTokenizer, BertModel


# tracking
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb

# utils
from task import Task, define_dataset_config, define_tasks_config

# init logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)


def get_args():
    '''Load args from config.yaml'''
    config = instantiate(OmegaConf.load("config.yaml"))
    return config


args = get_args()
logging.info(f"Config: {args}")

device = torch.device(
    args.device) if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"Device: {device}")


def save_config(cfg: DictConfig, output_dir: str):
    config_file = os.path.join(output_dir, "config.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_file, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


@contextlib.contextmanager
def stream_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


# region MODEL


class SSCModule(nn.Module):  # Single sentence classification
    def __init__(self, hidden_size, dropout_prob=0.1, output_classes=2):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, output_classes))

    def forward(self, x):
        return self.output_layer(x)


class PTSModule(nn.Module):  # Pairwise text similarity
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.output_layer(x).view(-1)


class PTCModule(nn.Module):  # Pariwise text classification
    def __init__(self, hidden_size, k_steps, output_classes, dropout_prob=0.1, stochastic_prediction_dropout_prob=0.1):
        super().__init__()
        self.stochastic_prediction_dropout = stochastic_prediction_dropout_prob
        self.k_steps = k_steps
        self.hidden_size = hidden_size
        self.output_classes = output_classes

        self.GRU = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size, batch_first=True)

        self.W1 = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
        )

        self.W2 = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size))

        self.W3 = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(4 * hidden_size, output_classes),
        )

    def forward(self, premises: torch.Tensor, hypotheses: torch.Tensor):
        batch_size = premises.size(0)

        output_probabilities = torch.zeros(
            batch_size, self.output_classes).to(device)

        flatten_hypotheses = hypotheses.reshape(-1, self.hidden_size)
        flatten_premise = premises.reshape(-1, self.hidden_size)

        alfas = F.softmax(
            self.W1(flatten_hypotheses).view(batch_size, - 1), -1)
        s_state = (alfas.unsqueeze(1) @ hypotheses)  # (Bs,1,hidden)

        layer_output = self.W2(flatten_premise).view(
            batch_size, -1, self.hidden_size)
        layer_output_transpose = torch.transpose(layer_output, 1, 2)

        actual_k = 0
        for k in range(self.k_steps):
            betas = F.softmax(s_state @ layer_output_transpose, -1)
            x_input = betas @ premises
            _, s_state = self.GRU(x_input, s_state.transpose(0, 1))
            s_state = s_state.transpose(0, 1).to(device)
            concatenated_features = torch.cat([s_state, x_input, (s_state - x_input).abs(), x_input * s_state],
                                              -1).to(device)
            if torch.rand(()) > self.stochastic_prediction_dropout or (not self.training):
                output_probabilities += self.W3(
                    concatenated_features).squeeze()
                actual_k += 1

        return output_probabilities / actual_k


class PRModule(nn.Module):  # Pairwise ranking module
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.output_layer(x)).view(x.size(0))


class MultiTask_BERT(nn.Module):
    def __init__(self, bert_pretrained_model="bert-base-uncased"):
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model)
        self.bert = BertModel.from_pretrained(bert_pretrained_model)
        self.hidden_size = self.bert.config.hidden_size
        k_steps = self.bert.config.num_hidden_layers

        # Single-Sentence Classification modules
        self.CoLa = SSCModule(self.hidden_size, dropout_prob=0.05)
        self.SST_2 = SSCModule(self.hidden_size)

        # Pairwise Text Similarity module
        self.STS_B = PTSModule(self.hidden_size)

        # Pairwise Text Classification
        self.MNLI = PTCModule(self.hidden_size, k_steps, output_classes=Task.MNLIm.num_classes(), dropout_prob=0.3,
                              stochastic_prediction_dropout_prob=0.3)
        self.RTE = PTCModule(self.hidden_size, k_steps,
                             output_classes=Task.RTE.num_classes())
        self.WNLI = PTCModule(self.hidden_size, k_steps,
                              output_classes=Task.WNLI.num_classes())
        self.QQP = PTCModule(self.hidden_size, k_steps,
                             output_classes=Task.QQP.num_classes())
        self.MRPC = PTCModule(self.hidden_size, k_steps,
                              output_classes=Task.MRPC.num_classes())
        self.SNLI = SSCModule(
            self.hidden_size, output_classes=Task.SNLI.num_classes())
        self.SciTail = SSCModule(
            self.hidden_size, output_classes=Task.SciTail.num_classes())

        # Pairwise Ranking
        self.QNLI = PRModule(self.hidden_size)

    def forward(self, x, task: Task):
        tokenized_input = self.tokenizer(
            x, padding=True, truncation=True, return_tensors='pt')
        for name, data in tokenized_input.items():
            tokenized_input[name] = tokenized_input[name].to(device)

        bert_output = self.bert(**tokenized_input).last_hidden_state
        cls_embedding = bert_output[:, 0, :]
        if task == Task.CoLA:
            return self.CoLa(cls_embedding)
        elif task == Task.SST_2:
            return self.SST_2(cls_embedding)
        elif task == Task.STS_B:
            return self.STS_B(cls_embedding)
        elif task == Task.MNLIm or task == Task.MNLImm or task == task.AX:
            premises, hypotheses = self.preprocess_PTC_input(
                bert_output, tokenized_input)
            return self.MNLI(premises, hypotheses)
        elif task == Task.RTE:
            premises, hypotheses = self.preprocess_PTC_input(
                bert_output, tokenized_input)
            return self.RTE(premises, hypotheses)
        elif task == Task.WNLI:
            premises, hypotheses = self.preprocess_PTC_input(
                bert_output, tokenized_input)
            return self.WNLI(premises, hypotheses)
        elif task == Task.QQP:
            premises, hypotheses = self.preprocess_PTC_input(
                bert_output, tokenized_input)
            return self.QQP(premises, hypotheses)
        elif task == Task.MRPC:
            premises, hypotheses = self.preprocess_PTC_input(
                bert_output, tokenized_input)
            return self.MRPC(premises, hypotheses)
        elif task == Task.SNLI:
            return self.SNLI(cls_embedding)
        elif task == Task.SciTail:
            return self.SciTail(cls_embedding)
        elif task == Task.QNLI:
            return self.QNLI(cls_embedding)

    @staticmethod
    def loss_for_task(t: Task):
        losses = {
            Task.CoLA: "CrossEntropyLoss",
            Task.SST_2: "CrossEntropyLoss",
            Task.STS_B: "MSELoss",
            Task.MNLIm: "CrossEntropyLoss",
            Task.WNLI: "CrossEntropyLoss",
            Task.QQP: "CrossEntropyLoss",
            Task.MRPC: "CrossEntropyLoss",
            Task.QNLI: "BCELoss",
            Task.SNLI: "CrossEntropyLoss",
            Task.SciTail: "CrossEntropyLoss",
            Task.RTE: "CrossEntropyLoss"
        }

        return losses[t]

    def preprocess_PTC_input(self, bert_output, tokenized_input):
        mask_premises = tokenized_input.attention_mask * \
            torch.logical_not(tokenized_input.token_type_ids)
        premises_mask = mask_premises.unsqueeze(
            2).repeat(1, 1, self.hidden_size)
        longest_premise = torch.max(
            torch.sum(torch.logical_not(tokenized_input.token_type_ids), -1))
        # Not include CLS embedding
        premises = (bert_output * premises_mask)[:, 1:longest_premise, :]

        mask_hypotheses = tokenized_input.attention_mask * tokenized_input.token_type_ids
        hypotheses_mask = mask_hypotheses.unsqueeze(
            2).repeat(1, 1, self.hidden_size)
        longest_hypothesis = torch.max(
            torch.sum(tokenized_input.token_type_ids, -1))
        hypotheses = (
            bert_output * hypotheses_mask).flip([1])[:, :longest_hypothesis, :].flip([1])

        return premises, hypotheses


# endregion

# region UTILS FUNCS
def split_n(chunk_length, sequence):
    if type(sequence) == dict:
        key_splits = {}
        for key, subseq in sequence.items():
            key_splits[key] = split_n(chunk_length, subseq)

        splits_count = len(next(iter(key_splits.values())))
        splits = []

        # Now "transpose" from dict of chunked lists to list of dicts (each with a chunk)
        for i in range(splits_count):
            s = {}
            for key, subseq in key_splits.items():
                s[key] = subseq[i]

            splits.append(s)

        return splits

    else:
        splits = []

        splits_count = math.ceil(len(sequence) / chunk_length)
        for i in range(splits_count):
            splits.append(
                sequence[i * chunk_length:min(len(sequence), (i + 1) * chunk_length)])

        return splits


def retry_with_batchsize_halving(train_task=None):
    def inner(train_fn):
        @wraps(train_fn)
        def wrapper(*args, **kwargs):
            retry = True
            task = train_task or kwargs.get("task")
            input_data = kwargs["input_data"]
            batch_size = len(input_data)
            label = kwargs.get("label", [0] * batch_size)
            optimizer = kwargs["optimizer"]

            while retry and batch_size > 0:
                microbatches = split_n(batch_size, input_data)
                microlabels = split_n(batch_size, label)

                for microbatch, microlabel in zip(microbatches, microlabels):
                    try:
                        new_kwargs = dict(
                            kwargs, input_data=microbatch, label=microlabel)
                        train_fn(*args, **new_kwargs)
                    except RuntimeError as e:
                        logging.error(
                            f"{e} Error in current task {task} with batch size {batch_size}. Retrying...")
                        batch_size //= 2
                        optimizer.zero_grad(set_to_none=True)
                        break
                    finally:
                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    retry = False

            if retry:
                logging.info(f"Skipping {task} batch... (size: {batch_size})")

        return wrapper

    return inner


@retry_with_batchsize_halving()
def train_minibatch(input_data, task, label, model, task_criterion, **kwargs):
    output = model(input_data, task)
    loss = task_criterion(output, label)
    loss.backward()
    del output
# endregion


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    seed_everything(args.seed)
    NUM_EPOCHS = args.epochs

    datasets_config = define_dataset_config()
    tasks_config = define_tasks_config(datasets_config)

    task_actions = []
    for task in iter(Task):
        if task in args.tasks:
            train_loader = tasks_config[task]["train_loader"]
            task_actions.extend([task] * len(train_loader))

    # Create wandb run if needed
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=[args.model, args.task],
            name=f"{args.model}_task_{args.task}_epochs_{args.epochs}_seed_{args.seed}"
        )
    model = MultiTask_BERT()
    model.to(device)
    optimizer = optim.Adamax(model.parameters(), lr=5e-5)
    initial_epoch = 1
    training_start = datetime.datetime.now().isoformat()

    if args.from_checkpoint:
        logging.info("Loading from checkpoint")
        checkpoint = torch.load(args.from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initial_epoch = checkpoint['epoch'] + 1
        training_start = checkpoint["training_start"]

    else:
        logging.info("Starting training from scratch")

    logging.info(
        f"------------------ training-start:  {training_start} --------------------------)")

    losses = {'BCELoss': BCELoss(), 'CrossEntropyLoss': CrossEntropyLoss(),
              'MSELoss': MSELoss()}
    for name, loss in losses.items():
        losses[name].to(device)

    for epoch in range(initial_epoch, NUM_EPOCHS + 1):
        with stream_redirect_tqdm() as orig_stdout:
            epoch_bar = tqdm(
                sample(task_actions, len(task_actions)), file=orig_stdout)
            model.train()

            train_loaders = {task: iter(
                tasks_config[task]["train_loader"]) for task in set(task_actions)}

            for task_action in epoch_bar:
                train_loader = tasks_config[task_action]["train_loader"]
                epoch_bar.set_description(
                    f"current task: {task_action.name} in epoch:{epoch}")

                try:
                    data = next(train_loaders[task_action])
                except StopIteration:
                    logging.info(f"Iterator ended early on task {task_action}")
                    continue

                optimizer.zero_grad(set_to_none=True)

                data_columns = [
                    col for col in tasks_config[task_action]["columns"] if col != "label"]
                input_data = list(zip(*(data[col] for col in data_columns)))

                label = data["label"]
                if label.dtype == torch.float64:
                    label = label.to(torch.float32)
                if task_action == Task.QNLI:
                    label = label.to(torch.float32)

                task_criterion = losses[MultiTask_BERT.loss_for_task(
                    task_action)]

                if len(data_columns) == 1:
                    input_data = list(map(operator.itemgetter(0), input_data))

                label = label.to(device)
                train_minibatch(input_data=input_data, task=task_action, label=label, model=model,
                                task_criterion=task_criterion, optimizer=optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_start': training_start
            }, os.path.join(hydra_cfg['runtime']['output_dir'], f'multitask_bert_base_seed_{args.seed}_epoch_{epoch}.tar'))

            model.eval()
            val_results = {}
            with torch.no_grad():
                task_bar = tqdm([task for task in Task if task in args.tasks],
                                file=orig_stdout)
                for task in task_bar:
                    task_bar.set_description(task.name)
                    val_loader = tasks_config[task]["val_loader"]

                    task_predicted_labels = torch.empty(0, device=device)
                    task_labels = torch.empty(0, device=device)
                    for val_data in val_loader:
                        data_columns = [
                            col for col in tasks_config[task]["columns"] if col != "label"]
                        input_data = list(
                            zip(*(val_data[col] for col in data_columns)))
                        label = val_data["label"].to(device)

                        if len(data_columns) == 1:
                            input_data = list(
                                map(operator.itemgetter(0), input_data))

                        model_output = model(input_data, task)

                        if task == Task.QNLI:
                            predicted_label = torch.round(model_output)
                        elif task.num_classes() > 1:
                            predicted_label = torch.argmax(model_output, -1)
                        else:
                            predicted_label = model_output

                        task_predicted_labels = torch.hstack(
                            (task_predicted_labels, predicted_label.view(-1)))
                        task_labels = torch.hstack((task_labels, label))

                    metrics = datasets_config[task].metrics
                    for metric in metrics:
                        metric_result = metric(
                            task_labels.cpu(), task_predicted_labels.cpu())
                        if type(metric_result) == tuple or type(metric_result) == stats.spearmanr:
                            metric_result = metric_result[0]
                        val_results[task.name, metric.__name__] = metric_result

                        logging.info(
                            f"val_results[{task.name}, {metric.__name__}] = {val_results[task.name, metric.__name__]}")
            data_frame = pd.DataFrame(
                data=val_results,
                index=[epoch])
            if args.use_wandb:
                wandb.log(val_results)
                run.finish()
            data_frame.to_csv(
                os.path.join(hydra_cfg['runtime']['output_dir'], f"{args.log_file}"), mode='a', index_label='Epoch')
            save_config(OmegaConf.to_container(args, resolve=True),
                        args.TRAINING_ARGS.output_dir)


if __name__ == '__main__':
    main()
