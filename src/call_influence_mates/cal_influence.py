from datasets import Dataset, Features, Sequence, Value
from transformers.trainer_pt_utils import IterableDatasetShard
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Union
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import lightning as L
import torch
import math
import time
import sys
import os
import jsonlines
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BertTokenizerFast
import argparse

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.utils import (
    get_default_supported_precision,
    chunked_cross_entropy,
    num_parameters,
)

# Hyperparameters and global settings
fsdp = False
learning_rate = 1e-3
batch_size = 16
micro_batch_size = 16
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
stable_iters = 400000
lr_decay_iters = 400000
warmup_iters = lr_decay_iters * 0.04
min_lr = 1e-4

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}
logger = None


def setup_fabric(devices: int = 1) -> L.Fabric:
    """Setup the Fabric environment."""
    precision = get_default_supported_precision(training=True)
    strategy = FSDPStrategy(
        auto_wrap_policy={Block},
        activation_checkpointing_policy={Block},
        state_dict_type="full",
        limit_all_gathers=True,
        cpu_offload=False,
    ) if fsdp else "auto"
    
    fabric = L.Fabric(
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
        loggers=logger,
    )
    fabric.print(hparams)
    return fabric


def train_collate_fn(batch):
    input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]

    # 对序列进行填充，确保长度一致
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # 使用 0 进行填充
    
    if padded_input_ids.size(1) > 1024: 
        padded_input_ids = padded_input_ids[:, :1024]
    elif padded_input_ids.size(1) < 1024:
        padding = torch.zeros((padded_input_ids.size(0), 1024 - padded_input_ids.size(1)), dtype=torch.long)
        padded_input_ids = torch.cat((padded_input_ids, padding), dim=1)

    return padded_input_ids.to("cuda")


def val_collate_fn(batch):
    input_ids = [torch.tensor(sample["input_ids"]) for sample in batch]
    labels = [torch.tensor(sample["labels"]) for sample in batch]

    x = pad_sequence(input_ids, batch_first=True, padding_value=0)
    y = pad_sequence(labels, batch_first=True, padding_value=-1)

    max_seq_length = 1024
    if max_seq_length:
        x = x[:, :max_seq_length]
        y = y[:, :max_seq_length]
    return x.to("cuda"), y.to("cuda")


def process_jsonl_files(base_directory, model_name, out_dir, resume, slices):
    """Traverse folders and process each JSONL file."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.to("cuda")  # Use GPU

    fabric = setup_fabric()  # Set up Fabric environment
    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # Traverse each subdirectory and process JSONL files
    data_files = []
    for subdir, _, files in os.walk(base_directory):
        for file in files:
            data_files.append(subdir+'/'+file)

    data_to_save = []
    for file in data_files:
        if file.endswith('.jsonl'):
            file_path = os.path.join(subdir, file)
            output_path = out_dir
            with jsonlines.open(file_path, 'r') as reader:
                for line in reader:
                    data_to_save.append(line)
    process_single_file(file_path, output_path, model, tokenizer, fabric, optimizer, resume, data_to_save, slices)


def process_single_file(file_path, output_path, model, tokenizer, fabric, optimizer, resume, data_to_save, slices):
    """Process a single JSONL file, calculate loss, and save results."""
    data_to_save = datasets.Dataset.from_list(data_to_save)
    data_to_save = datasets.DatasetDict({'train':data_to_save})
    def preprocess_function(examples):
        return tokenizer(examples['content'], truncation = False, padding='max_length',max_length=1024)
    data_to_save = data_to_save["train"].map(preprocess_function, batched=True)
    data_train = IterableDatasetShard(
        data_to_save,
        batch_size=micro_batch_size,
        num_processes=fabric.world_size,
        process_index=fabric.global_rank,
    )

    train_dataloader = DataLoader(data_train, batch_size=1, collate_fn=train_collate_fn)
    val_dataloader = DataLoader(
        torch.load("/fs-computility/llm/shared/baitianyi/train/iclr/MATES/data/lambada_openai/train-1024.pt"),
        batch_size=64,
        collate_fn=val_collate_fn,
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader,
        val_dataloader,
    )
    val_dataloaders = [val_dataloader]

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 1250* slices,
        "step_count": 0,
    }
    train_iter = iter(train_dataloader)
    for i in range(1250*slices-1):
        next(train_iter)
    data = []
    for _ in tqdm(range(1250*slices,min(1250*(slices+1),len(data_to_save)))):
        fabric.load(resume,state,strict=False)
        input_ids = next(train_iter)
        scores = train(fabric, state, input_ids, val_dataloaders)
        data.append(scores)
    features = Features(
        {
            "scores": Sequence(Value("float32")),
        }
    )

    data_to_save = data_to_save.select(range(1250*slices,min(1250*(slices+1),len(data_to_save))))
    data_to_save = data_to_save.add_column("influence", data)
    data_to_save = data_to_save.remove_columns('content')
    data_to_save = data_to_save.remove_columns('input_ids')
    data_to_save = data_to_save.remove_columns('attention_mask')
    data_to_save = data_to_save.to_pandas()
    data_to_save.to_json(output_path, orient="records", lines=True)

def train(fabric, state, input_ids, val_dataloaders):
    model = state["model"]
    optimizer = state["optimizer"]
    # lr = get_wsd_lr(state["iter_num"]) if decay_lr else learning_rate
    lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    logits = model(input_ids)
    logits = logits.logits
    loss = chunked_cross_entropy(
        logits[:, :-1, :].contiguous(),
        input_ids[:, 1:].contiguous(),
        chunk_size=0,
    )
    fabric.backward(loss)
    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
    optimizer.step()
    optimizer.zero_grad()
    return evaluate(fabric, model, val_dataloaders)

@torch.no_grad()
def evaluate(fabric, model, val_dataloaders):
    model.eval()
    losses = []
    for val_dataloader in val_dataloaders:
        loss = torch.tensor(0.0, device=fabric.device)
        cnt = 0
        for input_ids, labels in val_dataloader:
            logits = model(input_ids)
            logits = logits.logits
            loss += chunked_cross_entropy(
                logits[:, :-1, :],
                labels[:, 1:],
                chunk_size=0,
            )
            cnt += 1
        loss = loss / cnt
        losses.append(loss.item())
    model.train()
    return losses

def get_wsd_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it < stable_iters:
        return learning_rate
    return learning_rate * math.pow(0.5, (it - stable_iters) / 400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some jsonl files.")
    parser.add_argument("--base_directory", type=str, required=True, help="Input folder path")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output folder path")
    parser.add_argument("--resume", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--slices", type=int, required=True, help="Checkpoint file")

    args = parser.parse_args()
    process_jsonl_files(args.base_directory, args.model_name, args.out_dir, args.resume, args.slices)
