from datasets import Dataset, DatasetDict
import pandas as pd
import os, json
import evaluate
import numpy as np

import random
import torch
import re
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import argparse
from pathlib import Path
import torch.nn as nn

#from petrel_client.client import Client
from collections import defaultdict
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report
from sklearn.metrics import top_k_accuracy_score

from transformers import BertTokenizerFast, RobertaTokenizerFast
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import BertForSequenceClassification, RobertaForSequenceClassification
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import RandomSampler

from preprocess import *
from sampler import ClassAwareSampler
from typing import  Optional
import warnings
warnings.filterwarnings("ignore")

# DEFAULT_CONF_PATH = '~/petreloss.conf'
#client = Client(DEFAULT_CONF_PATH)

# USER_NAME = ''
# USER_PASSWORD = ''

# hypers that don't effect much, but can still change for better result#
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 15
WEIGHT_DECAY = 0.01
DEFAULT_THRESHOLD = 0.8

# may not change otherwise may cause gpu memory overflow
PER_DEVICE_BATCH_SIZE = 16

########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained_model_name_or_path', type=str, default='/mnt/petrelfs/baitianyi/dup/train_bert/train/DownloadModel/bert-base-uncased')
parser.add_argument('--output_dir', type=str, default='/mnt/petrelfs/baitianyi/dup/train_bert/train/Bert-0.1')
parser.add_argument('--ptm_type', type=str, default='bert')
parser.add_argument('--data_folder', type=str, default='/mnt/petrelfs/baitianyi/dup/train_bert/version2')
parser.add_argument('--use_test', type=str, default='True')
parser.add_argument('--only_test', type=str, default='False')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--weight_type', type=str, default='average')
parser.add_argument('--sample_type', type=str, default='')
parser.add_argument('--level_type', type=str, default='')



def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")

def set_random_seed(seed):
    if seed is not None:
        assert seed > 0
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True # False
        torch.backends.cuda.matmul.allow_tf32 = False # if set it to True will be much faster but not accurate


class CustomTrainer(Trainer):
    def __init__(self, sampler_type, cates, cls_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.sampler_type = sampler_type
        self.cates = cates
        self.cls_weight = cls_weight

    #def compute_loss(self, model, inputs, return_outputs=False):
    #    labels = inputs.get("label")
        # forward pass
    #    outputs = model(**inputs)
    #    logits = outputs.get("logits")
        #compute custom loss (suppose one has 3 labels with different weights)
    #    loss_fct = nn.CrossEntropyLoss(weight=self.weight.to(labels.device))
    #    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
    #    return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # 使用权重 (如果存在)
        if self.cls_weight is not None and labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.cls_weight.to(labels.device))
        else:
            # print(f"Inside compute_loss, using default weight.")
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        elif self.sampler_type:
                return ClassAwareSampler(self.train_dataset, self.cates, self.cls_weight, self.args.seed)
        else:
            return RandomSampler(self.train_dataset)
    

def main():
    args = parser.parse_args() 
    # print args
    #Print arguments
    for k,v in sorted(vars(args).items()):
        print(k,'=',v, '\n')

    data_folder = args.data_folder
    pretrained_model_name_or_path = args.pretrained_model_name_or_path
    output_dir = Path(args.output_dir) / now()
    ptm_type = args.ptm_type
    use_test = (args.use_test == 'True')
    only_test = (args.only_test == 'True')
    dropout = args.dropout
    seed = args.seed
    weight_type = args.weight_type
    sample_type = args.sample_type
    level_type = args.level_type

    if only_test:
        output_dir = args.output_dir
    else:
        output_dir = Path(args.output_dir) / now()
        output_dir.mkdir(parents=True, exist_ok=True)
    print('output_dir:',output_dir)

    set_random_seed(seed)
    
    cates, train_data, valid_data, weight = read_data(data_folder,weight_type, seed)

    # proxy_on(USER_NAME,USER_PASSWORD)
    
    if only_test:
        # train_dataset = Dataset.from_list(train_data[:1])
        train_dataset = Dataset.from_list([]) 
    else:
        train_dataset = Dataset.from_list(train_data) 

    if only_test:
        valid_dataset = Dataset.from_list(valid_data[:1])
    else:
        valid_dataset = Dataset.from_list(valid_data)
    
    if only_test:
        assert use_test

    if use_test:
        test_dataset = Dataset.from_list(valid_data)

    if use_test:
        my_datasets = DatasetDict({
            'train':train_dataset,
            'valid':valid_dataset,
            'test':test_dataset,
        })
    else:
        my_datasets = DatasetDict({
            'train':train_dataset,
            'valid':valid_dataset,
        })

    if ptm_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    elif ptm_type == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    # def preprocess_function(examples):
    #     # Tokenize the inputs (text)
    #     tokenized_inputs = tokenizer(examples['content'], truncation=True, max_length=512)
    #     # Convert 'selected_topic' to integer labels
    #     label_map = {label: i for i, label in enumerate(set(examples['selected_topic']))}
    #     tokenized_inputs['labels'] = [label_map[label] for label in examples['selected_topic']]
    #     return tokenized_inputs

    # def preprocess_function(examples):
    #     # Tokenize the inputs (text)
    #     tokenized_inputs = tokenizer(examples['content'], truncation=True, max_length=512)        
    #     # Sort the unique topics to ensure consistent enumeration
    #     unique_topics = sorted(set(examples['selected_topic']))        
    #     # Create a stable label map based on the sorted topics
    #     label_map = {label: i for i, label in enumerate(unique_topics)}        
    #     # Convert 'selected_topic' to integer labels
    #     tokenized_inputs['labels'] = [label_map[label] for label in examples['selected_topic']]
    #     return tokenized_inputs
    
        # 在数据预处理函数中添加打印语句
    def preprocess_function(examples):
        # Tokenize the inputs (text)
        tokenized_inputs = tokenizer(examples['content'], truncation=True, max_length=512)        
        # Sort the unique topics to ensure consistent enumeration
        unique_topics = sorted(set(examples['selected_topic']))        
        # Create a stable label map based on the sorted topics
        label_map = {label: i for i, label in enumerate(unique_topics)}        
        # Convert 'selected_topic' to integer labels
        tokenized_inputs['labels'] = [label_map[label] for label in examples['selected_topic']]
        # 打印标签和主题的映射关系
        print("Label to Topic Mapping:", label_map)
        return tokenized_inputs


    tokenized_datasets = my_datasets.map(preprocess_function, batched=True)
    # print(f"Sample labels from train_dataset: {tokenized_datasets['train'][0]['labels']}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if ptm_type == 'bert':
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=len(cates), hidden_dropout_prob=dropout)
    elif ptm_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=len(cates), hidden_dropout_prob=dropout)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=len(cates), hidden_dropout_prob=dropout)

    def compute_metrics(eval_pred, metric_name='macro'):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
    
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average=metric_name)
        precision = precision_score(labels, predictions, average=metric_name)
        recall = recall_score(labels, predictions, average=metric_name)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    training_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        # seed = seed,
        disable_tqdm=True,
        # save_steps=9500,
        # save_total_limit=10
    )
    # Ensure all model parameters are contiguous
    def make_model_parameters_contiguous(model):
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                print(f"Making {name} contiguous")
                param.data = param.contiguous()

    make_model_parameters_contiguous(model)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'] if len(tokenized_datasets['train']) > 0 else None,  # 处理空的train数据集
        eval_dataset=tokenized_datasets['valid'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        cls_weight=weight,
        cates=cates,
        sampler_type=sample_type
    )

    print('==========================================train==========================================')
    if only_test:
        trainer._load_from_checkpoint(os.path.join(output_dir, 'best_ckpt'))
    # else:
    #     trainer.train()
    #     make_model_parameters_contiguous(model)
    #     trainer.save_model(os.path.join(output_dir, 'best_ckpt'))
    else:
        if trainer.train_dataset is not None:  # 仅当train_dataset不为空时才进行训练
            trainer.train()
            make_model_parameters_contiguous(model)
            trainer.save_model(os.path.join(output_dir, 'best_ckpt'))

    # res = trainer.evaluate()
    # print(res)

    if use_test:
        print('==========================================test==========================================')
        res = trainer.predict(tokenized_datasets['test'])

        lcrk = int(os.environ.get('LOCAL_RANK', '0'))
        test_result = defaultdict(list)

        if lcrk == 0:
            # print(res, flush=True)
            preds = []
            for data, pred in zip(tokenized_datasets['test'], res[0]):
                logits = pred.tolist()
                pred_label = np.argmax(logits, axis=-1)
                preds.append({'pred':str(pred_label), 'logits':logits, 'labels':data['labels'], 'untokenized':tokenizer.decode(data['input_ids']), 'content':data['content']})

                pred_ = torch.tensor(pred)
                probs = torch.softmax(pred_, dim=-1)
                pos_prob = probs[1]
                pred = (pos_prob > DEFAULT_THRESHOLD)
                test_result['pred'].append(pred_label)
                test_result['labels'].append(data['labels'])
                test_result['logits'].append(logits)

            with open(os.path.join(output_dir, 'pred.jsonl'), 'w') as f:
                for data in preds:
                    f.write(json.dumps(data)+'\n')
            
            labels = unique_labels(test_result['labels'], test_result['pred'])
            label_names = [cates[i] for i in labels]
            report = classification_report( test_result['labels'], test_result['pred'], target_names=label_names, output_dict=True)

            df = pd.DataFrame(report).transpose()
            df = pd.concat([df[:-3].sort_values("support",inplace=False, ascending=False),df[-3:]])
            print(df.to_markdown())

            # threshold
            thresholds = sorted([i/10 for i in range(0,10,2)],reverse=True)
            for threshold in thresholds:
                count = df['precision'][df['precision']>threshold].count()
                print(f'precition above {threshold},\t tree count: {count}')

            # topk accuracy
            print(len(cates))
            for k in range(1,6):
                topk_acc = top_k_accuracy_score(test_result['labels'], test_result['logits'], k=k, labels=list(range(len(cates))))
                print(f'top {k} acc: ', topk_acc)

main()
