from datasets import Dataset, DatasetDict
import pandas as pd
import os, json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from pathlib import Path
import torch.nn as nn
from transformers import BertTokenizerFast, RobertaTokenizerFast, AutoTokenizer
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from collections import defaultdict
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import classification_report, top_k_accuracy_score

from preprocess import *
import warnings
warnings.filterwarnings("ignore")

DEFAULT_THRESHOLD = 0.8
PER_DEVICE_BATCH_SIZE = 16

class CustomTrainer(Trainer):
    def __init__(self, sampler_type, cates, cls_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.sampler_type = sampler_type
        self.cates = cates
        self.cls_weight = cls_weight

def main():
    pretrained_model_name_or_path = '/mnt/petrelfs/baitianyi/dup/train_bert/train/DownloadModel/bert-base-uncased'
    output_dir = '/mnt/petrelfs/baitianyi/dup/train_bert/train/Bert-0.1'
    ptm_type = 'bert'
    data_folder = '/mnt/petrelfs/baitianyi/dup/train_bert/version2'
    dropout = 0.1
    weight_type = 'average'
    sample_type = ''
    
    # 只读取 test 数据
    cates, _, test_data, weight = read_data(data_folder, weight_type, seed=42)
    test_dataset = Dataset.from_list(test_data)

    # 创建 DatasetDict 只包含 test 数据集
    my_datasets = DatasetDict({
        'test': test_dataset,
    })

    # 根据 ptm_type 加载 tokenizer
    if ptm_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    elif ptm_type == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

    def preprocess_function(examples):
        tokenized_inputs = tokenizer(examples['content'], truncation=True, max_length=512)
        unique_topics = sorted(set(examples['selected_topic']))
        label_map = {label: i for i, label in enumerate(unique_topics)}
        tokenized_inputs['labels'] = [label_map[label] for label in examples['selected_topic']]
        return tokenized_inputs

    # 对 test 数据集进行预处理
    tokenized_datasets = my_datasets.map(preprocess_function, batched=True)

    # 根据 ptm_type 加载模型
    if ptm_type == 'bert':
        model = BertForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=len(cates), hidden_dropout_prob=dropout)
    elif ptm_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=len(cates), hidden_dropout_prob=dropout)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=len(cates), hidden_dropout_prob=dropout)

    # 设置训练参数，但这里只用于测试
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        disable_tqdm=True,
    )

    # 创建自定义的 Trainer 对象
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        cls_weight=weight,
        cates=cates,
        sampler_type=sample_type
    )

    print('==========================================test==========================================')
    # 进行测试并获取预测结果
    res = trainer.predict(tokenized_datasets['test'])

    test_result = defaultdict(list)
    preds = []
    for data, pred in zip(tokenized_datasets['test'], res[0]):
        logits = pred.tolist()
        pred_label = np.argmax(logits, axis=-1)
        preds.append({'pred': str(pred_label), 'logits': logits, 'labels': data['labels'], 'untokenized': tokenizer.decode(data['input_ids']), 'content': data['content']})

        pred_ = torch.tensor(pred)
        probs = torch.softmax(pred_, dim=-1)
        pos_prob = probs[1]
        pred = (pos_prob > DEFAULT_THRESHOLD)
        test_result['pred'].append(pred_label)
        test_result['labels'].append(data['labels'])
        test_result['logits'].append(logits)

    # 保存预测结果
    with open(os.path.join(output_dir, 'pred.jsonl'), 'w') as f:
        for data in preds:
            f.write(json.dumps(data) + '\n')

    # 生成分类报告
    labels = unique_labels(test_result['labels'], test_result['pred'])
    label_names = [cates[i] for i in labels]
    report = classification_report(test_result['labels'], test_result['pred'], target_names=label_names, output_dict=True)

    df = pd.DataFrame(report).transpose()
    df = pd.concat([df[:-3].sort_values("support", inplace=False, ascending=False), df[-3:]])
    print(df.to_markdown())

    # 输出top-k准确率
    for k in range(1, 6):
        topk_acc = top_k_accuracy_score(test_result['labels'], test_result['logits'], k=k, labels=list(range(len(cates))))
        print(f'Top {k} acc: ', topk_acc)

main()
